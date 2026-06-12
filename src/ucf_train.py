import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import ucf_option
from model import CLIPVAD
from ucf_test import test as ucf_test
from utils.tools import get_prompt_text, get_batch_label
from utils.ucf_dataset import UCFDataset


EPS = 1e-8


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_topk(valid_len, divisor):
    valid_len = int(valid_len)
    divisor = max(1, int(divisor))
    return min(valid_len, max(1, int(valid_len // divisor + 1)))


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def CLASM(logits, labels, lengths, device, topk_divisor):
    instance_logits_list = []
    labels = labels / (torch.sum(labels, dim=1, keepdim=True) + EPS)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        valid_len = max(1, min(valid_len, logits.shape[1]))
        k = get_topk(valid_len, topk_divisor)

        tmp, _ = torch.topk(logits[i, :valid_len], k=k, largest=True, dim=0)
        instance_logits_list.append(torch.mean(tmp, dim=0))

    instance_logits = torch.stack(instance_logits_list)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def CLAS2(logits, labels, lengths, device, topk_divisor):
    instance_logits_list = []
    binary_labels = 1.0 - labels[:, 0].reshape(labels.shape[0])
    binary_labels = binary_labels.to(device)

    scores = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(scores.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        valid_len = max(1, min(valid_len, scores.shape[1]))
        k = get_topk(valid_len, topk_divisor)

        tmp, _ = torch.topk(scores[i, :valid_len], k=k, largest=True)
        instance_logits_list.append(torch.mean(tmp))

    instance_logits = torch.stack(instance_logits_list)
    return F.binary_cross_entropy(instance_logits, binary_labels)


def SmoothnessLoss(logits, lengths, device):
    scores = torch.sigmoid(logits).squeeze(-1)
    loss = torch.zeros(1, device=device)

    for i in range(scores.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        valid_len = max(1, min(valid_len, scores.shape[1]))

        if valid_len < 2:
            continue

        score_vid = scores[i, :valid_len]
        loss += torch.sum((score_vid[1:] - score_vid[:-1]) ** 2)

    return loss / scores.shape[0]


def RankingLoss(logits, labels, lengths, device, topk_divisor, margin=1.0):
    """
    Pairwise MIL ranking loss:
    anomaly video top-k score should be higher than normal video top-k score.

    logits: [B, T, 1]
    labels: [B, C], first class is Normal
    """
    scores = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    video_scores = []

    for i in range(scores.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        valid_len = max(1, min(valid_len, scores.shape[1]))
        k = get_topk(valid_len, topk_divisor)

        topk_score, _ = torch.topk(scores[i, :valid_len], k=k, largest=True)
        video_scores.append(torch.mean(topk_score))

    video_scores = torch.stack(video_scores)

    is_anomaly = (labels[:, 0] < 0.5).to(device)
    anomaly_scores = video_scores[is_anomaly]
    normal_scores = video_scores[~is_anomaly]

    if anomaly_scores.numel() == 0 or normal_scores.numel() == 0:
        return torch.zeros(1, device=device)

                                                     
    diff = anomaly_scores.view(-1, 1) - normal_scores.view(1, -1)
    loss = torch.clamp(margin - diff, min=0.0)
    return loss.mean()

def compute_text_norm_loss(text_features, device):
    text_feature_norm = F.normalize(text_features, p=2, dim=-1)
    normal_vec = text_feature_norm[0]

    loss = torch.zeros(1, device=device)
    for j in range(1, text_features.shape[0]):
        loss += torch.abs(torch.dot(normal_vec, text_feature_norm[j]))

    return loss / max(1, text_features.shape[0] - 1)


def build_optimizer(model, lr):
    params_decay = []
    params_no_decay = []

    no_decay_names = ["linear.weight", "ctx", "bias", "ln_1", "ln_2"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_names):
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": params_decay, "weight_decay": 1e-4},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )


def run_one_batch(model, batch, args, label_map, prompt_text, device):
    visual_feat = batch[0].to(device)
    audio_feat = batch[1].to(device)
    text_labels_raw = batch[2]
    feat_lengths = batch[3].to(device)

    text_labels_tensor = get_batch_label(text_labels_raw, prompt_text, label_map).to(device)

    text_features, logits1, logits2 = model(
        visual_feat,
        audio_feat,
        prompt_text,
        feat_lengths,
    )

    l_bce = CLAS2(logits1, text_labels_tensor, feat_lengths, device, args.topk_divisor)
    l_rank = RankingLoss(
        logits1,
        text_labels_tensor,
        feat_lengths,
        device,
        args.topk_divisor,
        margin=0.5,
    )

    l_align = torch.zeros(1, device=device)
    l_smooth = torch.zeros(1, device=device)
    l_norm = torch.zeros(1, device=device)
    l_focal = torch.zeros(1, device=device)

    if args.loss_mode == "aux_only" or args.branch_mode == "aux_only":
        loss = l_bce

    elif args.loss_mode == "simple":
        l_align = CLASM(logits2, text_labels_tensor, feat_lengths, device, args.topk_divisor)
        loss = l_bce + args.lambda_align * l_align + args.lambda_rank * l_rank

    elif args.loss_mode == "xd":
        l_smooth = SmoothnessLoss(logits1, feat_lengths, device)

        l_nce = CLASM(logits2, text_labels_tensor, feat_lengths, device, args.topk_divisor)

        video_logits = []
        for b in range(logits2.shape[0]):
            valid_len = int(feat_lengths[b].item())
            valid_len = max(1, min(valid_len, logits2.shape[1]))
            k = get_topk(valid_len, args.topk_divisor)

            tmp, _ = torch.topk(logits2[b, :valid_len], k=k, dim=0)
            video_logits.append(torch.mean(tmp, dim=0))

        video_logits = torch.stack(video_logits)
        l_focal = sigmoid_focal_loss(video_logits, text_labels_tensor)
        l_align = (l_nce + l_focal) / 2.0

        l_norm = compute_text_norm_loss(text_features, device)

        loss = (
            l_bce
            + args.lambda_align * l_align
            + args.lambda_smooth * l_smooth
            + args.lambda_norm * l_norm
        )

    else:
        raise ValueError(f"Unknown loss_mode: {args.loss_mode}")

    return loss, {
        "cls": float(l_bce.item()),
        "align": float(l_align.item()),
        "rank": float(l_rank.item()),
        "smooth": float(l_smooth.item()),
        "norm": float(l_norm.item()),
        "focal": float(l_focal.item()),
        "total": float(loss.item()),
    }


def make_balanced_train_loader(args, label_map):
    normal_dataset = UCFDataset(
        args.visual_length,
        args.train_list,
        is_test=False,
        label_map=label_map,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        audio_dim=args.audio_dim,
        filter_label="normal",
        label_mode=args.ucf_label_mode,
        train_crop=args.train_crop,
        feature_norm=args.feature_norm,
        train_sampling=args.train_sampling,
        crop_fusion=args.crop_fusion,
    )
    anomaly_dataset = UCFDataset(
        args.visual_length,
        args.train_list,
        is_test=False,
        label_map=label_map,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        audio_dim=args.audio_dim,
        filter_label="anomaly",
        label_mode=args.ucf_label_mode,
        train_crop=args.train_crop,
        feature_norm=args.feature_norm,
        train_sampling=args.train_sampling,
        crop_fusion=args.crop_fusion,
    )

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class BalancedLoader:
        def __init__(self, n_loader, a_loader):
            self.n_loader = n_loader
            self.a_loader = a_loader

        def __len__(self):
            return min(len(self.n_loader), len(self.a_loader))

        def __iter__(self):
            for normal_batch, anomaly_batch in zip(self.n_loader, self.a_loader):
                visual = torch.cat([normal_batch[0], anomaly_batch[0]], dim=0)
                audio = torch.cat([normal_batch[1], anomaly_batch[1]], dim=0)
                labels = list(normal_batch[2]) + list(anomaly_batch[2])
                actual_len = torch.cat([normal_batch[3], anomaly_batch[3]], dim=0)
                raw_len = torch.cat([normal_batch[4], anomaly_batch[4]], dim=0)
                sample_idx = torch.cat([normal_batch[5], anomaly_batch[5]], dim=0)
                yield visual, audio, labels, actual_len, raw_len, sample_idx

    print(
        f"[Data] Balanced training: normal={len(normal_dataset)}, "
        f"anomaly={len(anomaly_dataset)}, steps/epoch={min(len(normal_loader), len(anomaly_loader))}, "
        f"train_crop={args.train_crop}, label_mode={args.ucf_label_mode}"
    )
    return BalancedLoader(normal_loader, anomaly_loader)


def make_single_train_loader(args, label_map):
    train_dataset = UCFDataset(
        args.visual_length,
        args.train_list,
        is_test=False,
        label_map=label_map,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        audio_dim=args.audio_dim,
        filter_label=None,
        label_mode=args.ucf_label_mode,
        train_crop=args.train_crop,
        feature_norm=args.feature_norm,
        train_sampling=args.train_sampling,
        crop_fusion=args.crop_fusion,
    )

    print(
        f"[Data] Single training loader: samples={len(train_dataset)}, "
        f"train_crop={args.train_crop}, label_mode={args.ucf_label_mode}"
    )
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def train(model, train_loader, test_loader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)

    optimizer = build_optimizer(model, args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=float(args.scheduler_rate))
    prompt_text = get_prompt_text(label_map)

    print(
        f"[Config] UCF-V2 | label_mode={args.ucf_label_mode}, classes={len(label_map)}, "
        f"loss_mode={args.loss_mode}, topk_divisor={args.topk_divisor}, "
        f"score_mode={args.score_mode}, alpha={args.score_alpha}, "
        f"layers={args.visual_layers}, attn_window={args.attn_window}, "
        f"batch_size={args.batch_size}, lr={args.lr}"
    )

    auc_best = 0.0
    start_epoch = 0

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"[Info] Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        auc_best = float(checkpoint.get("auc", checkpoint.get("ap", 0.0)))
        print(f"[Info] Resuming from epoch {start_epoch}, best AUC: {auc_best:.6f}")
    else:
        print("[Info] Starting UCF-V2 training from scratch.")
        print("--> [Sanity Check] Epoch 0 evaluation...")
        model.eval()
        with torch.no_grad():
            eval_result = ucf_test(
                model,
                test_loader,
                args.visual_length,
                prompt_text,
                gt,
                device,
                branch_mode=args.branch_mode,
                stride=args.stride,
                score_mode=args.score_mode,
                score_alpha=args.score_alpha,
            )
        print(f"Epoch 0 selected AUC: {eval_result['selected_auc']:.6f}")
        print("-" * 78)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()

        meters = {k: 0.0 for k in ["cls", "align", "rank", "smooth", "norm", "focal", "total"]}

        for i, batch in enumerate(train_loader):
            loss, batch_metrics = run_one_batch(model, batch, args, label_map, prompt_text, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in meters:
                meters[k] += batch_metrics[k]

            if (i + 1) % 20 == 0:
                denom = i + 1
                print(
                    f"Epoch: {epoch + 1}/{args.max_epoch} | "
                    f"Iter: {i + 1}/{len(train_loader)} | "
                    f"Cls: {meters['cls'] / denom:.4f} | "
                    f"Align: {meters['align'] / denom:.4f} | "
                    f"Rank: {meters['rank'] / denom:.4f} | "
                    f"Smooth: {meters['smooth'] / denom:.4f} | "
                    f"Norm: {meters['norm'] / denom:.4f} | "
                    f"Focal: {meters['focal'] / denom:.4f} | "
                    f"Total: {meters['total'] / denom:.4f}"
                )

            if args.eval_interval > 0 and (i + 1) % args.eval_interval == 0:
                print(f"--> Mid-epoch evaluation: Epoch {epoch + 1}, Iter {i + 1}...")
                model.eval()
                with torch.no_grad():
                    eval_result = ucf_test(
                        model,
                        test_loader,
                        args.visual_length,
                        prompt_text,
                        gt,
                        device,
                        branch_mode=args.branch_mode,
                        stride=args.stride,
                        score_mode=args.score_mode,
                        score_alpha=args.score_alpha,
                    )

                current_auc = float(eval_result["selected_auc"])
                print(
                    f"Epoch {epoch + 1} Iter {i + 1} selected AUC: "
                    f"{current_auc:.6f}"
                )

                if current_auc > auc_best:
                    auc_best = current_auc
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "auc": auc_best,
                        "epoch": epoch + 1,
                        "iter": i + 1,
                        "branch_mode": args.branch_mode,
                        "score_mode": args.score_mode,
                        "score_alpha": args.score_alpha,
                        "label_mode": args.ucf_label_mode,
                        "loss_mode": args.loss_mode,
                        "topk_divisor": args.topk_divisor,
                        "train_crop": args.train_crop,
                        "visual_layers": args.visual_layers,
                        "attn_window": args.attn_window,
                        "lr": args.lr,
                    }
                    torch.save(checkpoint, args.checkpoint_path)
                    print(
                        f"--> [Saved] Best UCF-V2 mid-epoch model by AUC: "
                        f"{auc_best:.6f}"
                    )

                model.train()

        scheduler.step()

        print(f"--> Evaluating UCF-V2 at Epoch {epoch + 1}...")
        model.eval()
        with torch.no_grad():
            eval_result = ucf_test(
                model,
                test_loader,
                args.visual_length,
                prompt_text,
                gt,
                device,
                branch_mode=args.branch_mode,
                stride=args.stride,
                score_mode=args.score_mode,
                score_alpha=args.score_alpha,
            )

        current_auc = float(eval_result["selected_auc"])
        print(f"Epoch {epoch + 1} selected AUC: {current_auc:.6f}")

        if current_auc > auc_best:
            auc_best = current_auc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc": auc_best,
                "epoch": epoch + 1,
                "branch_mode": args.branch_mode,
                "score_mode": args.score_mode,
                "score_alpha": args.score_alpha,
                "label_mode": args.ucf_label_mode,
                "loss_mode": args.loss_mode,
                "topk_divisor": args.topk_divisor,
                "train_crop": args.train_crop,
                "visual_layers": args.visual_layers,
                "attn_window": args.attn_window,
                "lr": args.lr,
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"--> [Saved] Best UCF-V2 model by AUC: {auc_best:.6f}")

    if os.path.exists(args.checkpoint_path):
        print("Training finished. Saving final best model state_dict.")
        best_ckpt = torch.load(args.checkpoint_path, map_location=device)
        torch.save(best_ckpt["model_state_dict"], args.model_path)
        print(f"[Saved] Final model to {args.model_path}")
    else:
        print("[Warning] No checkpoint was saved. Please check training logs.")


def main():
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = ucf_option.get_ucf_label_map(args.ucf_label_mode)
    args.classes_num = len(label_map)

    first_label = list(label_map.values())[0]
    if "normal" not in first_label.lower():
        raise ValueError("[Critical] The first class in UCF label map must be Normal.")

    ensure_parent_dir(args.checkpoint_path)
    ensure_parent_dir(args.model_path)

    if args.balanced_batch:
        train_loader = make_balanced_train_loader(args, label_map)
    else:
        train_loader = make_single_train_loader(args, label_map)

    test_dataset = UCFDataset(
        args.visual_length,
        args.test_list,
        is_test=True,
        label_map=label_map,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        audio_dim=args.audio_dim,
        filter_label=None,
        label_mode=args.ucf_label_mode,
        train_crop=args.train_crop,
        feature_norm=args.feature_norm,
        train_sampling=args.train_sampling,
        crop_fusion=args.crop_fusion,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
        branch_mode=args.branch_mode,
    )

    print(f"Start UCF-V2 training on {device}")
    train(model, train_loader, test_loader, args, label_map, device)


if __name__ == "__main__":
    main()
