import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option


EPS = 1e-8


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Binary focal loss for semantic prediction scores.
    inputs: raw logits
    targets: binary / multi-hot targets
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)

    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def CLASM(logits, labels, lengths, device):
    """
    MIL-based semantic alignment loss.
    logits: [B, T, C]
    labels: [B, C], multi-hot category labels
    lengths: valid snippet lengths
    """
    instance_logits_list = []

    labels = labels / (torch.sum(labels, dim=1, keepdim=True) + EPS)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        k = min(valid_len, max(3, int(valid_len // 16 + 1)))

        tmp, _ = torch.topk(
            logits[i, 0:valid_len],
            k=k,
            largest=True,
            dim=0
        )
        instance_logits_list.append(torch.mean(tmp, dim=0))

    instance_logits = torch.stack(instance_logits_list)

    milloss = -torch.mean(
        torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0
    )

    return milloss


def CLAS2(logits, labels, lengths, device):
    """
    MIL-based auxiliary binary classification loss.
    logits: [B, T, 1]
    labels: [B, C], first class is normal
    """
    instance_logits_list = []

    binary_labels = 1 - labels[:, 0].reshape(labels.shape[0])
    binary_labels = binary_labels.to(device)

    scores = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(scores.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        k = min(valid_len, max(3, int(valid_len // 16 + 1)))

        tmp, _ = torch.topk(
            scores[i, 0:valid_len],
            k=k,
            largest=True
        )
        instance_logits_list.append(torch.mean(tmp))

    instance_logits = torch.stack(instance_logits_list)

    clsloss = F.binary_cross_entropy(instance_logits, binary_labels)

    return clsloss


def SmoothnessLoss(logits, lengths, device):
    """
    Temporal smoothness loss on auxiliary snippet-level anomaly scores.
    logits: [B, T, 1]
    """
    scores = torch.sigmoid(logits).squeeze(-1)
    loss = torch.zeros(1, device=device)

    for i in range(scores.shape[0]):
        valid_len = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])

        if valid_len < 2:
            continue

        score_vid = scores[i, :valid_len]
        loss += torch.sum((score_vid[1:] - score_vid[:-1]) ** 2)

    return loss / scores.shape[0]


def compute_text_norm_loss(text_features, device):
    """
    Encourage normal-class text feature and anomaly-class text features
    to be less collapsed.
    """
    l_norm = torch.zeros(1, device=device)

    text_feature_norm = F.normalize(text_features, p=2, dim=-1)
    normal_vec = text_feature_norm[0]

    for j in range(1, text_features.shape[0]):
        l_norm += torch.abs(torch.dot(normal_vec, text_feature_norm[j]))

    l_norm = l_norm / (text_features.shape[0] - 1)

    return l_norm


def run_eval(model, test_loader, args, prompt_text, gt, gtsegments, gtlabels, device):
    """
    Evaluation wrapper. Pass branch_mode to keep training and testing consistent.
    """
    return test(
        model,
        test_loader,
        args.visual_length,
        prompt_text,
        gt,
        gtsegments,
        gtlabels,
        device,
        branch_mode=args.branch_mode
    )


def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    first_label = list(label_map.values())[0]
    if "normal" not in first_label.lower():
        raise ValueError("[Critical] Label map 的第一项必须包含 Normal 语义。")

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    params_decay = []
    params_no_decay = []

    no_decay_names = [
        "linear.weight",
        "ctx",
        "bias",
        "ln_1",
        "ln_2"
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_names):
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": params_decay, "weight_decay": 1e-4},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=float(args.lr)
    )

    scheduler = MultiStepLR(
        optimizer,
        args.scheduler_milestones,
        gamma=float(args.scheduler_rate)
    )

    prompt_text = get_prompt_text(label_map)

    print(
        f"[Config] branch_mode={args.branch_mode}, "
        f"lambda_align={args.lambda_align}, "
        f"lambda_smooth={args.lambda_smooth}, "
        f"batch_size={args.batch_size}, "
        f"lr={args.lr}"
    )

    ap_best = 0.0
    start_epoch = 0

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"[Info] Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        ap_best = checkpoint.get("ap", 0.0)

        print(f"[Info] Resuming from epoch {start_epoch}, best AP: {ap_best:.4f}")

    else:
        print("[Info] Starting training from scratch (Strict Expert Mode)...")
        print("--> [Sanity Check] Checking Zero-Shot Baseline (Epoch 0)...")

        model.eval()
        with torch.no_grad():
            AUC, AP, mAP = run_eval(
                model,
                test_loader,
                args,
                prompt_text,
                gt,
                gtsegments,
                gtlabels,
                device
            )

        print(f"Epoch 0 (Untrained) Result | AUC: {AUC:.6f} | AP: {AP:.6f}")
        print("---------------------------------------------------------------")

    for e in range(start_epoch, args.max_epoch):
        model.train()

        metrics = {
            "cls": 0.0,
            "align": 0.0,
            "smooth": 0.0,
            "norm": 0.0,
            "total": 0.0,
        }

        log_interval = 20

        for i, item in enumerate(train_loader):
                                                             
                                                         
                                    
                                                                     
            visual_feat = item[0].to(device)
            audio_feat = item[1].to(device)
            text_labels_raw = item[2]
            feat_lengths = item[3].to(device)

            text_labels_tensor = get_batch_label(
                text_labels_raw,
                prompt_text,
                label_map
            ).to(device)

            text_features, logits1, logits2 = model(
                visual_feat,
                audio_feat,
                prompt_text,
                feat_lengths
            )

                                   
            l_bce = CLAS2(
                logits1,
                text_labels_tensor,
                feat_lengths,
                device
            )

                             
            l_smooth = SmoothnessLoss(
                logits1,
                feat_lengths,
                device
            )

                                                
            if args.branch_mode == "aux_only":
                l_align = torch.zeros(1, device=device)
                l_norm = torch.zeros(1, device=device)

                loss = l_bce + args.lambda_smooth * l_smooth

            else:
                                         
                l_nce = CLASM(
                    logits2,
                    text_labels_tensor,
                    feat_lengths,
                    device
                )

                video_logits = torch.zeros(0, device=device)

                for b in range(logits2.shape[0]):
                    valid_len = int(feat_lengths[b].item())
                    k = min(valid_len, max(3, int(valid_len // 16 + 1)))

                    tmp, _ = torch.topk(
                        logits2[b, :valid_len],
                        k=k,
                        dim=0
                    )

                    video_logits = torch.cat(
                        [video_logits, torch.mean(tmp, dim=0, keepdim=True)],
                        dim=0
                    )

                l_focal = sigmoid_focal_loss(
                    video_logits,
                    text_labels_tensor
                )

                l_align = (l_nce + l_focal) / 2.0

                                                            
                l_norm = compute_text_norm_loss(
                    text_features,
                    device
                )

                loss = (
                    l_bce
                    + args.lambda_align * l_align
                    + args.lambda_smooth * l_smooth
                    + (l_norm * 1e-4)
                )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            metrics["cls"] += l_bce.item()
            metrics["align"] += l_align.item()
            metrics["smooth"] += l_smooth.item()
            metrics["norm"] += l_norm.item()
            metrics["total"] += loss.item()

            if (i + 1) % log_interval == 0:
                denom = i + 1

                print(
                    f"Epoch: {e + 1}/{args.max_epoch} | "
                    f"Iter: {i + 1}/{len(train_loader)} | "
                    f"Cls: {metrics['cls'] / denom:.4f} | "
                    f"Align: {metrics['align'] / denom:.4f} | "
                    f"Smooth: {metrics['smooth'] / denom:.4f} | "
                    f"Norm: {metrics['norm'] / denom:.4f} | "
                    f"Total: {metrics['total'] / denom:.4f}"
                )

        scheduler.step()

        print(f"--> Testing at Epoch {e + 1}...")

        model.eval()
        with torch.no_grad():
            AUC, AP, mAP = run_eval(
                model,
                test_loader,
                args,
                prompt_text,
                gt,
                gtsegments,
                gtlabels,
                device
            )

        print(f"Epoch {e + 1} Result | AUC: {AUC:.4f} | AP: {AP:.4f}")

        if AP > ap_best:
            ap_best = AP

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ap": ap_best,
                    "epoch": e + 1,
                    "branch_mode": args.branch_mode,
                    "lambda_align": args.lambda_align,
                    "lambda_smooth": args.lambda_smooth,
                },
                args.checkpoint_path
            )

            print(f"--> [Saved] Best Model (AP: {ap_best:.4f})")

    if os.path.exists(args.checkpoint_path):
        print("Training Finished. Saving final best model.")

        best_ckpt = torch.load(args.checkpoint_path, map_location=device)
        torch.save(best_ckpt["model_state_dict"], args.model_path)

        print(f"[Saved] Final model to {args.model_path}")
    else:
        print("[Warning] No checkpoint was saved. Please check training results.")


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = xd_option.parser.parse_args()

    setup_seed(args.seed)

    label_map = xd_option.xd_label_map

    ensure_parent_dir(args.checkpoint_path)
    ensure_parent_dir(args.model_path)

    train_dataset = XDDataset(
        args.visual_length,
        args.train_list,
        False,
        label_map
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataset = XDDataset(
        args.visual_length,
        args.test_list,
        True,
        label_map
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
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
        branch_mode=args.branch_mode
    )

    print(
        f"Start training on {device} "
        f"(Batch Size: {args.batch_size}, LR: {args.lr})"
    )

    train(
        model,
        train_loader,
        test_loader,
        args,
        label_map,
        device
    )