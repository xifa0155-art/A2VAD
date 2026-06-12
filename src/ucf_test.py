import re
from scipy.ndimage import gaussian_filter1d
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import ucf_option
from model import CLIPVAD
from utils.tools import get_prompt_text
from utils.ucf_dataset import UCFDataset


def _minmax_normalize(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val > 1e-5:
        return (scores - min_val) / (max_val - min_val)
    return scores

def _concat_valid_chunks(x, lengths_list, raw_length):
    valid = []
    for j, length in enumerate(lengths_list):
        valid.append(x[j, :length])
    out = np.concatenate(valid, axis=0)
    return out[:raw_length]

def _build_window_starts(length, visual_length, window_mode="chunk", window_stride=128):
    length = int(length)

    if length <= visual_length:
        return [0]

    if window_mode == "chunk":
        return list(range(0, length, visual_length))

    if window_mode == "sliding":
        if window_stride <= 0:
            raise ValueError("test-window-stride must be positive when using sliding mode.")

        max_start = length - visual_length
        starts = list(range(0, max_start + 1, window_stride))

        if starts[-1] != max_start:
            starts.append(max_start)

        return starts

    raise ValueError(f"Unknown test_window_mode: {window_mode}")

def _replace_crop_index(path_str, crop_idx):
    """
    Replace suffix like __5.npy with __{crop_idx}.npy.
    """
    return re.sub(r"__\d+\.npy$", f"__{crop_idx}.npy", path_str)


def _load_visual_audio_from_path(dataset, raw_path, raw_label):
    visual_path = dataset._resolve_visual_path(raw_path, raw_label)
    rel_path = dataset._extract_relative_path(raw_path)

    visual = np.load(visual_path).astype(np.float32, copy=False)
    raw_t = int(visual.shape[0])

    audio_path = dataset._resolve_audio_path(rel_path)
    if audio_path is not None:
        audio = np.load(audio_path).astype(np.float32, copy=False)
        audio = dataset._fit_audio_length(audio, raw_t, dataset.audio_dim)
    else:
        audio = np.zeros((raw_t, dataset.audio_dim), dtype=np.float32)

    return torch.from_numpy(visual).float(), torch.from_numpy(audio).float(), raw_t


def _get_test_crops(dataset, vid_idx, enable_ensemble=False, crop_num=10):
    """
    Return a list of (visual, audio, length, raw_length) for center crop or 10-crop.
    """
    raw_path, raw_label = dataset.samples[vid_idx]

    if not enable_ensemble:
        visual, audio, raw_t = _load_visual_audio_from_path(dataset, raw_path, raw_label)
        return [(visual, audio, raw_t, raw_t)]

    crops = []
    for crop_idx in range(crop_num):
        crop_path = _replace_crop_index(raw_path, crop_idx)
        try:
            visual, audio, raw_t = _load_visual_audio_from_path(dataset, crop_path, raw_label)
            crops.append((visual, audio, raw_t, raw_t))
        except FileNotFoundError:
            continue

    if len(crops) == 0:
        visual, audio, raw_t = _load_visual_audio_from_path(dataset, raw_path, raw_label)
        crops.append((visual, audio, raw_t, raw_t))

    return crops

def _infer_one_crop_scores(
    model,
    visual,
    audio,
    length,
    raw_length,
    visual_length,
    prompt_text,
    device,
    branch_mode="dual",
    test_window_mode="chunk",
    test_window_stride=128,
):
    starts = _build_window_starts(
        length,
        visual_length,
        window_mode=test_window_mode,
        window_stride=test_window_stride,
    )

    n_windows = len(starts)

    v_in = torch.zeros(
        n_windows,
        visual_length,
        visual.shape[-1],
        device=device,
        dtype=torch.float32,
    )
    a_in = torch.zeros(
        n_windows,
        visual_length,
        audio.shape[-1],
        device=device,
        dtype=torch.float32,
    )

    lengths_list = []

    for j, s in enumerate(starts):
        s = int(s)
        e = min(s + visual_length, length)
        act_l = e - s

        if act_l <= 0:
            act_l = 1
            s = max(0, min(s, length - 1))
            e = s + 1

        v_in[j, :act_l] = visual[s:e].to(device).float()
        a_in[j, :act_l] = audio[s:e].to(device).float()

        if act_l < visual_length:
            v_in[j, act_l:] = visual[e - 1].to(device).float()
            a_in[j, act_l:] = audio[e - 1].to(device).float()

        lengths_list.append(act_l)

    lengths_tensor = torch.tensor(lengths_list, dtype=torch.int32, device=device)

    text_feat, logits1, logits2 = model(v_in, a_in, prompt_text, lengths_tensor)

    aux_window = torch.sigmoid(logits1).squeeze(-1).detach().cpu().numpy()

    if branch_mode == "aux_only":
        semantic_window = aux_window.copy()
    else:
        probs = logits2.softmax(dim=-1).detach().cpu().numpy()
        semantic_window = 1.0 - probs[:, :, 0]

    aux_sum = np.zeros(raw_length, dtype=np.float32)
    sem_sum = np.zeros(raw_length, dtype=np.float32)
    count = np.zeros(raw_length, dtype=np.float32)

    for j, s in enumerate(starts):
        s = int(s)
        valid_len = min(lengths_list[j], raw_length - s)

        if valid_len <= 0:
            continue

        aux_sum[s:s + valid_len] += aux_window[j, :valid_len]
        sem_sum[s:s + valid_len] += semantic_window[j, :valid_len]
        count[s:s + valid_len] += 1.0

    count = np.maximum(count, 1.0)

    aux_scores = aux_sum / count
    semantic_scores = sem_sum / count

    return aux_scores, semantic_scores

def _select_score(pred_aux, pred_semantic, score_mode, score_alpha):
    pred_avg = (pred_aux + pred_semantic) / 2.0
    pred_weighted = score_alpha * pred_aux + (1.0 - score_alpha) * pred_semantic

    if score_mode == "aux":
        return pred_aux, "aux"
    if score_mode == "semantic":
        return pred_semantic, "semantic"
    if score_mode == "avg":
        return pred_avg, "avg"
    if score_mode == "weighted":
        return pred_weighted, f"weighted(alpha={score_alpha:.2f})"

                                                                             
                                                      
    return None, "max"


def test(
    model,
    test_dataloader,
    visual_length,
    prompt_text,
    gt,
    device,
    branch_mode="dual",
    stride=16,
    score_mode="semantic",
    score_alpha=0.5,
    test_crop_ensemble=False,
    test_crop_num=10,
    test_crop_agg="mean",
    smooth_sigma=0.0,
    smooth_target="snippet",
    video_minmax=False,
    test_window_mode="chunk",
    test_window_stride=128,
):
    model.to(device)
    model.eval()

    all_aux = []
    all_semantic = []

    with torch.no_grad():
        for vid_idx, item in enumerate(test_dataloader):
            crop_items = _get_test_crops(
                test_dataloader.dataset,
                vid_idx,
                enable_ensemble=test_crop_ensemble,
                crop_num=test_crop_num,
            )

            crop_aux_scores = []
            crop_semantic_scores = []

            for visual, audio, length, raw_length in crop_items:
                if raw_length <= 0:
                    continue

                aux_scores, semantic_scores = _infer_one_crop_scores(
                    model,
                    visual,
                    audio,
                    length,
                    raw_length,
                    visual_length,
                    prompt_text,
                    device,
                    branch_mode=branch_mode,
                    test_window_mode=test_window_mode,
                    test_window_stride=test_window_stride,
                )

                crop_aux_scores.append(aux_scores)
                crop_semantic_scores.append(semantic_scores)

            min_crop_len = min(len(x) for x in crop_aux_scores)
            crop_aux_scores = [x[:min_crop_len] for x in crop_aux_scores]
            crop_semantic_scores = [x[:min_crop_len] for x in crop_semantic_scores]

            crop_aux_arr = np.stack(crop_aux_scores, axis=0)
            crop_sem_arr = np.stack(crop_semantic_scores, axis=0)

            if test_crop_agg == "mean":
                aux_scores = np.mean(crop_aux_arr, axis=0)
                semantic_scores = np.mean(crop_sem_arr, axis=0)

            elif test_crop_agg == "max":
                aux_scores = np.max(crop_aux_arr, axis=0)
                semantic_scores = np.max(crop_sem_arr, axis=0)

            elif test_crop_agg == "top3":
                k = min(3, crop_aux_arr.shape[0])
                aux_scores = np.mean(np.sort(crop_aux_arr, axis=0)[-k:], axis=0)
                semantic_scores = np.mean(np.sort(crop_sem_arr, axis=0)[-k:], axis=0)

            elif test_crop_agg == "center_max_avg":
                center_idx = min(5, crop_aux_arr.shape[0] - 1)
                aux_scores = 0.5 * crop_aux_arr[center_idx] + 0.5 * np.max(crop_aux_arr, axis=0)
                semantic_scores = 0.5 * crop_sem_arr[center_idx] + 0.5 * np.max(crop_sem_arr, axis=0)

            else:
                raise ValueError(f"Unknown test_crop_agg: {test_crop_agg}")

            if smooth_sigma > 0 and smooth_target == "snippet":
                aux_scores = gaussian_filter1d(aux_scores, sigma=smooth_sigma)
                semantic_scores = gaussian_filter1d(semantic_scores, sigma=smooth_sigma)

                                                       
                                                                                   
                                                                                 
            if video_minmax:
                aux_scores = _minmax_normalize(aux_scores)
                semantic_scores = _minmax_normalize(semantic_scores)

            aux_frame = np.repeat(aux_scores, stride)
            semantic_frame = np.repeat(semantic_scores, stride)

            if smooth_sigma > 0 and smooth_target == "frame":
                aux_frame = gaussian_filter1d(aux_frame, sigma=smooth_sigma)
                semantic_frame = gaussian_filter1d(semantic_frame, sigma=smooth_sigma)

            all_aux.append(aux_frame)
            all_semantic.append(semantic_frame)

    pred_aux = np.concatenate(all_aux)
    pred_semantic = np.concatenate(all_semantic)
    pred_avg = (pred_aux + pred_semantic) / 2.0
    pred_weighted = score_alpha * pred_aux + (1.0 - score_alpha) * pred_semantic

    min_len = min(len(gt), len(pred_aux), len(pred_semantic))
    gt_eval = gt[:min_len].astype(np.float32)
    pred_aux = pred_aux[:min_len]
    pred_semantic = pred_semantic[:min_len]
    pred_avg = pred_avg[:min_len]
    pred_weighted = pred_weighted[:min_len]

    metrics = {}
    for name, pred in [
        ("aux", pred_aux),
        ("semantic", pred_semantic),
        ("avg", pred_avg),
        (f"weighted(alpha={score_alpha:.2f})", pred_weighted),
    ]:
        metrics[name] = {
            "auc": roc_auc_score(gt_eval, pred),
            "ap": average_precision_score(gt_eval, pred),
        }

    if score_mode == "max":
        selected_name = max(metrics, key=lambda k: metrics[k]["auc"])
        selected_auc = metrics[selected_name]["auc"]
        selected_ap = metrics[selected_name]["ap"]
    else:
        selected_pred, selected_name = _select_score(pred_aux, pred_semantic, score_mode, score_alpha)
        selected_auc = roc_auc_score(gt_eval, selected_pred)
        selected_ap = average_precision_score(gt_eval, selected_pred)

    print("")
    print("=" * 78)
    print("UCF-Crime Frame-level Evaluation")
    print("=" * 78)
    print(f"GT frames:       {len(gt)}")
    print(f"Pred frames:     {len(pred_aux)}")
    print(f"Eval frames:     {min_len}")
    for name, value in metrics.items():
        print(f"{name:<22} | AUC: {value['auc']:.6f} | AP: {value['ap']:.6f}")
    print(f"Selected ({selected_name}) AUC: {selected_auc:.6f} | AP: {selected_ap:.6f}")
    print("=" * 78)
    print("")

    return {
        "auc_aux": metrics["aux"]["auc"],
        "ap_aux": metrics["aux"]["ap"],
        "auc_semantic": metrics["semantic"]["auc"],
        "ap_semantic": metrics["semantic"]["ap"],
        "auc_avg": metrics["avg"]["auc"],
        "ap_avg": metrics["avg"]["ap"],
        "auc_weighted": metrics[f"weighted(alpha={score_alpha:.2f})"]["auc"],
        "ap_weighted": metrics[f"weighted(alpha={score_alpha:.2f})"]["ap"],
        "selected_auc": selected_auc,
        "selected_ap": selected_ap,
        "selected_score": selected_name,
    }




def main():
    args = ucf_option.parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = ucf_option.get_ucf_label_map(args.ucf_label_mode)
    args.classes_num = len(label_map)
    prompt_text = get_prompt_text(label_map)

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
        crop_fusion=args.crop_fusion,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    gt = np.load(args.gt_path)

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
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)

    print(f"[Info] Loaded model from {args.model_path}")
    print(f"[Info] label_mode={args.ucf_label_mode}, classes={args.classes_num}, score_mode={args.score_mode}")
    test(
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
        test_crop_ensemble=args.test_crop_ensemble,
        test_crop_num=args.test_crop_num,
        test_crop_agg=args.test_crop_agg,
        smooth_sigma=args.smooth_sigma,
        smooth_target=args.smooth_target,
        video_minmax=getattr(args, "video_minmax", False),
        test_window_mode=args.test_window_mode,
        test_window_stride=args.test_window_stride,
    )


if __name__ == "__main__":
    main()
