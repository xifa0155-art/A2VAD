import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import xd_option
from utils.dataset import XDDataset
from utils.tools import get_prompt_text
from model import CLIPVAD
from sklearn.metrics import average_precision_score, roc_auc_score
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d


class SOTAMAPCalculator:
    def __init__(self, iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
        self.iou_thresholds = iou_thresholds

    def compute_iou(self, box1, box2):
        b1_s, b1_e = float(box1[0]), float(box1[1])
        b2_s, b2_e = float(box2[0]), float(box2[1])

        inter_start = max(b1_s, b2_s)
        inter_end = min(b1_e, b2_e)
        inter_len = max(0.0, inter_end - inter_start)

        union_len = (b1_e - b1_s) + (b2_e - b2_s) - inter_len
        return inter_len / (union_len + 1e-8)

    def soft_nms(self, boxes, scores, sigma=0.5, thresh=0.001):
        if boxes.numel() == 0:
            return [], []

        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)

        n = boxes.shape[0]
        indexes = torch.arange(n, dtype=torch.long).to(boxes.device)

        scores, idx = scores.sort(descending=True)
        boxes = boxes[idx]
        indexes = indexes[idx]

        keep = []
        keep_scores = []

        while boxes.shape[0] > 0:
            best_box = boxes[0]
            keep.append(indexes[0].item())
            keep_scores.append(scores[0].item())

            if boxes.shape[0] == 1:
                break

            other_boxes = boxes[1:]

            b1_s, b1_e = best_box[0], best_box[1]
            b2_s, b2_e = other_boxes[:, 0], other_boxes[:, 1]

            inter_s = torch.max(b1_s, b2_s)
            inter_e = torch.min(b1_e, b2_e)
            inter_len = torch.clamp(inter_e - inter_s, min=0.0)

            union_len = (b1_e - b1_s) + (b2_e - b2_s) - inter_len
            ious = inter_len / (union_len + 1e-8)

            decay_factor = torch.exp(-(ious ** 2) / sigma)

            scores = scores[1:] * decay_factor
            boxes = boxes[1:]
            indexes = indexes[1:]

            mask = scores > thresh
            boxes = boxes[mask]
            scores = scores[mask]
            indexes = indexes[mask]

            if scores.numel() > 0:
                scores, idx = scores.sort(descending=True)
                boxes = boxes[idx]
                indexes = indexes[idx]

        return keep, keep_scores

    def run(self, all_class_probs, gtsegments, gtlabels):
        all_gts = {}
        lbl_map = {
            'A': 0,
            'B1': 1,
            'B2': 2,
            'B4': 3,
            'B5': 4,
            'B6': 5,
            'G': 6,
        }

        for vid_idx, (segments, labels) in enumerate(zip(gtsegments, gtlabels)):
            if isinstance(labels, (int, np.integer)):
                labels = [labels] * len(segments)

            for seg, lbl in zip(segments, labels):
                final_lbl = lbl_map.get(lbl, lbl) if isinstance(lbl, str) else int(lbl)

                if final_lbl <= 0:
                    continue

                all_gts.setdefault(final_lbl, {}).setdefault(vid_idx, []).append(
                    [float(seg[0]), float(seg[1])]
                )

        all_preds = {}
        thresholds = np.linspace(0.05, 0.85, 30)
        max_merge_gap = 48.0
        min_duration = 16.0

        for vid_idx, probs in enumerate(all_class_probs):
            for cls in range(1, probs.shape[1]):
                scores = probs[:, cls]
                scores = np.power(scores, 0.7)

                proposals = []

                for thresh in thresholds:
                    binary = scores > thresh
                    padded = np.pad(binary, (1, 1), 'constant')
                    diff = np.diff(padded.astype(int))

                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]

                    seg_starts = starts * 16.0
                    seg_ends = ends * 16.0

                    if len(seg_starts) == 0:
                        continue

                    merged_segs = []
                    curr_s, curr_e = seg_starts[0], seg_ends[0]

                    for k in range(1, len(seg_starts)):
                        next_s, next_e = seg_starts[k], seg_ends[k]

                        if next_s - curr_e < max_merge_gap:
                            curr_e = next_e
                        else:
                            merged_segs.append([curr_s, curr_e])
                            curr_s, curr_e = next_s, next_e

                    merged_segs.append([curr_s, curr_e])

                    for s, e in merged_segs:
                        duration = e - s

                        if duration < min_duration:
                            continue

                        snip_s = int(s / 16)
                        snip_e = int(e / 16)
                        snip_e = min(snip_e, len(scores))

                        if snip_s >= snip_e:
                            continue

                        inner = scores[snip_s:snip_e]
                        score_inner = np.mean(inner)

                        len_half = (snip_e - snip_s) // 2
                        outer_s = max(0, snip_s - len_half)
                        outer_e = min(len(scores), snip_e + len_half)

                        outer_vals = []

                        if snip_s > outer_s:
                            outer_vals.append(np.mean(scores[outer_s:snip_s]))

                        if outer_e > snip_e:
                            outer_vals.append(np.mean(scores[snip_e:outer_e]))

                        score_outer = np.mean(outer_vals) if outer_vals else 0.0
                        final_score = score_inner + (score_inner - score_outer)

                        proposals.append([s, e, final_score])

                if len(proposals) == 0:
                    continue

                props_tensor = torch.tensor(proposals, dtype=torch.float32)

                if props_tensor.dim() == 1:
                    props_tensor = props_tensor.unsqueeze(0)

                boxes = props_tensor[:, 0:2].to(torch.device('cpu'))
                scores_t = props_tensor[:, 2].to(torch.device('cpu'))

                keep_idx, keep_scores = self.soft_nms(
                    boxes,
                    scores_t,
                    sigma=0.5,
                    thresh=0.001,
                )

                keep_idx = keep_idx[:200]

                for i, idx in enumerate(keep_idx):
                    final_s = boxes[idx, 0].item()
                    final_e = boxes[idx, 1].item()
                    final_score = keep_scores[i]

                    all_preds.setdefault(cls, []).append(
                        [vid_idx, float(final_s), float(final_e), float(final_score)]
                    )

        map_iou = {}

        for thresh_iou in self.iou_thresholds:
            aps = []
            all_classes = sorted(list(set(all_gts.keys())))

            for cls_id in all_classes:
                preds = sorted(
                    all_preds.get(cls_id, []),
                    key=lambda x: x[3],
                    reverse=True,
                )

                gts_flat = [
                    [v, s, e, False]
                    for v, list_s in all_gts.get(cls_id, {}).items()
                    for s, e in list_s
                ]

                if not gts_flat:
                    continue

                if not preds:
                    aps.append(0.0)
                    continue

                tp = np.zeros(len(preds))
                fp = np.zeros(len(preds))

                for p_idx, pred in enumerate(preds):
                    p_vid = pred[0]
                    p_box = pred[1:3]

                    best_iou = 0.0
                    best_gi = -1

                    for gi, gt_item in enumerate(gts_flat):
                        if gt_item[0] == p_vid and not gt_item[3]:
                            iou = self.compute_iou(p_box, gt_item[1:3])

                            if iou > best_iou:
                                best_iou = iou
                                best_gi = gi

                    if best_iou >= thresh_iou:
                        tp[p_idx] = 1
                        gts_flat[best_gi][3] = True
                    else:
                        fp[p_idx] = 1

                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)

                recall = tp_cumsum / len(gts_flat)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

                ap = 0.0

                for t in np.arange(0, 1.01, 0.01):
                    mask = recall >= t
                    p = np.max(precision[mask]) if np.any(mask) else 0.0
                    ap += p

                aps.append(ap / 101.0)

            map_iou[f'@{thresh_iou}'] = np.mean(aps) if aps else 0.0

        return map_iou








def test(
    model,
    testdataloader,
    visual_length,
    prompt_text,
    gt,
    gtsegments,
    gtlabels,
    device,
    branch_mode='dual',
):
    model.eval()

    if isinstance(prompt_text, torch.Tensor):
        prompt_text = prompt_text.to(device)

    pred_scores_binary = []
    all_class_probs = []

    gt_offset = 0

    with torch.no_grad():
        for vid_idx, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            audio = item[1].squeeze(0)
            length = item[3].item()
            raw_length = item[4].item()

            n_chunks = int(np.ceil(length / visual_length))

            v_in = torch.zeros(
                n_chunks,
                visual_length,
                visual.shape[-1],
                device=device,
            )
            a_in = torch.zeros(
                n_chunks,
                visual_length,
                audio.shape[-1],
                device=device,
            )

            lengths_list = []

            for j in range(n_chunks):
                s = j * visual_length
                e = min((j + 1) * visual_length, length)
                act_l = e - s

                v_in[j, :act_l] = visual[s:e].to(device)
                a_in[j, :act_l] = audio[s:e].to(device)

                if act_l < visual_length:
                    v_in[j, act_l:] = visual[e - 1].to(device)
                    a_in[j, act_l:] = audio[e - 1].to(device)

                lengths_list.append(act_l)

            ret = model(
                v_in,
                a_in,
                prompt_text,
                torch.tensor(lengths_list).int().to(device),
                return_features=False,
            )

            text_feat, logits1, logits2 = ret

            if branch_mode == 'aux_only':
                score_snippet = torch.sigmoid(logits1).squeeze(-1).cpu().numpy()

                valid_scores_list = [
                    score_snippet[j, :lengths_list[j]]
                    for j in range(len(lengths_list))
                ]
                scores_concatenated = np.concatenate(valid_scores_list, axis=0)

                if len(scores_concatenated) != raw_length:
                    x_old = np.linspace(0, 1, len(scores_concatenated))
                    x_new = np.linspace(0, 1, raw_length)

                    f = scipy.interpolate.interp1d(
                        x_old,
                        scores_concatenated,
                        axis=0,
                        kind='linear',
                    )
                    scores_final = f(x_new)
                else:
                    scores_final = scores_concatenated

                probs_frame = np.repeat(scores_final, 16, axis=0)
                binary_score = gaussian_filter1d(probs_frame, sigma=20)

                pred_scores_binary.append(binary_score)

                gt_offset += raw_length * 16
                continue

            probs_map = logits2.softmax(dim=-1).cpu().numpy()

            valid_probs_list = [
                probs_map[j, :lengths_list[j], :]
                for j in range(len(lengths_list))
            ]
            probs_concatenated = np.concatenate(valid_probs_list, axis=0)

            if len(probs_concatenated) != raw_length:
                x_old = np.linspace(0, 1, len(probs_concatenated))
                x_new = np.linspace(0, 1, raw_length)

                f = scipy.interpolate.interp1d(
                    x_old,
                    probs_concatenated,
                    axis=0,
                    kind='linear',
                )
                probs_final = f(x_new)
            else:
                probs_final = probs_concatenated

            probs_smooth_3 = np.zeros_like(probs_final)
            probs_smooth_5 = np.zeros_like(probs_final)
            probs_smooth_9 = np.zeros_like(probs_final)

            for c in range(probs_final.shape[1]):
                probs_smooth_3[:, c] = gaussian_filter1d(probs_final[:, c], sigma=3.0)
                probs_smooth_5[:, c] = gaussian_filter1d(probs_final[:, c], sigma=5.0)
                probs_smooth_9[:, c] = gaussian_filter1d(probs_final[:, c], sigma=9.0)

            probs_ensemble = (probs_smooth_3 + probs_smooth_5 + probs_smooth_9) / 3.0

            all_class_probs.append(probs_ensemble)

            probs_frame = np.repeat(probs_final, 16, axis=0)
            binary_score = gaussian_filter1d(1.0 - probs_frame[:, 0], sigma=20)

            pred_scores_binary.append(binary_score)

            gt_offset += raw_length * 16

    pred_all = np.concatenate(pred_scores_binary)

    min_l = min(len(gt), len(pred_all))
    gt_align = gt[:min_l]
    pred_align = pred_all[:min_l]

    auc = roc_auc_score(gt_align, pred_align)
    ap_bin = average_precision_score(gt_align, pred_align)

    print(f"\nFinal Result -> AUC: {auc:.6f} | AP: {ap_bin:.6f}")

    map_avg = 0.0

    if gtsegments is not None and len(all_class_probs) > 0:
        try:
            m_results = SOTAMAPCalculator().run(
                all_class_probs,
                gtsegments,
                gtlabels,
            )

            print("")
            print("-" * 65)
            print(f"{'mAP@IoU(%)':^65}")
            print("-" * 65)
            print(f"{'0.1':^10}{'0.2':^10}{'0.3':^10}{'0.4':^10}{'0.5':^10}{'AVG':^10}")

            v1 = m_results.get('@0.1', 0.0) * 100
            v2 = m_results.get('@0.2', 0.0) * 100
            v3 = m_results.get('@0.3', 0.0) * 100
            v4 = m_results.get('@0.4', 0.0) * 100
            v5 = m_results.get('@0.5', 0.0) * 100

            avg = np.mean([v1, v2, v3, v4, v5])

            print(f"{v1:^10.2f}{v2:^10.2f}{v3:^10.2f}{v4:^10.2f}{v5:^10.2f}{avg:^10.2f}")
            print("-" * 65)
            print("")

            map_avg = avg

        except Exception as e:
            print(f"[Error] mAP calculation failed: {e}")

    return auc, ap_bin, map_avg


if __name__ == '__main__':
    args = xd_option.parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    label_map = xd_option.xd_label_map
    prompt_text = get_prompt_text(label_map)

    test_dataset = XDDataset(
        args.visual_length,
        args.test_list,
        True,
        label_map,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    gt = np.load(args.gt_path)

    try:
        gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
        gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    except Exception:
        gtsegments, gtlabels = None, None

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
    model = model.to(device)

    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        load_info = model.load_state_dict(
            {k.replace('module.', ''): v for k, v in state_dict.items()},
            strict=False,
        )

        print("Missing keys:")
        for k in load_info.missing_keys:
            print("  ", k)

        print("Unexpected keys:")
        for k in load_info.unexpected_keys:
            print("  ", k)
        print("✅ Model Loaded!")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    test(
        model,
        test_loader,
        args.visual_length,
        prompt_text,
        gt,
        gtsegments,
        gtlabels,
        device,
        branch_mode=args.branch_mode,
    )
