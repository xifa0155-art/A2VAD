import csv
import re
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data


class UCFDataset(data.Dataset):
    """
    UCF-Crime dataset adapter for the current AVad/AGST codebase.

    Key points:
    1. Fix stale absolute paths in ucf_CLIP_rgb*.csv by extracting the relative
       path after "UCFClipFeatures/" and joining it with visual_root.
    2. UCF has no audio in this setting, so zero audio features are returned to
       keep the current CLIPVAD forward signature unchanged.
    3. Train samples are sampled/padded to visual_length; test samples keep full
       T for exact frame-level AUC alignment.
    4. Supports binary labels and original 14-class UCF labels.
    5. Supports optional center-crop training by keeping only files ending with
       "__5.npy".
    """

    def __init__(
        self,
        visual_length: int,
        list_file: str,
        is_test: bool,
        label_map: dict,
        visual_root: str = "./Dataset/features/UCF_CLIP/UCFClipFeatures",
        audio_root: Optional[str] = None,
        audio_dim: int = 512,
        filter_label: Optional[str] = None,
        label_mode: str = "multiclass",
        train_crop: str = "all",
        feature_norm: str = "none",
        train_sampling: str = "linspace",
        crop_fusion: str = "none",
    ):
        self.visual_length = int(visual_length)
        self.is_test = bool(is_test)
        self.label_map = label_map
        self.visual_root = Path(visual_root)
        self.audio_root = Path(audio_root) if audio_root else None
        self.audio_dim = int(audio_dim)
        self.filter_label = filter_label
        self.label_mode = label_mode
        self.train_crop = train_crop
        self.feature_norm = feature_norm
        self.train_sampling = train_sampling
        self.crop_fusion = crop_fusion

        if self.crop_fusion not in {"none", "mean", "max"}:
            raise ValueError("crop_fusion must be 'none', 'mean', or 'max'.")

        if self.train_sampling not in {"linspace", "jitter"}:
            raise ValueError("train_sampling must be 'linspace' or 'jitter'.")

        if self.label_mode not in {"binary", "multiclass"}:
            raise ValueError("label_mode must be 'binary' or 'multiclass'.")
        if self.train_crop not in {"all", "center", "random"}:
            raise ValueError("train_crop must be 'all', 'center', or 'random'.")

        self.samples: List[Tuple[str, str]] = self._read_csv(list_file)

                                                    
                                                     
        if self.crop_fusion != "none":
            dedup = {}
            for p, y in self.samples:
                key = self._canonical_crop_path(p)
                if key not in dedup:
                    dedup[key] = (key, y)
            self.samples = list(dedup.values())

        elif not self.is_test:
            if self.train_crop == "center":
                self.samples = [
                    (p, y) for p, y in self.samples
                    if Path(self._extract_relative_path(p)).name.endswith("__5.npy")
                ]

            elif self.train_crop == "random":
                dedup = {}
                for p, y in self.samples:
                    key = self._canonical_crop_path(p)
                    if key not in dedup:
                        dedup[key] = (key, y)
                self.samples = list(dedup.values())

        if filter_label is not None:
            mode = filter_label.lower()
            if mode not in {"normal", "anomaly"}:
                raise ValueError("filter_label must be one of None, 'normal', or 'anomaly'.")

            kept = []
            for path, label in self.samples:
                is_normal = self._is_normal_label(label)
                if mode == "normal" and is_normal:
                    kept.append((path, label))
                elif mode == "anomaly" and not is_normal:
                    kept.append((path, label))
            self.samples = kept

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {list_file} with filter_label={filter_label}, "
                f"label_mode={label_mode}, train_crop={train_crop}."
            )

    @staticmethod
    def _read_csv(list_file: str) -> List[Tuple[str, str]]:
        samples = []
        with open(list_file, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) == 0:
            raise RuntimeError(f"Empty list file: {list_file}")

        start = 1 if rows[0] and rows[0][0].strip().lower() in {"path", "file"} else 0

        for row in rows[start:]:
            if not row:
                continue
            path = row[0].strip()
            label = row[1].strip() if len(row) > 1 else ""
            if path:
                samples.append((path, label))

        return samples

    @staticmethod
    def _is_normal_label(label: str) -> bool:
        return "normal" in str(label).strip().lower()






    def _format_label(self, raw_label: str) -> str:
        if self.label_mode == "binary":
            return "Normal" if self._is_normal_label(raw_label) else "Anomaly"
        return raw_label.strip()

    @staticmethod
    def _extract_relative_path(raw_path: str) -> Path:
        p = raw_path.strip().replace("\\", "/")
        marker = "UCFClipFeatures/"
        if marker in p:
            return Path(p.split(marker, 1)[1])

        if not os.path.isabs(p):
            return Path(p)

        return Path(os.path.basename(p))

    def _resolve_visual_path(self, raw_path: str, raw_label: str) -> Path:
        raw = Path(raw_path)

        if raw.exists():
            return raw

        rel = self._extract_relative_path(raw_path)
        basename = rel.name
        label = raw_label.strip()

        candidates = [self.visual_root / rel]
        if label:
            candidates.append(self.visual_root / label / basename)

        candidates.append(self.visual_root / "Training_Normal_Videos_Anomaly" / basename)
        candidates.append(self.visual_root / "Testing_Normal_Videos_Anomaly" / basename)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "[UCFDataset] Visual feature not found.\n"
            f"  Raw path: {raw_path}\n"
            f"  Parsed relative path: {rel}\n"
            f"  Visual root: {self.visual_root}\n"
            f"  Tried examples: {[str(c) for c in candidates[:4]]}"
        )


    def _load_crop_fused_visual(self, raw_path: str, raw_label: str):
        """
        Load __0~__9 crop features of one video and fuse them to [T, D].
        """
        canonical_path = self._canonical_crop_path(raw_path)

        feats = []
        for crop_idx in range(10):
            crop_raw_path = self._replace_crop_index(canonical_path, crop_idx)

            try:
                crop_visual_path = self._resolve_visual_path(crop_raw_path, raw_label)
                if not os.path.exists(crop_visual_path):
                    continue

                feat = np.load(crop_visual_path).astype(np.float32, copy=False)
                if feat.ndim != 2:
                    raise ValueError(
                        f"Expected visual feature shape [T, D], got {feat.shape}: {crop_visual_path}"
                    )

                feats.append(feat)

            except FileNotFoundError:
                continue

        if len(feats) == 0:
            visual_path = self._resolve_visual_path(canonical_path, raw_label)
            visual = np.load(visual_path).astype(np.float32, copy=False)
            if visual.ndim != 2:
                raise ValueError(
                    f"Expected visual feature shape [T, D], got {visual.shape}: {visual_path}"
                )
        else:
            min_len = min(x.shape[0] for x in feats)
            feats = [x[:min_len] for x in feats]
            crop_stack = np.stack(feats, axis=0)             

            if self.crop_fusion == "mean":
                visual = np.mean(crop_stack, axis=0).astype(np.float32, copy=False)
            elif self.crop_fusion == "max":
                visual = np.max(crop_stack, axis=0).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unsupported crop_fusion: {self.crop_fusion}")

        rel_path = self._extract_relative_path(canonical_path)
        return visual, rel_path


    @staticmethod
    def _replace_crop_index(path_str: str, crop_idx: int) -> str:
        return re.sub(r"__\d+\.npy$", f"__{crop_idx}.npy", path_str)


    @staticmethod
    def _canonical_crop_path(path_str: str) -> str:
        """
        Use __5.npy as canonical key so that __0~__9 of the same video are deduplicated.
        """
        return re.sub(r"__\d+\.npy$", "__5.npy", path_str)

    def _resolve_audio_path(self, rel_path: Path) -> Optional[Path]:
        if self.audio_root is None:
            return None
        p = self.audio_root / rel_path
        return p if p.exists() else None

    @staticmethod
    def _fit_audio_length(audio: np.ndarray, target_len: int, audio_dim: int) -> np.ndarray:
        if audio.ndim == 1:
            audio = audio.reshape(-1, audio_dim)

        if audio.shape[0] == target_len:
            return audio

        if audio.shape[0] > target_len:
            idx = np.linspace(0, audio.shape[0] - 1, target_len).astype(int)
            return audio[idx]

        pad_len = target_len - audio.shape[0]
        return np.pad(audio, ((0, pad_len), (0, 0)), mode="constant")

    def __getitem__(self, index: int):
        raw_path, raw_label = self.samples[index]

        if self.crop_fusion != "none":
            visual, rel_path = self._load_crop_fused_visual(raw_path, raw_label)

        else:
            load_path = raw_path
            if (not self.is_test) and self.train_crop == "random":
                crop_idx = np.random.randint(0, 10)
                load_path = self._replace_crop_index(raw_path, crop_idx)

            visual_path = self._resolve_visual_path(load_path, raw_label)
            rel_path = self._extract_relative_path(load_path)

            visual = np.load(visual_path)
            if visual.ndim != 2:
                raise ValueError(
                    f"Expected visual feature shape [T, D], got {visual.shape}: {visual_path}"
                )

            visual = visual.astype(np.float32, copy=False)

        if self.feature_norm == "l2":
            visual = visual / (np.linalg.norm(visual, axis=1, keepdims=True) + 1e-8)
        elif self.feature_norm == "standard":
            visual = (visual - visual.mean(axis=0, keepdims=True)) / (visual.std(axis=0, keepdims=True) + 1e-6)
            
        raw_t = int(visual.shape[0])

        audio_path = self._resolve_audio_path(rel_path)
        if audio_path is not None:
            audio = np.load(audio_path).astype(np.float32, copy=False)
            audio = self._fit_audio_length(audio, raw_t, self.audio_dim)
        else:
            audio = np.zeros((raw_t, self.audio_dim), dtype=np.float32)

        if self.is_test:
            sample_idx = np.arange(raw_t, dtype=np.int64)
            actual_len = raw_t
        else:
            if raw_t > self.visual_length:
                if self.train_sampling == "linspace":
                    sample_idx = np.linspace(0, raw_t - 1, self.visual_length).astype(np.int64)

                elif self.train_sampling == "jitter":
                    edges = np.linspace(0, raw_t, self.visual_length + 1).astype(np.int64)
                    sample_idx_list = []

                    for s, e in zip(edges[:-1], edges[1:]):
                        s = int(s)
                        e = int(e)

                        if e <= s:
                            sample_idx_list.append(min(s, raw_t - 1))
                        else:
                            sample_idx_list.append(np.random.randint(s, e))

                    sample_idx = np.array(sample_idx_list, dtype=np.int64)
                    sample_idx = np.clip(sample_idx, 0, raw_t - 1)

                else:
                    raise ValueError(f"Unknown train_sampling: {self.train_sampling}")

                visual = visual[sample_idx]
                audio = audio[sample_idx]
                actual_len = self.visual_length
            else:
                valid_idx = np.arange(raw_t, dtype=np.int64)
                pad_len = self.visual_length - raw_t

                visual = np.pad(visual, ((0, pad_len), (0, 0)), mode="constant")
                audio = np.pad(audio, ((0, pad_len), (0, 0)), mode="constant")

                pad_idx = np.full(pad_len, -1, dtype=np.int64)
                sample_idx = np.concatenate([valid_idx, pad_idx], axis=0)
                actual_len = raw_t

        label = self._format_label(raw_label)

        return (
            torch.from_numpy(visual).float(),
            torch.from_numpy(audio).float(),
            label,
            actual_len,
            raw_t,
            torch.from_numpy(sample_idx).long(),
        )

    def __len__(self) -> int:
        return len(self.samples)
