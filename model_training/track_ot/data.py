"""Sequence views of the existing guarded manifest. No new raw data/labels.

Context is a union of already permitted same-partition windows and candidates.
The original anchor rows, labels and evaluation coverage remain unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from railvivid.common import finite, fingerprint, file_hash, require_columns
from railvivid.errors import ArtifactError, DataIntegrityError, LeakageError
from railvivid.splits import assert_integrity


def sequence_index(frame, cfg):
    require_columns(frame, ["window_id", "partition", "run_id", "start_idx", "end_idx", "chainage_m", "label",
                            "eligible_supervised", "candidate_frame_ids", "candidate_offsets_m"], "sequence source")
    assert_integrity(frame, cfg["_base"])
    if frame.empty: raise DataIntegrityError("Empty sequence source")
    source = frame.reset_index(drop=True)
    if source.window_id.duplicated().any(): raise LeakageError("Duplicate anchor IDs")
    contexts = [None] * len(source)
    half = cfg["sequence"]["windows"] // 2
    stride = cfg["_base"]["data"]["stride_samples"]
    for (_, _), group in source.groupby(["partition", "run_id"], sort=False):
        group = group.sort_values("start_idx")
        indices = group.index.to_numpy(); starts = group.start_idx.to_numpy()
        breaks = np.r_[0, np.flatnonzero(np.diff(starts) != stride) + 1, len(group)]
        for first, last in zip(breaks[:-1], breaks[1:]):
            segment = indices[first:last]
            for j, anchor in enumerate(segment):
                ids = segment[max(0, j - half):min(len(segment), j + half + 1)]
                # Canonical spatial ordering handles reverse-direction traversals.
                contexts[anchor] = sorted(ids.tolist(), key=lambda k: (source.iloc[k].chainage_m, source.iloc[k].start_idx))
    records = []
    for i, indices in enumerate(contexts):
        group = source.iloc[indices]
        if group.partition.nunique() != 1 or group.run_id.nunique() != 1: raise LeakageError("Bag crosses source partition/run")
        image_positions = {}
        for row in group.itertuples():
            if len(row.candidate_frame_ids) != len(row.candidate_offsets_m): raise DataIntegrityError("Candidate ID/offset mismatch")
            for fid, offset in zip(row.candidate_frame_ids, row.candidate_offsets_m):
                position = float(row.chainage_m + offset)
                if fid in image_positions and abs(image_positions[fid] - position) > 0.001:
                    raise DataIntegrityError("A candidate frame has inconsistent positions across windows")
                image_positions[fid] = position
        candidates = sorted(image_positions, key=lambda fid: (image_positions[fid], fid))
        total_candidates = len(candidates)
        if total_candidates > cfg["sequence"]["max_frames"]:
            selected = np.linspace(0, total_candidates - 1, cfg["sequence"]["max_frames"], dtype=int)
            candidates = [candidates[j] for j in selected]
        labels = np.where(group.eligible_supervised.to_numpy(bool), group.label.to_numpy(int), -1)
        bag_label = 1 if (labels == 1).any() else (0 if (labels == 0).all() else -1)
        records.append({"anchor_index": i, "window_id": source.iloc[i].window_id, "partition": source.iloc[i].partition,
                        "context_indices": indices, "context_window_ids": group.window_id.tolist(),
                        "anchor_slot": indices.index(i), "bag_label": bag_label,
                        "frame_ids": candidates, "frame_positions": [image_positions[fid] for fid in candidates],
                        "available_frames": total_candidates,
                        "source_support_min_m": float(group.support_min_m.min()),
                        "source_support_max_m": float(group.support_max_m.max())})
    digest = fingerprint({"source": sorted(source.window_id.tolist()), "sequence": cfg["sequence"], "records": records})
    return source, records, digest


class PatchStore:
    """Read existing frozen DINO patch cache; never download/extract implicitly."""
    def __init__(self, store):
        self.path = Path(store.folder) / "patches.npy"
        meta = store.meta
        expected = meta.get("artifacts", {}).get("patches.npy")
        if not expected or file_hash(self.path) != expected: raise ArtifactError("Verified patches.npy is required for patch-pooling mode; extract it on the training device")
        self.array = np.load(self.path, mmap_mode="r")
        if self.array.ndim != 3 or self.array.shape[0] != len(store.index) or self.array.shape[2] != store.features.shape[1]:
            raise DataIntegrityError("Invalid DINO patch-cache shape")

    def __getstate__(self): return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]; self.array = np.load(self.path, mmap_mode="r")


class SequenceDataset(Dataset):
    def __init__(self, root, source, records, normalizer, cfg, vision, augment=False, corruption=None):
        self.root = Path(root); self.source = source; self.cfg = cfg; self.vision = vision
        self.augment = augment; self.epoch = 0; self.corruption = corruption or {}; self._arrays = OrderedDict()
        self.records = records
        self.frame = source.iloc[[r["anchor_index"] for r in records]].reset_index(drop=True).copy()
        if self.frame.partition.nunique() != 1: raise LeakageError("A dataset view must contain exactly one partition")
        self.mean = np.asarray(normalizer["mean"], dtype="float32")[:, None]
        self.std = np.asarray(normalizer["std"], dtype="float32")[:, None]
        if self.mean.shape != (6, 1) or (self.std <= 0).any(): raise DataIntegrityError("Invalid six-channel training normalizer")
        self.patch_store = None
        if cfg["encoder"]["patch_pooling"]:
            if not hasattr(vision, "_track_patch_store"): vision._track_patch_store = PatchStore(vision)
            self.patch_store = vision._track_patch_store
        required = {fid for r in records for fid in r["frame_ids"]}
        if required - set(vision.lookup): raise ArtifactError("Sequence candidates missing from frozen vision cache")
        self.partners = {}
        if augment and cfg["loss"]["recurrence_weight"] + cfg["loss"]["cycle_weight"] > 0:
            if set(self.frame.partition) != {"train"}: raise LeakageError("Cross-run partners may only be constructed from training rows")
            for run, query in self.frame.groupby("run_id"):
                for other_run, candidates in self.frame.groupby("run_id"):
                    if run == other_run: continue
                    candidates = candidates.sort_values("chainage_m")
                    positions = candidates.chainage_m.to_numpy(); rows = candidates.index.to_numpy()
                    for i in query.index:
                        p = self.frame.iloc[i].chainage_m
                        j = int(np.searchsorted(positions, p))
                        choices = [k for k in [j - 1, j] if 0 <= k < len(positions)]
                        k = min(choices, key=lambda k: abs(positions[k] - p))
                        if abs(positions[k] - p) <= cfg["loss"]["partner_tolerance_m"]:
                            self.partners.setdefault(i, []).append(int(rows[k]))

    def __len__(self): return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy(); state["_arrays"] = OrderedDict(); return state

    def array(self, run):
        if run not in self._arrays:
            path = self.root / "runs" / run / "acceleration.npy"
            try: self._arrays[run] = np.load(path, mmap_mode="r")
            except (OSError, ValueError) as e: raise ArtifactError(f"Cannot read waveform cache {path}: {e}") from e
            if len(self._arrays) > 8: self._arrays.popitem(last=False)
        self._arrays.move_to_end(run)
        return self._arrays[run]

    def item(self, index, augment=True):
        cfg = self.cfg; rec = self.records[index]; anchor = self.source.iloc[rec["anchor_index"]]
        n, m = cfg["sequence"]["windows"], cfg["sequence"]["max_frames"]
        t, d = cfg["_base"]["data"]["window_samples"], cfg["_base"]["model"]["vision_dim"]
        wave = np.zeros((n, 6, t), dtype="float32"); mask = np.zeros(n, dtype=bool)
        wp = np.zeros(n, dtype="float32"); labels = np.full(n, -1, dtype="float32")
        raw = self.array(anchor.run_id)
        for slot, k in enumerate(rec["context_indices"]):
            row = self.source.iloc[k]
            if row.start_idx < 0 or row.end_idx > len(raw) or row.end_idx - row.start_idx != t:
                raise DataIntegrityError(f"Out-of-bounds sequence waveform: {row.window_id}")
            x = np.asarray(raw[int(row.start_idx):int(row.end_idx), :6], dtype="float32").T
            finite(x, row.window_id)
            wave[slot] = (x - self.mean) / self.std; mask[slot] = True
            wp[slot] = row.chainage_m - anchor.chainage_m
            labels[slot] = row.label if row.eligible_supervised else -1
        ids = [self.vision.lookup[fid] for fid in rec["frame_ids"]]
        vision = np.zeros((m, d), dtype="float32"); quality = np.zeros((m, 2), dtype="float32")
        patches = np.zeros((m, cfg["encoder"]["patch_tokens"], d), dtype="float32")
        imask = np.zeros(m, dtype=bool); ip = np.zeros(m, dtype="float32")
        if ids:
            values = self.vision.features[ids]
            if self.patch_store:
                pp = np.asarray(self.patch_store.array[ids], dtype="float32")
                chosen = np.linspace(0, pp.shape[1] - 1, cfg["encoder"]["patch_tokens"], dtype=int)
                if len(set(chosen)) != len(chosen): raise DataIntegrityError("patch_tokens exceeds cache's patch count")
                patches[:len(ids)] = pp[:, chosen]
            vision[:len(ids)] = values; quality[:len(ids)] = self.vision.quality[ids]
            imask[:len(ids)] = True; ip[:len(ids)] = np.asarray(rec["frame_positions"]) - anchor.chainage_m
        key = f"{cfg['training']['seed']}:{self.epoch}:{rec['window_id']}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        vibration_present = True
        if self.augment and augment:
            wave[mask] = wave[mask] * rng.uniform(0.95, 1.05) + rng.normal(0, 0.02, wave[mask].shape).astype("float32")
            dropout = cfg["training"]["modality_dropout"]
            draw = rng.random()
            if draw < dropout / 2: imask[:] = False; vision[:] = 0; patches[:] = 0
            elif draw < dropout and imask.any(): vibration_present = False; wave[:] = 0
        noise = self.corruption.get("noise_std_normalized", 0)
        if noise: wave[mask] += rng.normal(0, noise, wave[mask].shape).astype("float32")
        for channel in self.corruption.get("missing_channels", []): wave[:, channel - 1] = 0
        if self.corruption.get("missing_vision"): imask[:] = False; vision[:] = 0; patches[:] = 0
        if self.corruption.get("missing_vibration"): vibration_present = False; wave[:] = 0
        ip[imask] += float(self.corruption.get("alignment_prior_shift_m", 0))
        values = {"waveform": wave, "wave_mask": mask, "wave_positions": wp, "instance_labels": labels,
                  "vision": vision, "vision_patches": patches, "vision_mask": imask, "image_positions": ip, "quality": quality,
                  "vibration_present": vibration_present, "anchor_slot": rec["anchor_slot"], "index": index,
                  "label": float(anchor.label), "bag_label": float(rec["bag_label"]), "anchor_chainage": float(anchor.chainage_m),
                  "bag_support_low": rec["source_support_min_m"], "bag_support_high": rec["source_support_max_m"]}
        for k in ["waveform", "vision", "vision_patches", "quality", "wave_positions", "image_positions"]: finite(values[k], k)
        return {k: torch.as_tensor(v) for k, v in values.items()}

    def __getitem__(self, index):
        item = self.item(index)
        available = index in self.partners
        partner = self.partners[index][self.epoch % len(self.partners[index])] if available else index
        if self.augment:
            other = self.item(partner, augment=False)
            for k, v in other.items(): item["partner_" + k] = v
            item["partner_available"] = torch.tensor(available)
        return item


def split_datasets(root, frame, normalizer, cfg, vision, augment=False, corruption=None):
    source, records, digest = sequence_index(frame, cfg)
    result = {}
    for part in ["train", "validation", "test"]:
        result[part] = SequenceDataset(root, source, [r for r in records if r["partition"] == part], normalizer, cfg, vision,
                                       augment=augment and part == "train", corruption=corruption)
    return result, records, digest
