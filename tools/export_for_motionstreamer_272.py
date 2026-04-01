#!/usr/bin/env python3
import argparse
import json
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mogen.models import build_architecture


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ReMoDiffuse predictions for MotionStreamer 272 evaluator."
    )
    parser.add_argument("config", help="Config file path.")
    parser.add_argument("checkpoint", help="Checkpoint path.")
    parser.add_argument(
        "--data-root",
        default="data/datasets/human_ml3d",
        help="Dataset root containing motion_data/, texts/, split/, mean_std/.",
    )
    parser.add_argument(
        "--split-file",
        default="split/test.txt",
        help="Split file relative to data-root.",
    )
    parser.add_argument(
        "--motion-dir",
        default="motion_data",
        help="Motion dir relative to data-root.",
    )
    parser.add_argument(
        "--text-dir",
        default="texts",
        help="Text dir relative to data-root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for exported predictions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override max sequence length (defaults to config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def select_caption(lines):
    for line in lines:
        parts = line.strip().split("#")
        if len(parts) >= 4:
            try:
                f_tag = float(parts[2])
                to_tag = float(parts[3])
                if np.isnan(f_tag) or np.isnan(to_tag):
                    continue
                if f_tag == 0.0 and to_tag == 0.0:
                    return parts[0].strip()
            except ValueError:
                continue
    if lines:
        return lines[0].split("#")[0].strip()
    return ""


def build_batches(ids, captions, lengths, max_seq_len, input_feats, batch_size, device):
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch_ids = ids[start:end]
        batch_caps = captions[start:end]
        batch_lengths = lengths[start:end]

        motion = torch.zeros(
            (len(batch_ids), max_seq_len, input_feats),
            dtype=torch.float32,
            device=device,
        )
        motion_mask = torch.zeros(
            (len(batch_ids), max_seq_len),
            dtype=torch.float32,
            device=device,
        )
        for i, length in enumerate(batch_lengths):
            motion_mask[i, :length] = 1.0

        motion_length = torch.tensor(batch_lengths, dtype=torch.long, device=device)
        motion_metas = [{"text": cap} for cap in batch_caps]
        yield batch_ids, motion, motion_mask, motion_length, motion_metas


def main():
    args = parse_args()
    mmcv.check_file_exist(args.config)
    mmcv.check_file_exist(args.checkpoint)

    data_root = args.data_root
    split_path = os.path.join(data_root, args.split_file)
    motion_root = os.path.join(data_root, args.motion_dir)
    text_root = os.path.join(data_root, args.text_dir)
    mean_path = os.path.join(data_root, "mean_std", "Mean.npy")
    std_path = os.path.join(data_root, "mean_std", "Std.npy")

    for path in [split_path, motion_root, text_root, mean_path, std_path]:
        if not os.path.exists(path):
            raise SystemExit(f"Missing required path: {path}")

    cfg = mmcv.Config.fromfile(args.config)
    max_seq_len = args.max_seq_len or cfg.model["model"]["max_seq_len"]
    input_feats = cfg.model["model"]["input_feats"]

    mean = np.load(mean_path).astype(np.float32).reshape(1, 1, -1)
    std = np.load(std_path).astype(np.float32).reshape(1, 1, -1)
    if mean.shape[-1] != input_feats or std.shape[-1] != input_feats:
        raise SystemExit(
            f"Mean/Std dim mismatch: mean {mean.shape[-1]}, std {std.shape[-1]}, input_feats {input_feats}"
        )

    with open(split_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise SystemExit("Split file is empty.")

    captions = []
    lengths = []
    for name in ids:
        motion_path = os.path.join(motion_root, name + ".npy")
        text_path = os.path.join(text_root, name + ".txt")
        if not os.path.isfile(motion_path):
            raise SystemExit(f"Missing motion file: {motion_path}")
        if not os.path.isfile(text_path):
            raise SystemExit(f"Missing text file: {text_path}")

        motion = np.load(motion_path)
        length = int(min(len(motion), max_seq_len))
        lengths.append(length)

        with open(text_path, "r") as f:
            lines = f.readlines()
        captions.append(select_caption(lines))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    if args.device == "cpu":
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])

    out_dir = args.output_dir
    pred_dir = os.path.join(out_dir, "pred_motion")
    os.makedirs(pred_dir, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for batch in build_batches(
            ids, captions, lengths, max_seq_len, input_feats, args.batch_size, device
        ):
            batch_ids, motion, motion_mask, motion_length, motion_metas = batch
            input_dict = {
                "motion": motion,
                "motion_mask": motion_mask,
                "motion_length": motion_length,
                "motion_metas": motion_metas,
                "inference_kwargs": {},
            }
            outputs = model(**input_dict)
            for i, sample in enumerate(outputs):
                pred_motion = sample["pred_motion"].cpu().numpy()
                pred_motion = pred_motion * std + mean
                pred_motion = pred_motion.squeeze(0)
                out_path = os.path.join(pred_dir, f"{batch_ids[i]}.npy")
                np.save(out_path, pred_motion)
                manifest.append(
                    {
                        "id": batch_ids[i],
                        "caption": motion_metas[i]["text"],
                        "length": int(motion_length[i].item()),
                        "path": os.path.relpath(out_path, out_dir),
                    }
                )

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": data_root,
        "split_file": args.split_file,
        "max_seq_len": max_seq_len,
        "input_feats": input_feats,
        "mean_path": mean_path,
        "std_path": std_path,
        "normalized_output": False,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"Exported {len(manifest)} samples to {out_dir}")


if __name__ == "__main__":
    main()
