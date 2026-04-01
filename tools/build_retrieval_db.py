#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch

try:
    import clip
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openai-clip. Install with "
        "`pip install git+https://github.com/openai/CLIP.git`."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CLIP-based retrieval database for ReMoDiffuse."
    )
    parser.add_argument(
        "--data-root",
        default="data/datasets/human_ml3d",
        help="Dataset root containing motion_data/, texts/, split/.",
    )
    parser.add_argument(
        "--split-file",
        default="split/train.txt",
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
        "--max-seq-len",
        type=int,
        default=196,
        help="Max sequence length to pad/crop motions.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP encoding.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for CLIP encoding.",
    )
    parser.add_argument(
        "--output",
        default="data/database/t2m_text_train.npz",
        help="Output npz path.",
    )
    parser.add_argument(
        "--mean-path",
        default=None,
        help="Mean.npy path for normalization. Defaults to data-root/mean_std/Mean.npy",
    )
    parser.add_argument(
        "--std-path",
        default=None,
        help="Std.npy path for normalization. Defaults to data-root/mean_std/Std.npy",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    return parser.parse_args()


def select_caption(lines):
    # Prefer the full-sequence caption (f_tag=0, to_tag=0) if present.
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
    # Fallback to the first caption.
    if lines:
        return lines[0].split("#")[0].strip()
    return ""


def pad_or_crop_motion(motion, max_len):
    length = motion.shape[0]
    if length >= max_len:
        return motion[:max_len], max_len
    pad_len = max_len - length
    pad = np.zeros((pad_len, motion.shape[1]), dtype=np.float32)
    return np.concatenate([motion, pad], axis=0), length


def main():
    args = parse_args()
    data_root = args.data_root
    split_path = os.path.join(data_root, args.split_file)
    motion_root = os.path.join(data_root, args.motion_dir)
    text_root = os.path.join(data_root, args.text_dir)
    mean_path = args.mean_path or os.path.join(data_root, "mean_std", "Mean.npy")
    std_path = args.std_path or os.path.join(data_root, "mean_std", "Std.npy")

    if not os.path.exists(split_path):
        raise SystemExit(f"Split file not found: {split_path}")
    if not os.path.isdir(motion_root):
        raise SystemExit(f"Motion dir not found: {motion_root}")
    if not os.path.isdir(text_root):
        raise SystemExit(f"Text dir not found: {text_root}")
    if not os.path.isfile(mean_path):
        raise SystemExit(f"Mean file not found: {mean_path}")
    if not os.path.isfile(std_path):
        raise SystemExit(f"Std file not found: {std_path}")

    with open(split_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]
    if not ids:
        raise SystemExit("Split file is empty.")

    # Load one sample to get feature dimension.
    sample_motion = np.load(os.path.join(motion_root, ids[0] + ".npy"))
    if sample_motion.ndim != 2:
        raise SystemExit("Motion file should be (T, D).")
    feat_dim = sample_motion.shape[1]
    mean = np.load(mean_path).astype(np.float32).reshape(1, -1)
    std = np.load(std_path).astype(np.float32).reshape(1, -1)
    if mean.shape[1] != feat_dim or std.shape[1] != feat_dim:
        raise SystemExit(
            f"Mean/Std dim mismatch: mean {mean.shape[1]}, std {std.shape[1]}, motion {feat_dim}"
        )

    num_samples = len(ids)
    motions = np.zeros((num_samples, args.max_seq_len, feat_dim), dtype=np.float32)
    m_lengths = np.zeros((num_samples,), dtype=np.int32)
    captions = []

    print(f"Loading motions and captions: {num_samples} samples")
    for i, name in enumerate(ids):
        motion_path = os.path.join(motion_root, name + ".npy")
        text_path = os.path.join(text_root, name + ".txt")
        if not os.path.isfile(motion_path):
            raise SystemExit(f"Missing motion file: {motion_path}")
        if not os.path.isfile(text_path):
            raise SystemExit(f"Missing text file: {text_path}")

        motion = np.load(motion_path).astype(np.float32)
        if motion.shape[1] != feat_dim:
            raise SystemExit(
                f"Feature dim mismatch at {name}: {motion.shape[1]} vs {feat_dim}"
            )
        motion = (motion - mean) / (std + 1e-9)
        motion, length = pad_or_crop_motion(motion, args.max_seq_len)
        motions[i] = motion
        m_lengths[i] = length

        with open(text_path, "r") as f:
            lines = f.readlines()
        caption = select_caption(lines)
        captions.append(caption)

        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{num_samples}")

    device = torch.device(args.device)
    print(f"Loading CLIP model: {args.clip_model} on {device}")
    model, _ = clip.load(args.clip_model, device=device, jit=False)
    model.eval()

    text_dim = model.token_embedding.weight.shape[1]
    n_ctx = model.positional_embedding.shape[0]
    text_features = np.zeros((num_samples, text_dim), dtype=np.float32)
    clip_seq_features = np.zeros((num_samples, n_ctx, text_dim), dtype=np.float32)

    print("Encoding captions with CLIP")
    with torch.no_grad():
        for start in range(0, num_samples, args.batch_size):
            end = min(start + args.batch_size, num_samples)
            batch_caps = captions[start:end]
            tokens = clip.tokenize(batch_caps, truncate=True).to(device)

            # Global text features
            feat = model.encode_text(tokens).float().cpu().numpy()
            text_features[start:end] = feat

            # Token-level features (77 x 512)
            x = model.token_embedding(tokens).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = model.ln_final(x).type(model.dtype)
            x = x.permute(1, 0, 2).float().cpu().numpy()
            clip_seq_features[start:end] = x

            if end % 1000 == 0 or end == num_samples:
                print(f"  encoded {end}/{num_samples}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving retrieval database to: {args.output}")
    np.savez_compressed(
        args.output,
        motions=motions,
        m_lengths=m_lengths,
        captions=np.array(captions, dtype=str),
        text_features=text_features,
        clip_seq_features=clip_seq_features,
    )
    print("Done.")


if __name__ == "__main__":
    main()
