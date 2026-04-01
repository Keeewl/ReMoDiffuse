#!/usr/bin/env python3
import argparse
import os

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mogen.models import build_architecture


T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 272 motion from text using ReMoDiffuse and save a GIF."
    )
    parser.add_argument("config", help="Config file path.")
    parser.add_argument("checkpoint", help="Checkpoint path.")
    parser.add_argument("--text", required=True, help="Input text prompt.")
    parser.add_argument(
        "--data-root",
        default="data/datasets/human_ml3d",
        help="Dataset root containing mean_std/Mean.npy and Std.npy.",
    )
    parser.add_argument(
        "--motion-length",
        type=int,
        default=196,
        help="Motion length (frames). Must be <= max_seq_len.",
    )
    parser.add_argument(
        "--out-gif",
        default="outputs/demo_text2motion_272.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--out-npy",
        default="outputs/demo_text2motion_272.npy",
        help="Output 272 motion .npy path (denormalized).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def accumulate_rotations(relative_rotations):
    total = [relative_rotations[0]]
    for r in relative_rotations[1:]:
        total.append(np.matmul(r, total[-1]))
    return np.asarray(total)


def recover_from_local_position(motion, njoint=22):
    nfrm, _ = motion.shape
    positions_no_heading = motion[:, 8:8 + 3 * njoint].reshape(nfrm, -1, 3)
    velocities_root_xy = motion[:, :2]
    global_heading_diff_rot = motion[:, 2:8]

    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))

    positions_with_heading = np.matmul(
        np.repeat(inv_heading_rot[:, None, :, :], njoint, axis=1),
        positions_no_heading[..., None],
    ).squeeze(-1)

    velocities_root_xyz = np.zeros((nfrm, 3))
    velocities_root_xyz[:, 0] = velocities_root_xy[:, 0]
    velocities_root_xyz[:, 2] = velocities_root_xy[:, 1]
    if nfrm > 1:
        velocities_root_xyz[1:, :] = np.matmul(
            inv_heading_rot[:-1], velocities_root_xyz[1:, :, None]
        ).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz, axis=0)
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]
    return positions_with_heading


def render_gif(joints, out_gif, title, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise SystemExit(
            "Missing Pillow writer. Install with: pip install pillow"
        ) from exc
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = joints.copy().reshape(len(joints), -1, 3)
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)

    height_offset = mins[1]
    data[:, :, 1] -= height_offset
    traj = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def plot_xz_plane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        plane = Poly3DCollection([verts])
        plane.set_facecolor((0.5, 0.5, 0.5, 0.4))
        ax.add_collection3d(plane)

    def update(index):
        ax.cla()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 3)
        ax.set_zlim(0, 3)
        ax.set_title(title)
        ax.grid(False)

        plot_xz_plane(
            mins[0] - traj[index, 0],
            maxs[0] - traj[index, 0],
            0,
            mins[2] - traj[index, 1],
            maxs[2] - traj[index, 1],
        )

        if index > 1:
            ax.plot3D(
                traj[:index, 0] - traj[index, 0],
                np.zeros_like(traj[:index, 0]),
                traj[:index, 1] - traj[index, 1],
                linewidth=1.0,
                color="blue",
            )

        colors = ["red", "blue", "black", "darkred", "darkblue"]
        for chain, color in zip(T2M_KINEMATIC_CHAIN, colors):
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=3.0,
                color=color,
            )

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000 / fps, repeat=False)
    out_dir = os.path.dirname(out_gif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    args = parse_args()
    mmcv.check_file_exist(args.config)
    mmcv.check_file_exist(args.checkpoint)

    cfg = mmcv.Config.fromfile(args.config)
    max_seq_len = cfg.model["model"]["max_seq_len"]
    input_feats = cfg.model["model"]["input_feats"]
    if args.motion_length > max_seq_len:
        raise SystemExit(
            f"motion_length {args.motion_length} exceeds max_seq_len {max_seq_len}"
        )

    mean_path = os.path.join(args.data_root, "mean_std", "Mean.npy")
    std_path = os.path.join(args.data_root, "mean_std", "Std.npy")
    for path in [mean_path, std_path]:
        if not os.path.exists(path):
            raise SystemExit(f"Missing required path: {path}")
    mean = np.load(mean_path).astype(np.float32).reshape(1, 1, -1)
    std = np.load(std_path).astype(np.float32).reshape(1, 1, -1)
    if mean.shape[-1] != input_feats or std.shape[-1] != input_feats:
        raise SystemExit(
            f"Mean/Std dim mismatch: mean {mean.shape[-1]}, std {std.shape[-1]}, input_feats {input_feats}"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    if args.device == "cpu":
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])

    motion = torch.zeros((1, max_seq_len, input_feats), dtype=torch.float32, device=device)
    motion_mask = torch.zeros((1, max_seq_len), dtype=torch.float32, device=device)
    motion_mask[:, : args.motion_length] = 1.0
    motion_length = torch.tensor([args.motion_length], dtype=torch.long, device=device)
    motion_metas = [{"text": args.text}]

    with torch.no_grad():
        outputs = model(
            motion=motion,
            motion_mask=motion_mask,
            motion_length=motion_length,
            motion_metas=motion_metas,
            inference_kwargs={},
        )

    pred_motion = outputs[0]["pred_motion"].cpu().numpy()
    pred_motion = pred_motion[: args.motion_length]
    pred_motion = pred_motion * std.squeeze(0) + mean.squeeze(0)

    out_npy_dir = os.path.dirname(args.out_npy)
    if out_npy_dir:
        os.makedirs(out_npy_dir, exist_ok=True)
    np.save(args.out_npy, pred_motion)

    joints = recover_from_local_position(pred_motion, njoint=22)
    render_gif(joints, args.out_gif, args.text, fps=args.fps)

    print(f"Saved motion: {args.out_npy}")
    print(f"Saved GIF: {args.out_gif}")


if __name__ == "__main__":
    main()
