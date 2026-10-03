import argparse
import copy
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PointLights,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, Pointclouds

import dataset_location
from eval_model import get_renderers, render_mesh
from model import SingleViewto3D
from r2n2_custom import R2N2


def load_model(kind, checkpoint, device):
    args = argparse.Namespace(arch="resnet18", type=kind, n_points=1000, batch_size=1, device=device, load_feat=False)
    model = SingleViewto3D(args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
    model.eval()
    return model, args


def encode(model, image):
    x = model.normalize(image.unsqueeze(0).permute(0, 3, 1, 2))
    return model.encoder(x).squeeze(-1).squeeze(-1)


def decode(model, args, feat):
    feat_args = copy.copy(args)
    feat_args.load_feat = True
    return model(feat, feat_args)


def cameras_at(azim, device, dist=1.5, elev=20.0):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return FoVPerspectiveCameras(R=R, T=T, device=device)


def render_colored_points(points, colors, cameras, image_size=256, radius=0.02):
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=PointsRasterizationSettings(image_size=image_size, radius=radius)),
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    )
    pc = Pointclouds(points=[points], features=[colors])
    return renderer(pc, cameras=cameras)[0, ..., :3].clamp(0, 1).cpu().numpy()


def image_panel(image):
    return image.cpu().numpy().clip(0, 1)


def error_heatmaps(samples, model, args, out_dir, device, max_err=0.1):
    cmap = matplotlib.colormaps["jet"]
    fig, axes = plt.subplots(len(samples), 5, figsize=(17, 3.4 * len(samples)))
    for row, s in enumerate(samples):
        with torch.no_grad():
            pred = decode(model, args, encode(model, s["image"]))[0]
        gt = sample_points_from_meshes(s["mesh"], 5000)[0]
        d_pred = knn_points(pred[None], gt[None], K=1).dists[0, :, 0].sqrt()
        d_gt = knn_points(gt[None], pred[None], K=1).dists[0, :, 0].sqrt()
        to_rgb = lambda d: torch.tensor(cmap((d / max_err).clamp(0, 1).cpu().numpy())[:, :3], dtype=torch.float32, device=device)
        panels = [image_panel(s["image"])]
        titles = ["input"]
        for azim in (45, 225):
            panels.append(render_colored_points(pred, to_rgb(d_pred), cameras_at(azim, device)))
            titles.append(f"pred -> GT dist (azim {azim})")
        for azim in (45, 225):
            panels.append(render_colored_points(gt, to_rgb(d_gt), cameras_at(azim, device), radius=0.012))
            titles.append(f"GT -> pred dist (azim {azim})")
        for col, (p, t) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(p)
            axes[row, col].set_title(t, fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(s["name"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_err))
    fig.colorbar(sm, ax=axes, fraction=0.015, label="nearest-neighbour distance")
    plt.savefig(os.path.join(out_dir, "error_heatmaps.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def point_slot_colors(samples, model, args, out_dir, device):
    with torch.no_grad():
        preds = [decode(model, args, encode(model, s["image"]))[0] for s in samples]
    mean = torch.stack(preds).mean(0)
    colors = (mean - mean.min(0).values) / (mean.max(0).values - mean.min(0).values)
    fig, axes = plt.subplots(2, len(samples), figsize=(3.4 * len(samples), 7))
    for col, (s, pred) in enumerate(zip(samples, preds)):
        axes[0, col].imshow(image_panel(s["image"]))
        axes[1, col].imshow(render_colored_points(pred, colors, cameras_at(45, device)))
        axes[0, col].set_title(s["name"], fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle("Each output slot of the point decoder has a fixed colour across all inputs")
    plt.savefig(os.path.join(out_dir, "point_slot_colors.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def interpolation(sample_a, sample_b, models, out_dir, device, steps=7):
    alphas = np.linspace(0, 1, steps)
    mesh_renderer, _ = get_renderers(device)
    lights = PointLights(location=[[0.0, 1.0, -3.0]], device=device)
    cams = cameras_at(45, device)
    fig, axes = plt.subplots(len(models), steps + 2, figsize=(2.6 * (steps + 2), 2.8 * len(models)))
    for row, (kind, (model, args)) in enumerate(models.items()):
        with torch.no_grad():
            fa, fb = encode(model, sample_a["image"]), encode(model, sample_b["image"])
            for col, a in enumerate(alphas):
                out = decode(model, args, (1 - a) * fa + a * fb)
                if kind == "point":
                    pts = out[0]
                    panel = render_colored_points(pts, torch.full_like(pts, 0.6).index_fill_(1, torch.tensor([2], device=device), 1.0), cams)
                else:
                    panel = render_mesh(Meshes([out.verts_packed()], [out.faces_packed()]), mesh_renderer, cams, lights).numpy()
                axes[row, col + 1].imshow(panel)
                axes[row, col + 1].set_title(f"{kind}  a={a:.2f}", fontsize=9)
        axes[row, 0].imshow(image_panel(sample_a["image"]))
        axes[row, -1].imshow(image_panel(sample_b["image"]))
        axes[row, 0].set_title("input A", fontsize=9)
        axes[row, -1].set_title("input B", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    plt.savefig(os.path.join(out_dir, "interpolation.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def voxel_confidence(samples, model, args, out_dir, n_slices=5):
    fig, axes = plt.subplots(2 * len(samples), n_slices + 1, figsize=(2.6 * (n_slices + 1), 5.4 * len(samples)))
    for i, s in enumerate(samples):
        with torch.no_grad():
            prob = torch.sigmoid(decode(model, args, encode(model, s["image"])))[0, 0].cpu().numpy()
        gt = s["voxels"][0].cpu().numpy()
        axis = int(np.argmax([np.ptp(np.nonzero(gt)[k]) for k in range(3)]))
        occupied = np.nonzero(gt)[axis]
        idxs = np.linspace(occupied.min() + 2, occupied.max() - 2, n_slices).astype(int)
        r_pred, r_gt = 2 * i, 2 * i + 1
        axes[r_pred, 0].imshow(image_panel(s["image"]))
        axes[r_pred, 0].set_title(s["name"], fontsize=9)
        axes[r_gt, 0].axis("off")
        for col, idx in enumerate(idxs, start=1):
            im = axes[r_pred, col].imshow(np.take(prob, idx, axis=axis), vmin=0, vmax=1, cmap="magma")
            axes[r_pred, col].contour(np.take(prob, idx, axis=axis), levels=[0.5], colors="cyan", linewidths=0.8)
            axes[r_gt, col].imshow(np.take(gt, idx, axis=axis), vmin=0, vmax=1, cmap="gray")
            axes[r_pred, col].set_title(f"pred p(occ), slice {idx}", fontsize=8)
            axes[r_gt, col].set_title(f"GT, slice {idx}", fontsize=8)
        for ax in list(axes[r_pred]) + list(axes[r_gt]):
            ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.015, label="predicted occupancy probability (cyan = 0.5 contour)")
    plt.savefig(os.path.join(out_dir, "voxel_confidence.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Q2.5 interpretation")
    parser.add_argument("--out_dir", default="results/q25", type=str)
    parser.add_argument("--samples", default=[0, 150, 400, 600], nargs="+", type=int, help="test set indices")
    parser.add_argument("--interp", default=[150, 400], nargs=2, type=int, help="test set indices to interpolate between")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    device = args.device

    dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                   dataset_location.SPLITS_PATH, return_voxels=True)

    def get_sample(i):
        d = dataset[i]
        return {"name": f"test #{i}", "image": d["images"].to(device),
                "mesh": Meshes([d["verts"].to(device)], [d["faces"].to(device)]), "voxels": d["voxels"]}

    samples = [get_sample(i) for i in args.samples]
    point_model = load_model("point", "checkpoint_point.pth", device)
    mesh_model = load_model("mesh", "checkpoint_mesh.pth", device)
    vox_model = load_model("vox", "checkpoint_vox.pth", device)

    error_heatmaps(samples, *point_model, args.out_dir, device)
    point_slot_colors(samples, *point_model, args.out_dir, device)
    interpolation(get_sample(args.interp[0]), get_sample(args.interp[1]),
                  {"point": point_model, "mesh": mesh_model}, args.out_dir, device)
    voxel_confidence(samples[:3], *vox_model, args.out_dir)
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
