import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes

import dataset_location
import losses
from eval_model import evaluate, get_renderers, preprocess, visualize
from model import SingleViewto3D
from r2n2_custom import R2N2


def get_args_parser():
    parser = argparse.ArgumentParser("Per-class eval", add_help=False)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--type", default="point", choices=["vox", "point", "mesh"], type=str)
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--n_vis", default=5, type=int, help="renders saved per class")
    parser.add_argument("--n_diversity", default=100, type=int, help="samples per class for the diversity metric")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--load_feat", action="store_true")
    return parser


def pairwise_chamfer(clouds):
    n = clouds.shape[0]
    total, count = 0.0, 0
    for i in range(n - 1):
        src = clouds[i : i + 1].expand(n - i - 1, -1, -1)
        tgt = clouds[i + 1 :]
        total += losses.chamfer_loss(src, tgt).item() * (n - i - 1)
        count += n - i - 1
    return total / max(count, 1)


@torch.no_grad()
def main(args):
    args.batch_size = 1
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                   dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
                                         collate_fn=collate_batched_R2N2, shuffle=False)

    model = SingleViewto3D(args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
    model.eval()

    renderers = get_renderers(args.device)
    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    f1 = defaultdict(list)
    pred_clouds = defaultdict(list)
    gt_clouds = defaultdict(list)
    n_vis = defaultdict(int)

    for step, feed_dict in enumerate(loader):
        label = feed_dict["label"][0]
        images, mesh_gt = preprocess(feed_dict, args)
        predictions = model(images, args)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)
        f1[label].append([metrics["F1@%f" % t].item() for t in thresholds])

        if args.type == "point" and len(pred_clouds[label]) < args.n_diversity:
            pred_clouds[label].append(predictions[0].cpu())
            gt_clouds[label].append(sample_points_from_meshes(mesh_gt, args.n_points)[0])

        if n_vis[label] < args.n_vis:
            rend = visualize(feed_dict, predictions, mesh_gt, renderers, args)
            plt.imsave(os.path.join(args.out_dir, "vis", f"{label}_{n_vis[label]}_{args.type}.png"), rend)
            n_vis[label] += 1

        if step % 100 == 0:
            print(f"[{step}/{len(loader)}] {label}: F1@0.05 {f1[label][-1][-1]:.2f}")

    results = {"checkpoint": args.checkpoint, "splits": dataset_location.SPLITS_PATH, "seed": args.seed,
               "thresholds": thresholds, "per_class": {}}
    all_f1 = []
    for label, scores in sorted(f1.items()):
        scores = np.array(scores)
        all_f1.append(scores)
        entry = {"n": len(scores), "F1": scores.mean(0).round(3).tolist(), "F1@0.05": round(float(scores[:, -1].mean()), 3)}
        if args.type == "point":
            entry["diversity_pred"] = round(pairwise_chamfer(torch.stack(pred_clouds[label]).to(args.device)), 5)
            entry["diversity_gt"] = round(pairwise_chamfer(torch.stack(gt_clouds[label]).to(args.device)), 5)
        results["per_class"][label] = entry
    results["overall_F1@0.05"] = round(float(np.concatenate(all_f1)[:, -1].mean()), 3)

    with open(os.path.join(args.out_dir, f"results_{args.type}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Per-class eval", parents=[get_args_parser()])
    main(parser.parse_args())
