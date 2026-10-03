import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import random
import os
import numpy as np
import torch.nn.functional as F
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--vox_thresh', default=0.5, type=float, help='occupancy probability used as the marching cubes isovalue')
    parser.add_argument('--seed', default=0, type=int, help='seeds the random view picked for each test model')
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = torch.sigmoid(predictions)
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=getattr(args, 'vox_thresh', 0.5))
        if len(faces_src) == 0:
            zero = torch.zeros(predictions.shape[0])
            return {f"{m}@{t:f}": zero for t in thresholds for m in ["Precision", "Recall", "F1"]}
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics


def voxels_to_mesh(voxels, center, isovalue=0.5):
    H, W, D = voxels.shape[2:]
    verts, faces = mcubes.marching_cubes(voxels.detach().cpu().squeeze().numpy(), isovalue=isovalue)
    if len(faces) == 0:
        return None
    verts = torch.tensor(verts).float().unsqueeze(0)
    faces = torch.tensor(faces.astype(np.int64))
    verts = utils_vox.Mem2Ref(verts, H, W, D)
    Rot = axis_angle_to_matrix(torch.tensor([[0.0, -math.pi, 0.0]]))
    verts = Rotate(Rot).transform_points(verts)
    verts = verts - verts.mean(1, keepdim=True) + center
    return pytorch3d.structures.Meshes([verts[0]], [faces])


def get_renderers(device, image_size=256):
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)),
        shader=HardPhongShader(device=device),
    )
    points_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=PointsRasterizationSettings(image_size=image_size, radius=0.02)),
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    )
    return mesh_renderer, points_renderer


def render_mesh(mesh, renderer, cameras, lights, color=(0.7, 0.7, 1.0)):
    if mesh is None:
        return torch.ones(renderer.rasterizer.raster_settings.image_size, renderer.rasterizer.raster_settings.image_size, 3)
    verts = mesh.verts_packed()
    textures = TexturesVertex(verts_features=(torch.ones_like(verts) * torch.tensor(color, device=verts.device)).unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([verts], [mesh.faces_packed()], textures=textures)
    return renderer(mesh, cameras=cameras, lights=lights)[0, ..., :3].clamp(0, 1).cpu()


def render_points(points, renderer, cameras, color=(0.7, 0.7, 1.0)):
    rgb = torch.ones_like(points) * torch.tensor(color, device=points.device)
    pc = Pointclouds(points=[points], features=[rgb])
    return renderer(pc, cameras=cameras)[0, ..., :3].clamp(0, 1).cpu()


def visualize(feed_dict, predictions, mesh_gt, renderers, args, dist=1.5, elev=20.0, azim=45.0):
    mesh_renderer, points_renderer = renderers
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=args.device)
    lights = PointLights(location=[[0.0, 1.0, -3.0]], device=args.device)
    image_size = mesh_renderer.rasterizer.raster_settings.image_size

    gt = mesh_gt[0].to(args.device)
    gt_render = render_mesh(gt, mesh_renderer, cameras, lights)

    if args.type == "vox":
        center = gt.verts_packed().mean(0).cpu()
        pred_mesh = voxels_to_mesh(torch.sigmoid(predictions[:1]), center, getattr(args, 'vox_thresh', 0.5))
        pred_render = render_mesh(pred_mesh.to(args.device) if pred_mesh is not None else None, mesh_renderer, cameras, lights)
    elif args.type == "point":
        pred_render = render_points(predictions[0].detach(), points_renderer, cameras)
    elif args.type == "mesh":
        pred_mesh = predictions[0]
        pred_mesh = pytorch3d.structures.Meshes([pred_mesh.verts_packed().detach()], [pred_mesh.faces_packed()])
        pred_render = render_mesh(pred_mesh, mesh_renderer, cameras, lights)

    rgb = feed_dict['images'][0].squeeze().float().cpu()
    rgb = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False)
    rgb = rgb[0].permute(1, 2, 0).clamp(0, 1)

    return torch.cat([rgb, gt_render, pred_render], dim=1).numpy()



def evaluate_model(args):
    random.seed(args.seed)
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    renderers = get_renderers(args.device)
    os.makedirs('vis', exist_ok=True)

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        if (step % args.vis_freq) == 0:
            with torch.no_grad():
                rend = visualize(feed_dict, predictions, mesh_gt, renderers, args)
            plt.imsave(f'vis/{step}_{args.type}.png', rend)
      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
