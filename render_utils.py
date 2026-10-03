import imageio
import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights, look_at_view_transform
from pytorch3d.structures import Meshes

from eval_model import get_renderers, render_mesh, render_points, voxels_to_mesh


def to_renderable(x, kind):
    if kind == "vox":
        return voxels_to_mesh(x.detach().cpu(), center=torch.zeros(3))
    if kind == "mesh":
        return Meshes([x.verts_packed().detach()], [x.faces_packed()])
    return x.detach()


def render_views(items, kind, device, n_views=36, dist=1.5, elev=20.0, image_size=256, colors=None):
    mesh_renderer, points_renderer = get_renderers(device, image_size)
    lights = PointLights(location=[[0.0, 1.0, -3.0]], device=device)
    items = [to_renderable(x, kind) for x in items]
    frames = []
    for azim in np.linspace(0, 360, n_views, endpoint=False):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        row = []
        for i, x in enumerate(items):
            color = colors[i] if colors is not None else (0.7, 0.7, 1.0)
            if kind == "point":
                row.append(render_points(x.to(device), points_renderer, cameras, color=color))
            else:
                row.append(render_mesh(x.to(device) if x is not None else None, mesh_renderer, cameras, lights, color=color))
        frames.append((torch.cat(row, dim=1).numpy() * 255).astype(np.uint8))
    return frames


def save_gif(frames, path, fps=12):
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
