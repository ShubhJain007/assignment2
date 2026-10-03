import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = F.binary_cross_entropy_with_logits(voxel_src, voxel_tgt.float())
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	dists_src = knn_points(point_cloud_src, point_cloud_tgt, K=1).dists[..., 0]
	dists_tgt = knn_points(point_cloud_tgt, point_cloud_src, K=1).dists[..., 0]
	loss_chamfer = dists_src.mean(dim=1) + dists_tgt.mean(dim=1)
	return loss_chamfer.mean()

def smoothness_loss(mesh_src):
	verts = mesh_src.verts_packed()
	edges = mesh_src.edges_packed()
	V = verts.shape[0]
	v0, v1 = edges[:, 0], edges[:, 1]

	neighbor_sum = torch.zeros_like(verts)
	neighbor_sum.index_add_(0, v0, verts[v1])
	neighbor_sum.index_add_(0, v1, verts[v0])

	degree = torch.zeros(V, device=verts.device, dtype=verts.dtype)
	degree.index_add_(0, v0, torch.ones_like(v0, dtype=verts.dtype))
	degree.index_add_(0, v1, torch.ones_like(v1, dtype=verts.dtype))

	laplacian = neighbor_sum / degree.clamp(min=1).unsqueeze(1) - verts
	loss_laplacian = laplacian.norm(dim=1).mean()
	return loss_laplacian
