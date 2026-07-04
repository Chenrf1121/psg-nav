"""
Helpers to visualize top-down maps and frontiers.

Functions:
- plot_frontiers(full_map, frontier_mask=None, boundaries=None, centers=None, save_path=None, show=True, origin='upper')

Inputs:
- full_map: 2D array-like (H x W). Occupancy / map values (float/0-1) or torch tensor.
- frontier_mask: 2D bool/0-1 mask where True indicates frontier pixels (same shape as full_map).
- boundaries: optional list of dicts like {"first_two":[(r,c),(r,c)], "last_two":[(r,c),(r,c)], "contour": np.ndarray(...)}.
- centers: optional (K,2) array-like of representative (row, col) points — one per frontier.
- save_path: optional filepath to save PNG. If None and show=False nothing is saved.
- show: whether to call plt.show().
- origin: 'upper' or 'lower' to control y-axis direction in imshow.

Usage:
- import utils.visualize_frontiers as vf
- vf.plot_frontiers(full_map, frontier_mask, boundaries, centers, save_path='debug.png')
"""
from typing import Optional, List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import skimage.morphology
import copy

def calculate_frontiers(full_map, fbe_free_map):
    """
    Calculate frontiers based on the given full map and free map.

    Args:
        full_map (torch.Tensor): The full map tensor.
        fbe_free_map (torch.Tensor): The free map tensor.

    Returns:
        torch.Tensor: Locations of frontiers.
        int: Number of frontiers.
    """
    fbe_map = torch.zeros_like(full_map[0, 0])
    fbe_map[fbe_free_map[0, 0] > 0] = 1  # first free
    fbe_map[skimage.morphology.binary_dilation(full_map[0, 0].cpu().numpy(), skimage.morphology.disk(4))] = 3  # then dilate obstacle

    fbe_cp = copy.deepcopy(fbe_map)
    fbe_cpp = copy.deepcopy(fbe_map)
    fbe_cp[fbe_cp == 0] = 4  # don't know space is 4
    fbe_cp[fbe_cp < 4] = 0  # free and obstacle
    selem = skimage.morphology.disk(1)
    fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0  # don't know space is 0 dilate unknown space

    diff = fbe_map - fbe_cpp  # intersection between unknown area and free area
    frontier_map = diff == 1
    frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
    num_frontiers = len(torch.where(frontier_map)[0])

    if num_frontiers == 0:
        return None, None,0

    return frontier_map, frontier_locations, num_frontiers

def _to_numpy(arr):
    import torch
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, (list, tuple)):
        return np.array(arr)
    # torch tensor -> cpu numpy
    try:
        if hasattr(arr, "cpu"):
            return arr.cpu().numpy()
    except Exception:
        pass
    return np.array(arr)


def plot_frontiers(
    full_map,
    frontier_mask: Optional[np.ndarray] = None,
    boundaries: Optional[List[Dict]] = None,
    centers: Optional[np.ndarray] = None,
    landmark_nodes: Optional[np.ndarray] = None,
    landmark_edges: Optional[List[Tuple[Tuple[float,float], Tuple[float,float]]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    origin: str = "upper",
    
):
    """
    Show (and optionally save) a figure with:
      - full_map as grayscale,
      - frontier_mask overlay in transparent red,
      - centers as red 'x' markers,
      - first_two endpoints as red squares,
      - last_two endpoints as red triangles.

    Coordinates:
      - Boundaries and centers are assumed to be (row, col) pairs.
      - When plotting scatter, x = col, y = row.
    """
    full_map = _to_numpy(full_map)
    frontier_mask = _to_numpy(frontier_mask)
    centers = _to_numpy(centers)

    H, W = full_map.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    # Render a white-background occupancy-style map similar to visualize()
    # full_map values assumed in [0,1] or boolean where >0.5 is obstacle
    try:
        fmap = full_map.astype(float)
    except Exception:
        fmap = np.array(full_map, dtype=float)
    H, W = fmap.shape
    # base white canvas (RGB, values 0..1)
    canvas = np.ones((H, W, 3), dtype=float)
    # colors (RGB 0..1)
    free_rgb = np.array(colors.to_rgb('#E7E7E7'))
    obs_rgb = np.array(colors.to_rgb('#A2A2A2'))
    # obstacle mask: treat values > 0.5 as obstacle
    obs_mask = fmap > 0.5
    # free mask: where not obstacle
    free_mask = ~obs_mask
    canvas[free_mask] = free_rgb
    canvas[obs_mask] = obs_rgb
    ax.imshow(canvas, origin=origin, interpolation="nearest")
    ax.set_title("full_map with frontiers (white background)")
    ax.set_xlim([-0.5, W - 0.5])
    ax.set_ylim([H - 0.5, -0.5] if origin == "upper" else [-0.5, H - 0.5])
    ax.set_xticks([])
    ax.set_yticks([])

    if frontier_mask is not None:
        # show frontier_mask as red transparent overlay
        # create an RGBA image where red=frontier
        cmap_overlay = np.zeros((H, W, 4), dtype=float)
        mask = frontier_mask.astype(bool)
        cmap_overlay[mask, 0] = 1.0  # red channel
        cmap_overlay[mask, 3] = 0.5  # alpha
        ax.imshow(cmap_overlay, origin=origin, interpolation="nearest")

    # Plot centers (representative points) as red X
    if centers is not None and len(centers) > 0:
        centers = np.asarray(centers)
        rows = centers[:, 0]
        cols = centers[:, 1]
        ax.scatter(cols, rows, c="red", s=30, marker="x", linewidths=1.5, label="center")

    # Plot landmark graph: nodes and edges
    if landmark_edges is not None and len(landmark_edges) > 0:
        # draw edges first (so nodes are on top)
        for e in landmark_edges:
            try:
                p1, p2 = e
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]], c="#FFD700", linewidth=1.0, alpha=0.9)
            except Exception:
                pass

    if landmark_nodes is not None and len(landmark_nodes) > 0:
        ln = np.asarray(landmark_nodes)
        ax.scatter(ln[:, 1], ln[:, 0], c="#1f77b4", s=30, marker="o", edgecolors="k", label="landmark")

    # Plot endpoints from boundaries
    if boundaries is not None:
        for b in boundaries:
            # first_two: list of two (r,c)
            if "first_two" in b and b["first_two"] is not None:
                f2 = np.asarray(b["first_two"])
                ax.scatter(f2[:, 1], f2[:, 0], c="red", s=40, marker="s", edgecolors="k")
            # last_two
            if "last_two" in b and b["last_two"] is not None:
                l2 = np.asarray(b["last_two"])
                ax.scatter(l2[:, 1], l2[:, 0], c="red", s=40, marker="^", edgecolors="k")
            # optional: small line connecting first_two to last_two
            try:
                p1 = b["first_two"][0]
                p2 = b["last_two"][-1]
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]], c="red", linewidth=0.6, alpha=0.6)
            except Exception:
                pass

    # Legend (optional)
    # ax.legend(loc='upper right')

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)