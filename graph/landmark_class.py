import numpy as np
import json
import time
import os
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage
from sklearn.decomposition import PCA

class Landmark:
    def __init__(self, id:int, xy:Tuple[float,float], label:Optional[str]=None,
                 desc:Optional[np.ndarray]=None, confidence:float=1.0, t:Optional[float]=None):
        self.id = id
        self.xy = (float(xy[0]), float(xy[1]))
        self.label = label
        self.desc = None if desc is None else np.asarray(desc, dtype=np.float32)
        self.confidence = float(confidence)
        self.t = time.time() if t is None else t

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.xy[0],
            "y": self.xy[1],
            "label": self.label,
            "confidence": self.confidence,
            "t": self.t,
            "desc_shape": None if self.desc is None else list(self.desc.shape)
        }

class LandmarkMap:
    def __init__(self):
        self.landmarks = {}  # id -> Landmark
        self.next_id = 1

    def add_landmark(self, xy:Tuple[float,float], label:Optional[str]=None,
                     desc:Optional[np.ndarray]=None, confidence:float=1.0, t:Optional[float]=None):
        lid = self.next_id
        lm = Landmark(lid, xy, label, desc, confidence, t)
        self.landmarks[lid] = lm
        self.next_id += 1
        return lm

    def update_landmark(self, lid:int, xy:Optional[Tuple[float,float]]=None,
                        desc:Optional[np.ndarray]=None, confidence_delta:float=0.1, t:Optional[float]=None):
        if lid not in self.landmarks:
            return None
        lm = self.landmarks[lid]
        if xy is not None:
            lm.xy = (float(xy[0]), float(xy[1]))
        if desc is not None:
            lm.desc = np.asarray(desc, dtype=np.float32)
        lm.confidence = min(1.0, lm.confidence + confidence_delta)
        lm.t = time.time() if t is None else t
        return lm

    def find_nearest(self, xy:Tuple[float,float], radius:float):
        out = []
        x,y = xy
        for lm in self.landmarks.values():
            dx = lm.xy[0] - x
            dy = lm.xy[1] - y
            d2 = dx*dx + dy*dy
            if d2 <= radius*radius:
                out.append((lm, np.sqrt(d2)))
        out.sort(key=lambda x: x[1])
        return [l for l,_ in out]

    def save(self, path:str):
        data = {"next_id": self.next_id, "landmarks": {k: v.to_dict() for k,v in self.landmarks.items()}}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path:str):
        with open(path, "r") as f:
            data = json.load(f)
        self.next_id = data.get("next_id", 1)
        self.landmarks = {}
        # descriptors are not saved by default in this simple JSON; you can extend to .npy files if needed
        for k,v in data["landmarks"].items():
            lm = Landmark(int(v["id"]), (v["x"], v["y"]), v.get("label"), None, v.get("confidence"), v.get("t"))
            self.landmarks[int(k)] = lm

# Descriptor utilities
def extract_occupancy_patch(occ_map:np.ndarray, center:Tuple[float,float],
                            half_size:int=8, threshold:float=0.5, blur_sigma:float=0.8) -> np.ndarray:
    """
    occ_map: 2D numpy float array; occupancy probability or binary (0..1) with shape (H,W)
    center: (x, y) in pixel coordinates (col, row) same as your pos usage
    half_size: half patch size in pixels -> resulting patch is (2*half_size+1)^2
    returns flattened float32 descriptor (smoothed)
    """
    x, y = int(round(center[0])), int(round(center[1]))
    H, W = occ_map.shape
    x0, x1 = x - half_size, x + half_size + 1
    y0, y1 = y - half_size, y + half_size + 1

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if pad_top or pad_left or pad_right or pad_bottom:
        occ_pad = np.pad(occ_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0.5)
        x0 += pad_left; x1 += pad_left
        y0 += pad_top; y1 += pad_top
    else:
        occ_pad = occ_map

    patch = occ_pad[y0:y1, x0:x1].astype(np.float32)
    # If the map stored free/occ as 0/1 convert to float prob
    # Smooth to reduce noise
    if blur_sigma > 0:
        patch = ndimage.gaussian_filter(patch, sigma=blur_sigma)
    # normalize to zero mean, unit variance (helps L2 matching)
    if patch.std() > 1e-6:
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    return patch.flatten().astype(np.float32)

def reduce_descriptor(desc:np.ndarray, method:str='pca', dim:int=64, pca_obj:Optional[PCA]=None):
    """
    reduce descriptor dimensionsto speed up matching. PCA method returns the vector and optionally fitted PCA object.
    """
    if method is None:
        return desc, None
    if method == 'pca':
        if pca_obj is None:
            pca = PCA(n_components=min(dim, desc.shape[0]), svd_solver='auto')
            desc2 = pca.fit_transform(desc.reshape(1,-1))
            return desc2.ravel(), pca
        else:
            desc2 = pca_obj.transform(desc.reshape(1,-1))
            return desc2.ravel(), pca_obj
    elif method == 'downsample':
        # simple reshape downsample (works for square patches)
        L = int(np.sqrt(desc.shape[0]))
        if L*L != desc.shape[0]:
            return desc, None
        patch = desc.reshape(L,L)
        factor = max(1, L // int(np.sqrt(dim)))
        patch_small = patch[::factor, ::factor]
        return patch_small.flatten(), None
    else:
        return desc, None

def match_descriptor(desc:np.ndarray, candidates:List[np.ndarray], method='l2'):
    """
    Return index of best candidate and distance. If candidates empty returns (None, inf)
    """
    if not candidates:
        return None, float('inf')
    arr = np.stack(candidates, axis=0).astype(np.float32)
    q = desc.astype(np.float32)
    if method == 'l2':
        dists = np.linalg.norm(arr - q[None,:], axis=1)
    elif method == 'cos':
        arrn = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        qn = q / (np.linalg.norm(q) + 1e-9)
        dists = 1.0 - (arrn @ qn)
    else:
        dists = np.linalg.norm(arr - q[None,:], axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])