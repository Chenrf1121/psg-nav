"""
Unified Object Detector Interface for PSG-Nav
Supports both GLIP and FastSAM backends
"""

import sys
import torch
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Optional

# Add FastSAM to path
fastsam_path = '/home/YueChang/phd_ws/reshot_ws/FastSAM'
if fastsam_path not in sys.path:
    sys.path.insert(0, fastsam_path)


class ObjectDetector(ABC):
    """Abstract base class for object detectors"""

    @abstractmethod
    def detect(self, image: np.ndarray, categories: List[str]) -> Dict:
        """
        Detect objects in image

        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            categories: List of object categories to detect

        Returns:
            results: Dictionary with keys:
                - 'bboxes': List of [x1, y1, x2, y2] (N, 4)
                - 'labels': List of label strings (N,)
                - 'scores': List of confidence scores (N,)
        """
        pass


class GLIPDetector(ObjectDetector):
    """GLIP-based object detector"""

    def __init__(
        self,
        config_file: str = "GLIP/configs/pretrain/glip_Swin_L.yaml",
        weight_file: str = "GLIP/MODEL/glip_large_model.pth",
        device: str = "cuda",
        min_image_size: int = 800,
        confidence_threshold: float = 0.61
    ):
        """
        Initialize GLIP detector

        Args:
            config_file: Path to GLIP config file
            weight_file: Path to GLIP model weights
            device: Device to run on
            min_image_size: Minimum image size for GLIP
            confidence_threshold: Confidence threshold for detections
        """
        from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
        from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file)
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", device])

        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=min_image_size,
            confidence_threshold=confidence_threshold,
            show_mask_heatmaps=False
        )

        self.device = device
        print(f"GLIP Detector initialized (threshold={confidence_threshold})")

    def detect(self, image: np.ndarray, categories: List[str]) -> Dict:
        """
        Detect objects using GLIP

        Args:
            image: RGB image (H, W, 3)
            categories: List of categories (e.g., ['chair', 'table'])

        Returns:
            results: Dictionary with bboxes, labels, scores
        """
        # Format caption for GLIP (e.g., "chair . table . bed")
        caption = " . ".join(categories)

        # Run GLIP inference
        predictions = self.glip_demo.inference(image, caption)

        # Extract results
        bboxes = predictions.bbox.cpu().numpy()  # (N, 4) in [x1, y1, x2, y2]
        scores = predictions.get_field("scores").cpu().numpy()  # (N,)
        label_indices = predictions.get_field("labels").cpu().numpy()  # (N,) integers

        # Convert integer labels to string labels
        labels = [categories[idx - 1] if idx > 0 and idx <= len(categories) else "unknown"
                  for idx in label_indices]

        return {
            'bboxes': bboxes,
            'labels': labels,
            'scores': scores,
            'predictions': predictions  # Original GLIP predictions object
        }


class FastSAMDetector(ObjectDetector):
    """FastSAM + CLIP based object detector"""

    def __init__(
        self,
        model_path: str = '/data/YueChang/FastSAM/FastSAM-x.pt',
        device: str = 'cuda',
        clip_model_name: str = 'ViT-B/32',
        imgsz: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        retina_masks: bool = True,
        top_k_per_category: int = 5,
        clip_threshold: float = 0.2,
        min_area: int = 100
    ):
        """
        Initialize FastSAM detector

        Args:
            model_path: Path to FastSAM weights
            device: Device to run on
            clip_model_name: CLIP model variant
            imgsz: Input image size for FastSAM
            conf: FastSAM confidence threshold
            iou: IoU threshold for NMS
            retina_masks: Use high-resolution masks
            top_k_per_category: Return top-k detections per category
            clip_threshold: Minimum CLIP similarity score
            min_area: Minimum mask area in pixels
        """
        try:
            from fastsam import FastSAM
            import clip
        except ImportError as e:
            raise ImportError(f"FastSAM or CLIP not available: {e}")

        # Convert 'cuda' to specific device ID for FastSAM compatibility
        # FastSAM/ultralytics requires specific device ID (e.g., '0', '1') not 'cuda'
        if device == 'cuda':
            import torch
            if torch.cuda.is_available():
                fastsam_device = '0'  # FastSAM uses device ID
                torch_device = 'cuda:0'  # PyTorch uses cuda:N format
            else:
                fastsam_device = 'cpu'
                torch_device = 'cpu'
        elif device.isdigit():
            # If already a device ID like '0', '1', etc.
            fastsam_device = device
            torch_device = f'cuda:{device}'
        else:
            # 'cpu' or other formats
            fastsam_device = device
            torch_device = device

        self.fastsam_device = fastsam_device  # For FastSAM model
        self.device = torch_device  # For PyTorch operations (CLIP)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks
        self.top_k = top_k_per_category
        self.clip_threshold = clip_threshold
        self.min_area = min_area

        # Load FastSAM
        print(f"Loading FastSAM model from {model_path}...")
        self.fastsam_model = FastSAM(model_path)

        # Load CLIP (CLIP accepts 'cuda' or 'cuda:0' format)
        print(f"Loading CLIP model {clip_model_name} on device {torch_device}...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=torch_device)

        print(f"FastSAM Detector initialized (top_k={top_k_per_category}, "
              f"clip_threshold={clip_threshold})")

    def _format_results(self, result, min_area: int = 100):
        """Format FastSAM results

        IMPORTANT: FastSAM returns:
        - masks in original image resolution (due to retina_masks=True)
        - bboxes in FastSAM's processing resolution (aspect-ratio preserved from imgsz)

        We need to extract the actual processing resolution from the result.
        """
        annotations = []
        n = len(result.masks.data) if result.masks is not None else 0

        # Get actual FastSAM processing resolution from the result
        # This is NOT always imgsz x imgsz! FastSAM preserves aspect ratio.
        fastsam_h, fastsam_w = None, None
        if n > 0 and hasattr(result, 'orig_shape'):
            # orig_shape contains the FastSAM processing resolution
            fastsam_h, fastsam_w = result.orig_shape

        for i in range(n):
            mask = result.masks.data[i] == 1.0
            if torch.sum(mask) < min_area:
                continue

            annotation = {
                'id': i,
                'segmentation': mask.cpu().numpy(),
                'bbox': result.boxes.data[i].cpu().numpy()[:4],  # xyxy in FastSAM processing resolution
                'score': result.boxes.conf[i].cpu().item(),
                'area': mask.sum().item(),
                'fastsam_shape': (fastsam_h, fastsam_w) if fastsam_h is not None else None
            }
            annotations.append(annotation)

        return annotations

    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[int]:
        """Get bounding box from mask"""
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return [0, 0, 0, 0]

        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h

        for contour in contours[1:]:
            x_t, y_t, w_t, h_t = cv2.boundingRect(contour)
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)

        return [x1, y1, x2, y2]

    def _crop_image(self, image: Image.Image, annotations: List[Dict]) -> Tuple[List[Image.Image], List[int]]:
        """Crop image regions based on annotations"""
        cropped_boxes = []
        valid_indices = []

        for idx, ann in enumerate(annotations):
            if ann['area'] <= self.min_area:
                continue

            bbox = self._get_bbox_from_mask(ann['segmentation'])
            x1, y1, x2, y2 = bbox

            # Create white-background segmented image
            image_array = np.array(image)
            segmented_array = np.ones_like(image_array) * 255
            segmented_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]

            cropped_img = Image.fromarray(segmented_array)
            cropped_boxes.append(cropped_img)
            valid_indices.append(idx)

        return cropped_boxes, valid_indices

    @torch.no_grad()
    def _compute_similarity(self, cropped_boxes: List[Image.Image], text_queries: List[str]) -> np.ndarray:
        """Compute CLIP similarity between images and texts"""
        import clip

        if len(cropped_boxes) == 0 or len(text_queries) == 0:
            return np.zeros((0, len(text_queries)))

        # Encode images (batch)
        preprocessed = [self.clip_preprocess(img).to(self.device) for img in cropped_boxes]
        stacked_images = torch.stack(preprocessed)
        image_features = self.clip_model.encode_image(stacked_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode texts (batch)
        tokenized = clip.tokenize(text_queries).to(self.device)
        text_features = self.clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)

        return similarity.cpu().numpy()

    def detect(self, image: np.ndarray, categories: List[str]) -> Dict:
        """
        Detect objects using FastSAM + CLIP

        Args:
            image: RGB image (H, W, 3)
            categories: List of categories

        Returns:
            results: Dictionary with bboxes, labels, scores
        """
        # Store original image dimensions for coordinate scaling
        orig_h, orig_w = image.shape[:2]

        # Convert to PIL
        image_pil = Image.fromarray(image)

        # Step 1: FastSAM generates all masks
        everything_results = self.fastsam_model(
            image_pil,
            device=self.fastsam_device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )

        if not everything_results or len(everything_results) == 0:
            return {'bboxes': np.array([]).reshape(0, 4), 'labels': [], 'scores': np.array([])}

        # Step 2: Format results
        annotations = self._format_results(everything_results[0], min_area=self.min_area)

        if len(annotations) == 0:
            return {'bboxes': np.array([]).reshape(0, 4), 'labels': [], 'scores': np.array([])}

        # Step 3: Crop image regions
        cropped_boxes, valid_indices = self._crop_image(image_pil, annotations)

        if len(cropped_boxes) == 0:
            return {'bboxes': np.array([]).reshape(0, 4), 'labels': [], 'scores': np.array([])}

        # Step 4: CLIP similarity (batch)
        similarity = self._compute_similarity(cropped_boxes, categories)

        # Step 5: Extract top-k per category
        all_bboxes = []
        all_labels = []
        all_scores = []

        for cat_idx, category in enumerate(categories):
            category_scores = similarity[:, cat_idx]

            # Get top-k
            top_indices = np.argsort(category_scores)[::-1][:self.top_k]

            for idx in top_indices:
                score = category_scores[idx]
                if score < self.clip_threshold:
                    continue

                orig_idx = valid_indices[idx]
                ann = annotations[orig_idx]

                # Use bbox from FastSAM directly (already in xyxy format)
                # This is more accurate than recomputing from mask
                if 'bbox' in ann:
                    bbox = ann['bbox'][:4]  # FastSAM provides xyxy format
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()

                    # CRITICAL: Scale bbox from FastSAM's processing space to original image space
                    # FastSAM preserves aspect ratio, so its processing resolution may not be imgsz x imgsz
                    # Use the actual FastSAM processing shape if available
                    if 'fastsam_shape' in ann and ann['fastsam_shape'] is not None:
                        fastsam_h, fastsam_w = ann['fastsam_shape']
                        scale_x = orig_w / fastsam_w
                        scale_y = orig_h / fastsam_h
                    else:
                        # Fallback: assume square imgsz (may be incorrect)
                        scale_x = orig_w / self.imgsz
                        scale_y = orig_h / self.imgsz

                    bbox = [
                        bbox[0] * scale_x,  # x1
                        bbox[1] * scale_y,  # y1
                        bbox[2] * scale_x,  # x2
                        bbox[3] * scale_y   # y2
                    ]
                else:
                    # Fallback: compute from mask (mask is already in orig_h x orig_w)
                    bbox = self._get_bbox_from_mask(ann['segmentation'])

                all_bboxes.append(bbox)
                all_labels.append(category)
                all_scores.append(float(score))

        return {
            'bboxes': np.array(all_bboxes).reshape(-1, 4) if all_bboxes else np.array([]).reshape(0, 4),
            'labels': all_labels,
            'scores': np.array(all_scores)
        }


class FastSAMTextDetector(ObjectDetector):
    """FastSAM with text_prompt loop (21 iterations)"""

    def __init__(
        self,
        model_path: str = '/data/YueChang/FastSAM/FastSAM-x.pt',
        device: str = 'cuda',
        imgsz: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        retina_masks: bool = True,
        min_area: int = 100,
        text_conf_threshold: float = 0.5
    ):
        """
        Initialize FastSAM text_prompt detector

        Args:
            model_path: Path to FastSAM weights
            device: Device to run on
            imgsz: Input image size
            conf: FastSAM confidence threshold
            iou: IoU threshold for NMS
            retina_masks: Use high-resolution masks
            min_area: Minimum mask area in pixels
            text_conf_threshold: Confidence threshold for text prompt results
        """
        print(f"Loading FastSAM (text_prompt mode) model from {model_path}...")

        # Convert 'cuda' to specific device ID for FastSAM compatibility
        # FastSAM/ultralytics requires specific device ID (e.g., '0', '1') not 'cuda'
        if device == 'cuda':
            import torch
            if torch.cuda.is_available():
                fastsam_device = '0'  # FastSAM uses device ID
                torch_device = 'cuda:0'  # PyTorch uses cuda:N format
            else:
                fastsam_device = 'cpu'
                torch_device = 'cpu'
        elif device.isdigit():
            # If already a device ID like '0', '1', etc.
            fastsam_device = device
            torch_device = f'cuda:{device}'
        else:
            # 'cpu' or other formats
            fastsam_device = device
            torch_device = device

        # Add FastSAM to path
        import sys
        from configs.detector_config import FASTSAM_CODE_PATH
        if FASTSAM_CODE_PATH not in sys.path:
            sys.path.insert(0, FASTSAM_CODE_PATH)

        from fastsam import FastSAM, FastSAMPrompt

        # IMPORTANT: Import CLIP and inject it into fastsam.prompt module
        # This is needed because FastSAM's text_prompt uses 'clip' without importing it
        import clip
        import fastsam.prompt as prompt_module
        prompt_module.clip = clip

        # Load CLIP model once (to avoid reloading in every text_prompt call)
        # CLIP accepts 'cuda' or 'cuda:0' format
        print(f"Pre-loading CLIP model ViT-B/32 on device {torch_device}...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=torch_device)
        print(f"CLIP model loaded!")

        self.fastsam_device = fastsam_device  # For FastSAM model
        self.device = torch_device  # For PyTorch operations
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks
        self.min_area = min_area
        self.text_conf_threshold = text_conf_threshold

        # Load FastSAM model
        self.fastsam_model = FastSAM(model_path)
        self.FastSAMPrompt = FastSAMPrompt

        print(f"FastSAM (text_prompt mode) model loaded successfully!")

    def detect(self, image: np.ndarray, categories: List[str]) -> dict:
        """
        Detect objects using FastSAM with CLIP text matching (one best match per category)

        Args:
            image: RGB image (H, W, 3)
            categories: List of categories

        Returns:
            results: Dictionary with bboxes, labels, scores
        """
        # Store original image dimensions for coordinate scaling
        orig_h, orig_w = image.shape[:2]

        # Convert to PIL
        image_pil = Image.fromarray(image)

        # Step 1: FastSAM generates all masks once
        everything_results = self.fastsam_model(
            image_pil,
            device=self.fastsam_device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )

        if not everything_results or len(everything_results) == 0:
            return {'bboxes': np.array([]).reshape(0, 4), 'labels': [], 'scores': np.array([])}

        # Step 2: Format results to get annotations
        prompt_process = self.FastSAMPrompt(image_pil, everything_results, device=self.device)
        format_results = prompt_process._format_results(everything_results[0], 0)

        if not format_results or len(format_results) == 0:
            return {'bboxes': np.array([]).reshape(0, 4), 'labels': [], 'scores': np.array([])}

        # Step 3: Crop image regions for all annotations
        cropped_boxes, cropped_images, not_crop, filter_id, annotations = prompt_process._crop_image(format_results)

        if len(cropped_boxes) == 0:
            return {'bboxes': np.array([]).reshape(0, 4), 'labels': [], 'scores': np.array([])}

        print(f"[FastSAM Text] Starting detection for {len(categories)} categories with {len(cropped_boxes)} masks...")
        import time
        start_time = time.time()

        all_bboxes = []
        all_labels = []
        all_scores = []

        # Step 4: For each category, find the best matching mask using CLIP
        for idx, category in enumerate(categories):
            cat_start = time.time()

            # Compute CLIP similarity for this category
            text_query = f"a photo of a {category}"
            scores = prompt_process.retrieve(
                self.clip_model,
                self.clip_preprocess,
                cropped_boxes,
                text_query,
                device=self.device
            )

            # Find best match
            if len(scores) == 0:
                continue

            max_idx = scores.argsort()[-1]  # Index of highest score
            best_score = scores[max_idx]

            # Adjust index for filtered annotations
            actual_idx = max_idx + sum(np.array(filter_id) <= int(max_idx))

            if actual_idx >= len(annotations):
                continue

            ann = annotations[actual_idx]
            mask = ann['segmentation']

            # Check mask area
            area = np.sum(mask > 0) if isinstance(mask, np.ndarray) else 0

            if area < self.min_area:
                continue

            # Get bounding box from FastSAM annotation (more accurate than recomputing from mask)
            if 'bbox' in ann:
                bbox = ann['bbox'][:4]  # FastSAM provides xyxy format
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()

                # CRITICAL: Scale bbox from FastSAM's processing space to original image space
                # FastSAM preserves aspect ratio, so its processing resolution may not be imgsz x imgsz
                # Use the actual FastSAM processing shape if available
                if 'fastsam_shape' in ann and ann['fastsam_shape'] is not None:
                    fastsam_h, fastsam_w = ann['fastsam_shape']
                    scale_x = orig_w / fastsam_w
                    scale_y = orig_h / fastsam_h
                else:
                    # Fallback: assume square imgsz (may be incorrect)
                    scale_x = orig_w / self.imgsz
                    scale_y = orig_h / self.imgsz

                bbox = [
                    bbox[0] * scale_x,  # x1
                    bbox[1] * scale_y,  # y1
                    bbox[2] * scale_x,  # x2
                    bbox[3] * scale_y   # y2
                ]
            else:
                # Fallback: compute from mask (mask is already in orig_h x orig_w)
                bbox = self._get_bbox_from_mask(mask)

            # Check if bbox is valid
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                all_bboxes.append(bbox)
                all_labels.append(category)
                all_scores.append(float(best_score))

                cat_time = time.time() - cat_start
                print(f"  [{idx+1}/{len(categories)}] {category}: ✓ score={best_score:.3f}, area={area}, took {cat_time*1000:.0f}ms")

        total_time = time.time() - start_time
        print(f"[FastSAM Text] Detection complete: {len(all_bboxes)} objects found in {total_time:.2f}s")

        return {
            'bboxes': np.array(all_bboxes).reshape(-1, 4) if all_bboxes else np.array([]).reshape(0, 4),
            'labels': all_labels,
            'scores': np.array(all_scores)
        }

    def _get_bbox_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Extract bounding box from binary mask"""
        if isinstance(mask, dict):
            # RLE format
            from pycocotools import mask as mask_util
            mask = mask_util.decode(mask)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return np.array([0, 0, 0, 0])

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([cmin, rmin, cmax, rmax])


def create_detector(detector_type: str = 'glip', **kwargs) -> ObjectDetector:
    """
    Factory function to create object detector

    Args:
        detector_type: 'glip', 'fastsam_clip', or 'fastsam_text'
        **kwargs: Detector-specific arguments

    Returns:
        detector: ObjectDetector instance

    Example:
        # Create GLIP detector
        detector = create_detector('glip', confidence_threshold=0.61)

        # Create FastSAM + CLIP detector
        detector = create_detector('fastsam_clip', top_k_per_category=5, clip_threshold=0.2)

        # Create FastSAM + text_prompt detector
        detector = create_detector('fastsam_text', text_conf_threshold=0.5)
    """
    detector_type_lower = detector_type.lower()

    if detector_type_lower == 'glip':
        return GLIPDetector(**kwargs)
    elif detector_type_lower == 'fastsam_clip':
        return FastSAMDetector(**kwargs)
    elif detector_type_lower == 'fastsam_text':
        return FastSAMTextDetector(**kwargs)
    elif detector_type_lower == 'fastsam':
        # Backward compatibility: 'fastsam' defaults to CLIP mode
        return FastSAMDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}. Choose 'glip', 'fastsam_clip', or 'fastsam_text'")
