"""
RAG (Retrieval-Augmented Generation) Manager for Object Navigation

This module manages a dynamic knowledge base of successfully found objects.
It stores high-confidence crops and captions from successful episodes and uses
them to verify object detections in subsequent episodes.
"""

import os
import json
import pickle
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class RAGDocument:
    """A single document in the RAG knowledge base"""
    def __init__(self, obj_category: str, crop: np.ndarray, caption: str,
                 confidence: float, feature: Optional[np.ndarray] = None,
                 episode_id: Optional[str] = None, timestamp: Optional[str] = None,
                 dataset: Optional[str] = None, scene_id: Optional[str] = None):
        self.obj_category = obj_category  # Goal object category (e.g., "desk")
        self.crop = crop  # Image crop of the object
        self.caption = caption  # Detected caption
        self.confidence = confidence  # Detection confidence
        self.feature = feature  # Optional: pre-computed feature embedding
        self.episode_id = episode_id or "unknown"
        self.timestamp = timestamp or datetime.now().isoformat()
        self.dataset = dataset or "unknown"  # Dataset name (e.g., "mp3d", "hm3d")
        self.scene_id = scene_id or "unknown"  # Scene ID (e.g., "scene_01")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization (excluding image data)"""
        return {
            'obj_category': self.obj_category,
            'caption': self.caption,
            'confidence': float(self.confidence),
            'episode_id': self.episode_id,
            'timestamp': self.timestamp,
            'dataset': self.dataset,
            'scene_id': self.scene_id,
            'crop_shape': self.crop.shape if self.crop is not None else None,
        }


class RAGManager:
    """
    Manages a dynamic knowledge base of successfully found objects.

    Features:
    - Stores high-confidence object crops from successful episodes
    - Retrieves similar crops for verification in future episodes
    - Calculates cosine similarity between current detection and stored examples
    - Adjusts confidence based on caption consistency
    """

    def __init__(self, storage_dir: str = "data/rag_storage",
                 active_dataset: Optional[str] = None,
                 max_docs_per_category: int = 20,
                 similarity_threshold: float = 0.7,
                 caption_penalty: float = 0.3,
                 false_positive_threshold: float = 0.95,
                 max_false_positives_per_category: int = 20,
                 diversity_threshold: float = 0.97):
        """
        Args:
            storage_dir: Directory to store RAG documents
            active_dataset: Only load documents for this dataset when provided
            max_docs_per_category: Maximum number of documents to keep per object category
            similarity_threshold: Minimum cosine similarity to consider a match
            caption_penalty: Penalty factor when captions don't match (0-1)
            false_positive_threshold: Similarity threshold for rejecting false positives
            max_false_positives_per_category: Maximum false positive samples per category
            diversity_threshold: Minimum similarity to consider samples as duplicates (for diversity)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # False positive storage directory
        self.fp_storage_dir = Path(storage_dir).parent / "rag_false_positives"
        self.fp_storage_dir.mkdir(parents=True, exist_ok=True)

        self.active_dataset = active_dataset
        self.max_docs_per_category = max_docs_per_category
        self.similarity_threshold = similarity_threshold
        self.caption_penalty = caption_penalty
        self.false_positive_threshold = false_positive_threshold
        self.max_false_positives_per_category = max_false_positives_per_category
        self.diversity_threshold = diversity_threshold

        # In-memory storage: {obj_category: [RAGDocument, ...]}
        self.documents: Dict[str, List[RAGDocument]] = {}

        # False positive storage: {obj_category: [RAGDocument, ...]}
        self.false_positive_documents: Dict[str, List[RAGDocument]] = {}

        # Feature extractor (optional, for faster similarity computation)
        self.feature_extractor = None

        # Load existing documents
        self.load_documents()
        self.load_false_positives()

        print(f"[RAG] Initialized with storage at {self.storage_dir}")
        print(f"[RAG] Loaded {sum(len(docs) for docs in self.documents.values())} documents")
        print(f"[RAG] Loaded {sum(len(docs) for docs in self.false_positive_documents.values())} false positives")

    def add_successful_detection(self, obj_category: str, crop: np.ndarray,
                                 caption: str, confidence: float,
                                 episode_id: Optional[str] = None,
                                 dataset: Optional[str] = None,
                                 scene_id: Optional[str] = None) -> None:
        """
        Add a successful detection to the knowledge base with diversity checking.

        Args:
            obj_category: The goal object category (e.g., "desk")
            crop: Image crop of the detected object (H, W, 3)
            caption: Detected caption/label
            confidence: Detection confidence score
            episode_id: Episode identifier
            dataset: Dataset name (e.g., "mp3d", "hm3d")
            scene_id: Scene identifier (e.g., "scene_01")
        """
        if obj_category not in self.documents:
            self.documents[obj_category] = []

        # Create document and extract feature
        feature = self._extract_feature(crop) if crop is not None else None
        doc = RAGDocument(
            obj_category=obj_category,
            crop=crop,
            caption=caption,
            confidence=confidence,
            feature=feature,
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            dataset=dataset,
            scene_id=scene_id
        )

        # Check diversity: skip if too similar to existing samples
        if feature is not None and len(self.documents[obj_category]) > 0:
            max_similarity = self._compute_max_similarity(doc, self.documents[obj_category])
            if max_similarity >= self.diversity_threshold:
                print(f"[RAG] Skipping duplicate sample: {obj_category} - {caption} (conf: {confidence:.3f}, "
                      f"max_similarity: {max_similarity:.3f} >= {self.diversity_threshold})")
                return

        # Add to memory
        self.documents[obj_category].append(doc)

        # Maintain diversity: if over limit, remove most similar or lowest confidence
        removed_docs = []
        if len(self.documents[obj_category]) > self.max_docs_per_category:
            # Strategy: Remove the document with highest average similarity to others
            # This keeps the most diverse set
            original_docs = self.documents[obj_category][:]  # Make a copy
            self.documents[obj_category] = self._select_diverse_samples(
                self.documents[obj_category],
                self.max_docs_per_category
            )

            # Track removed docs for disk cleanup
            current_ids = {id(d) for d in self.documents[obj_category]}
            removed_docs = [d for d in original_docs if id(d) not in current_ids]

        # Save new document to disk
        if doc in self.documents[obj_category]:
            self._save_document(doc)
            print(f"[RAG] Added diverse sample: {obj_category} - {caption} (conf: {confidence:.3f})")
            print(f"[RAG] Total documents for '{obj_category}': {len(self.documents[obj_category])}")

            # Delete removed documents from disk
            for removed_doc in removed_docs:
                self._delete_document(removed_doc)

            if len(removed_docs) > 0:
                print(f"[RAG] Removed {len(removed_docs)} less diverse document(s) from disk")

    def add_false_positive(self, obj_category: str, crop: np.ndarray,
                          caption: str, confidence: float,
                          episode_id: Optional[str] = None,
                          dataset: Optional[str] = None,
                          scene_id: Optional[str] = None) -> None:
        """
        Add a false positive detection to the knowledge base with diversity checking.

        This is called when an episode fails, to store detections that led to failure.
        Future detections similar to these will be rejected.

        Args:
            obj_category: The goal object category (e.g., "desk")
            crop: Image crop of the detected object (H, W, 3)
            caption: Detected caption/label
            confidence: Detection confidence score
            episode_id: Episode identifier
            dataset: Dataset name (e.g., "mp3d", "hm3d")
            scene_id: Scene identifier (e.g., "scene_01")
        """
        if obj_category not in self.false_positive_documents:
            self.false_positive_documents[obj_category] = []

        # Create document and extract feature
        feature = self._extract_feature(crop) if crop is not None else None
        doc = RAGDocument(
            obj_category=obj_category,
            crop=crop,
            caption=caption,
            confidence=confidence,
            feature=feature,
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            dataset=dataset,
            scene_id=scene_id
        )

        # Check diversity: skip if too similar to existing false positives
        if feature is not None and len(self.false_positive_documents[obj_category]) > 0:
            max_similarity = self._compute_max_similarity(doc, self.false_positive_documents[obj_category])
            if max_similarity >= self.diversity_threshold:
                print(f"[RAG FP] Skipping duplicate false positive: {obj_category} - {caption} (conf: {confidence:.3f}, "
                      f"max_similarity: {max_similarity:.3f} >= {self.diversity_threshold})")
                return

        # Add to memory
        self.false_positive_documents[obj_category].append(doc)

        # Maintain diversity: if over limit, keep diverse samples
        removed_docs = []
        if len(self.false_positive_documents[obj_category]) > self.max_false_positives_per_category:
            # Select most diverse samples
            original_docs = self.false_positive_documents[obj_category][:]  # Make a copy
            self.false_positive_documents[obj_category] = self._select_diverse_samples(
                self.false_positive_documents[obj_category],
                self.max_false_positives_per_category
            )

            # Track removed docs for disk cleanup
            current_ids = {id(d) for d in self.false_positive_documents[obj_category]}
            removed_docs = [d for d in original_docs if id(d) not in current_ids]

        # Save to disk
        self._save_false_positive(doc)
        print(f"[RAG FP] Added diverse false positive: {obj_category} - {caption} (conf: {confidence:.3f})")
        print(f"[RAG FP] Total false positives for '{obj_category}': {len(self.false_positive_documents[obj_category])}")

        # Delete removed documents from disk
        for removed_doc in removed_docs:
            self._delete_false_positive(removed_doc)

        if len(removed_docs) > 0:
            print(f"[RAG FP] Removed {len(removed_docs)} less diverse false positive(s) from disk")

    def verify_detection(self, obj_category: str, crop: np.ndarray,
                        caption: str, confidence: float) -> Tuple[bool, float, str, Optional['RAGDocument']]:
        """
        Verify a detection against the knowledge base.

        Args:
            obj_category: The goal object category
            crop: Current image crop
            caption: Current detected caption
            confidence: Current detection confidence

        Returns:
            Tuple of (is_verified, adjusted_confidence, explanation, comparison_doc)
            - is_verified: True if detection passes verification
            - adjusted_confidence: Confidence score after RAG adjustment
            - explanation: Human-readable explanation of the decision
            - comparison_doc: The RAGDocument used for comparison (None if no comparison)
        """
        # Extract feature from current crop first (needed for both FP and positive checks)
        current_feature = self._extract_feature(crop)
        if current_feature is None:
            return True, confidence, "Failed to extract features, accepting detection", None

        # PRIORITY 1: Check against false positive database first!
        # If this looks like a known false positive, reject immediately
        if obj_category in self.false_positive_documents and len(self.false_positive_documents[obj_category]) > 0:
            best_fp_similarity = -1.0
            best_fp_doc = None

            for fp_doc in self.false_positive_documents[obj_category]:
                if fp_doc.feature is None:
                    continue
                similarity = self._cosine_similarity(current_feature, fp_doc.feature)
                if similarity > best_fp_similarity:
                    best_fp_similarity = similarity
                    best_fp_doc = fp_doc

            # If very similar to a false positive, REJECT this detection
            if best_fp_similarity >= self.false_positive_threshold:
                explanation = (f"✗ REJECTED: High similarity ({best_fp_similarity:.3f}) to known false positive "
                              f"from episode {best_fp_doc.episode_id}. This is likely a false detection.")
                print(f"[RAG FP] {explanation}")
                # Return is_verified=False with confidence=0 to signal rejection
                return False, 0.0, explanation, best_fp_doc

        # PRIORITY 2: Check against positive examples for verification
        # If no documents for this category, accept as-is
        if obj_category not in self.documents or len(self.documents[obj_category]) == 0:
            return True, confidence, f"No RAG documents for '{obj_category}', accepting detection", None

        # Step 1: Find documents with matching captions
        caption_matched_docs = []
        for doc in self.documents[obj_category]:
            if doc.feature is None:
                continue
            if self._captions_match(caption, doc.caption):
                caption_matched_docs.append(doc)

        # Step 2: If no caption match, accept as new instance
        if len(caption_matched_docs) == 0:
            explanation = (f"No caption match in RAG for '{caption}' "
                          f"(RAG has {len(self.documents[obj_category])} docs with different captions), "
                          f"accepting as new instance")
            return True, confidence, explanation, None

        # Step 3: Find most similar document among caption-matched docs
        best_similarity = -1.0
        best_doc = None

        for doc in caption_matched_docs:
            similarity = self._cosine_similarity(current_feature, doc.feature)
            if similarity > best_similarity:
                best_similarity = similarity
                best_doc = doc

        # Step 4: Use similarity to adjust confidence
        if best_similarity >= self.similarity_threshold:
            # High similarity with caption match - this is very likely correct!
            adjusted_conf = min(1.0, confidence * 1.1)
            explanation = (f"✓ RAG verified: caption match + high similarity={best_similarity:.3f}, "
                          f"boosting confidence")
            return True, adjusted_conf, explanation, best_doc
        else:
            # Low similarity despite caption match - suspicious!
            # This might be a different instance or a false positive
            penalty = self.caption_penalty * (1.0 - best_similarity)  # Lower similarity = higher penalty
            adjusted_conf = max(0.0, confidence * (1.0 - penalty))

            explanation = (f"⚠ Caption match but low similarity ({best_similarity:.3f} < {self.similarity_threshold}), "
                          f"adjusted_conf={adjusted_conf:.3f}")

            # Accept if adjusted confidence is still reasonable
            is_verified = adjusted_conf > 0.5
            return is_verified, adjusted_conf, explanation, best_doc

    def _extract_feature(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract feature embedding from image crop.

        For now, uses simple histogram-based features.
        Can be replaced with CLIP or other deep features.
        """
        if crop is None or crop.size == 0:
            return None

        try:
            # Resize to standard size
            crop_resized = cv2.resize(crop, (224, 224))

            # Option 1: Simple color histogram (fast, no GPU needed)
            feature = self._extract_color_histogram(crop_resized)

            return feature
        except Exception:
            return None

    def _extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Extract color histogram features"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Compute histogram for each channel
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])

        # Concatenate and normalize
        feature = np.concatenate([hist_r, hist_g, hist_b], axis=0).flatten()
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        return feature

    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors"""
        if feat1 is None or feat2 is None:
            return 0.0

        # Ensure same shape
        if feat1.shape != feat2.shape:
            return 0.0

        # Compute cosine similarity
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def _captions_match(self, caption1: str, caption2: str) -> bool:
        """Check if two captions are semantically equivalent"""
        # Simple exact match (case-insensitive)
        c1 = caption1.lower().strip()
        c2 = caption2.lower().strip()

        if c1 == c2:
            return True

        # Check if one is contained in the other
        if c1 in c2 or c2 in c1:
            return True

        # Could add more sophisticated matching here (e.g., word embeddings)
        return False

    def _compute_max_similarity(self, new_doc: RAGDocument, existing_docs: List[RAGDocument]) -> float:
        """
        Compute the maximum similarity between a new document and existing documents.

        Args:
            new_doc: The new document to check
            existing_docs: List of existing documents to compare against

        Returns:
            Maximum cosine similarity value (0-1)
        """
        if new_doc.feature is None:
            return 0.0

        max_sim = 0.0
        for doc in existing_docs:
            if doc.feature is None:
                continue
            sim = self._cosine_similarity(new_doc.feature, doc.feature)
            if sim > max_sim:
                max_sim = sim

        return max_sim

    def _select_diverse_samples(self, documents: List[RAGDocument], max_count: int) -> List[RAGDocument]:
        """
        Select the most diverse subset of documents.

        Uses a greedy algorithm:
        1. Start with the highest confidence document
        2. Iteratively add the document with lowest average similarity to already selected ones
        3. Continue until we have max_count documents

        Args:
            documents: List of documents to select from
            max_count: Maximum number of documents to select

        Returns:
            List of selected diverse documents
        """
        if len(documents) <= max_count:
            return documents

        # Filter documents with valid features
        valid_docs = [doc for doc in documents if doc.feature is not None]
        invalid_docs = [doc for doc in documents if doc.feature is None]

        if len(valid_docs) == 0:
            # If no valid features, fall back to confidence-based selection
            documents_sorted = sorted(documents, key=lambda d: d.confidence, reverse=True)
            return documents_sorted[:max_count]

        # Start with highest confidence document
        valid_docs_sorted = sorted(valid_docs, key=lambda d: d.confidence, reverse=True)
        selected = [valid_docs_sorted[0]]
        remaining = valid_docs_sorted[1:]

        # Greedily select most diverse documents
        while len(selected) < max_count and len(remaining) > 0:
            # For each remaining document, compute average similarity to selected ones
            best_doc = None
            lowest_avg_sim = float('inf')

            for doc in remaining:
                similarities = []
                for selected_doc in selected:
                    sim = self._cosine_similarity(doc.feature, selected_doc.feature)
                    similarities.append(sim)

                avg_sim = np.mean(similarities) if similarities else 0.0

                # Prefer documents with low average similarity (more diverse)
                # Break ties with confidence
                if avg_sim < lowest_avg_sim or (avg_sim == lowest_avg_sim and doc.confidence > best_doc.confidence):
                    lowest_avg_sim = avg_sim
                    best_doc = doc

            if best_doc is not None:
                selected.append(best_doc)
                remaining.remove(best_doc)

        # Add back invalid docs if we still have space
        remaining_slots = max_count - len(selected)
        if remaining_slots > 0 and len(invalid_docs) > 0:
            invalid_sorted = sorted(invalid_docs, key=lambda d: d.confidence, reverse=True)
            selected.extend(invalid_sorted[:remaining_slots])

        return selected

    def _save_document(self, doc: RAGDocument) -> None:
        """Save a document to disk using hierarchical structure: dataset/scene/success/category/"""
        try:
            # Build hierarchical path: dataset/scene/success/category/
            dataset_dir = self.storage_dir / doc.dataset
            scene_dir = dataset_dir / doc.scene_id
            success_dir = scene_dir / "success"
            category_dir = success_dir / doc.obj_category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique hash suffix to prevent filename collisions
            # Use episode_id + timestamp to create a short hash
            import hashlib
            hash_input = f"{doc.episode_id}_{doc.timestamp}".encode('utf-8')
            hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]  # Use first 8 chars

            # Simple filename: {caption}_{hash}
            base_name = f"{doc.caption}_{hash_suffix}"

            # Save crop image
            crop_path = category_dir / f"{base_name}_crop.jpg"
            if doc.crop is not None:
                cv2.imwrite(str(crop_path), doc.crop)

            # Save feature
            feature_path = category_dir / f"{base_name}_feature.npy"
            if doc.feature is not None:
                np.save(str(feature_path), doc.feature)

            # Save metadata
            metadata_path = category_dir / f"{base_name}_meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(doc.to_dict(), f, indent=2)

        except Exception as e:
            print(f"[RAG] Failed to save document: {e}")

    def _delete_document(self, doc: RAGDocument) -> None:
        """Delete a document from disk using hierarchical structure"""
        try:
            # Build hierarchical path
            dataset_dir = self.storage_dir / doc.dataset
            scene_dir = dataset_dir / doc.scene_id
            success_dir = scene_dir / "success"
            category_dir = success_dir / doc.obj_category
            if not category_dir.exists():
                return

            # Reconstruct hash suffix
            import hashlib
            hash_input = f"{doc.episode_id}_{doc.timestamp}".encode('utf-8')
            hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]
            base_name = f"{doc.caption}_{hash_suffix}"

            # Delete crop image
            crop_path = category_dir / f"{base_name}_crop.jpg"
            if crop_path.exists():
                crop_path.unlink()

            # Delete feature
            feature_path = category_dir / f"{base_name}_feature.npy"
            if feature_path.exists():
                feature_path.unlink()

            # Delete metadata
            metadata_path = category_dir / f"{base_name}_meta.json"
            if metadata_path.exists():
                metadata_path.unlink()

        except Exception as e:
            print(f"[RAG] Failed to delete document: {e}")

    def _save_false_positive(self, doc: RAGDocument) -> None:
        """Save a false positive document to disk using hierarchical structure: dataset/scene/fail/category/"""
        try:
            # Build hierarchical path: dataset/scene/fail/category/
            dataset_dir = self.fp_storage_dir / doc.dataset
            scene_dir = dataset_dir / doc.scene_id
            fail_dir = scene_dir / "fail"
            category_dir = fail_dir / doc.obj_category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique hash suffix to prevent filename collisions
            import hashlib
            hash_input = f"{doc.episode_id}_{doc.timestamp}".encode('utf-8')
            hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]

            # Simple filename: {caption}_{hash}
            base_name = f"{doc.caption}_{hash_suffix}"

            # Save crop image
            crop_path = category_dir / f"{base_name}_crop.jpg"
            if doc.crop is not None:
                cv2.imwrite(str(crop_path), doc.crop)

            # Save feature
            feature_path = category_dir / f"{base_name}_feature.npy"
            if doc.feature is not None:
                np.save(str(feature_path), doc.feature)

            # Save metadata
            metadata_path = category_dir / f"{base_name}_meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(doc.to_dict(), f, indent=2)

        except Exception as e:
            print(f"[RAG FP] Failed to save false positive: {e}")

    def _delete_false_positive(self, doc: RAGDocument) -> None:
        """Delete a false positive document from disk using hierarchical structure"""
        try:
            # Build hierarchical path
            dataset_dir = self.fp_storage_dir / doc.dataset
            scene_dir = dataset_dir / doc.scene_id
            fail_dir = scene_dir / "fail"
            category_dir = fail_dir / doc.obj_category
            if not category_dir.exists():
                return

            # Reconstruct hash suffix
            import hashlib
            hash_input = f"{doc.episode_id}_{doc.timestamp}".encode('utf-8')
            hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]
            base_name = f"{doc.caption}_{hash_suffix}"

            # Delete crop image
            crop_path = category_dir / f"{base_name}_crop.jpg"
            if crop_path.exists():
                crop_path.unlink()

            # Delete feature
            feature_path = category_dir / f"{base_name}_feature.npy"
            if feature_path.exists():
                feature_path.unlink()

            # Delete metadata
            metadata_path = category_dir / f"{base_name}_meta.json"
            if metadata_path.exists():
                metadata_path.unlink()

        except Exception as e:
            print(f"[RAG FP] Failed to delete false positive: {e}")

    def load_documents(self) -> None:
        """Load all documents from disk using hierarchical structure: dataset/scene/success/category/"""
        if not self.storage_dir.exists():
            return

        # Traverse: dataset/scene/success/category/
        for dataset_dir in self.storage_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name

            if self.active_dataset is not None and dataset_name != self.active_dataset:
                continue

            for scene_dir in dataset_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                scene_id = scene_dir.name

                success_dir = scene_dir / "success"
                if not success_dir.exists():
                    continue

                for category_dir in success_dir.iterdir():
                    if not category_dir.is_dir():
                        continue
                    obj_category = category_dir.name

                    if obj_category not in self.documents:
                        self.documents[obj_category] = []

                    # Load all metadata files in this category
                    for meta_path in category_dir.glob("*_meta.json"):
                        try:
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)

                            # Load corresponding crop and feature
                            base_name = meta_path.stem.replace('_meta', '')
                            crop_path = category_dir / f"{base_name}_crop.jpg"
                            feature_path = category_dir / f"{base_name}_feature.npy"

                            crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
                            feature = np.load(str(feature_path)) if feature_path.exists() else None

                            doc = RAGDocument(
                                obj_category=meta.get('obj_category', obj_category),
                                crop=crop,
                                caption=meta['caption'],
                                confidence=meta['confidence'],
                                feature=feature,
                                episode_id=meta.get('episode_id'),
                                timestamp=meta.get('timestamp'),
                                dataset=meta.get('dataset', dataset_name),
                                scene_id=meta.get('scene_id', scene_id)
                            )

                            self.documents[obj_category].append(doc)

                        except Exception as e:
                            print(f"[RAG] Failed to load document {meta_path}: {e}")

        # After loading all documents, enforce max_docs limit per category
        for obj_category in self.documents:
            if len(self.documents[obj_category]) > self.max_docs_per_category:
                print(f"[RAG] Category '{obj_category}' has {len(self.documents[obj_category])} documents, "
                      f"trimming to {self.max_docs_per_category}")

                # Sort by confidence and keep only top-k
                self.documents[obj_category].sort(key=lambda d: d.confidence, reverse=True)
                removed_docs = self.documents[obj_category][self.max_docs_per_category:]
                self.documents[obj_category] = self.documents[obj_category][:self.max_docs_per_category]

                # Delete excess documents from disk
                for removed_doc in removed_docs:
                    self._delete_document(removed_doc)

    def load_false_positives(self) -> None:
        """Load all false positive documents from disk using hierarchical structure: dataset/scene/fail/category/"""
        if not self.fp_storage_dir.exists():
            return

        # Traverse: dataset/scene/fail/category/
        for dataset_dir in self.fp_storage_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name

            if self.active_dataset is not None and dataset_name != self.active_dataset:
                continue

            for scene_dir in dataset_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                scene_id = scene_dir.name

                fail_dir = scene_dir / "fail"
                if not fail_dir.exists():
                    continue

                for category_dir in fail_dir.iterdir():
                    if not category_dir.is_dir():
                        continue
                    obj_category = category_dir.name

                    if obj_category not in self.false_positive_documents:
                        self.false_positive_documents[obj_category] = []

                    # Load all metadata files in this category
                    for meta_path in category_dir.glob("*_meta.json"):
                        try:
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)

                            # Load corresponding crop and feature
                            base_name = meta_path.stem.replace('_meta', '')
                            crop_path = category_dir / f"{base_name}_crop.jpg"
                            feature_path = category_dir / f"{base_name}_feature.npy"

                            crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
                            feature = np.load(str(feature_path)) if feature_path.exists() else None

                            doc = RAGDocument(
                                obj_category=meta.get('obj_category', obj_category),
                                crop=crop,
                                caption=meta['caption'],
                                confidence=meta['confidence'],
                                feature=feature,
                                episode_id=meta.get('episode_id'),
                                timestamp=meta.get('timestamp'),
                                dataset=meta.get('dataset', dataset_name),
                                scene_id=meta.get('scene_id', scene_id)
                            )

                            self.false_positive_documents[obj_category].append(doc)

                        except Exception as e:
                            print(f"[RAG FP] Failed to load false positive {meta_path}: {e}")

        # After loading all FPs, enforce max limit per category
        for obj_category in self.false_positive_documents:
            if len(self.false_positive_documents[obj_category]) > self.max_false_positives_per_category:
                print(f"[RAG FP] Category '{obj_category}' has {len(self.false_positive_documents[obj_category])} false positives, "
                      f"trimming to {self.max_false_positives_per_category}")

                # Keep only most recent (FIFO)
                removed_docs = self.false_positive_documents[obj_category][:-(self.max_false_positives_per_category)]
                self.false_positive_documents[obj_category] = self.false_positive_documents[obj_category][-(self.max_false_positives_per_category):]

                # Delete excess documents from disk
                for removed_doc in removed_docs:
                    self._delete_false_positive(removed_doc)

    def get_statistics(self) -> Dict:
        """Get statistics about the RAG knowledge base"""
        stats = {
            'total_documents': sum(len(docs) for docs in self.documents.values()),
            'categories': list(self.documents.keys()),
            'docs_per_category': {cat: len(docs) for cat, docs in self.documents.items()},
            'total_false_positives': sum(len(docs) for docs in self.false_positive_documents.values()),
            'fp_categories': list(self.false_positive_documents.keys()),
            'fps_per_category': {cat: len(docs) for cat, docs in self.false_positive_documents.items()},
        }
        return stats

    def print_statistics(self) -> None:
        """Print RAG statistics"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("📚 RAG Knowledge Base Statistics")
        print("="*60)
        print(f"Total positive documents: {stats['total_documents']}")
        print(f"Categories: {len(stats['categories'])}")
        for cat, count in stats['docs_per_category'].items():
            print(f"  - {cat}: {count} documents")
        print()
        print(f"Total false positives: {stats['total_false_positives']}")
        print(f"FP Categories: {len(stats['fp_categories'])}")
        for cat, count in stats['fps_per_category'].items():
            print(f"  - {cat}: {count} false positives")
        print("="*60 + "\n")
