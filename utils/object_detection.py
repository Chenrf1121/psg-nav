"""
Object Detection Module for PSG-Nav Agent
Handles object detection, goal detection, and RAG verification.
"""

import copy
import torch
import numpy as np

from configs.detector_config import DETECTOR_TYPE


class ObjectDetectionManager:
    """
    Manages object detection and goal finding for the PSG-Nav agent.
    This class encapsulates all detection-related logic to improve code readability.
    """

    def __init__(self, agent):
        """
        Initialize the ObjectDetectionManager.

        Args:
            agent: Reference to the main PSG_Nav_Agent instance
        """
        self.agent = agent

    def detect_objects(self, observations):
        """
        Main object detection function that runs the configured detector (GLIP or FastSAM)
        and checks for goal objects.

        Args:
            observations: Current observations dict containing RGB, depth, GPS, etc.
        """
        # Run object detection using configured detector
        if DETECTOR_TYPE == 'glip':
            self._detect_with_glip(observations)
        elif DETECTOR_TYPE in ['fastsam_clip', 'fastsam_text']:
            self._detect_with_fastsam(observations)
        else:
            raise ValueError(f"Unknown detector type: {DETECTOR_TYPE}")

        # Process detections for all object categories
        self._process_all_object_detections(observations)

        # Handle goal detection based on object type
        if self.agent.scenegraph.obj_goal in self.agent.scenegraph.small_objects:
            # Small objects: use GroundingDINO scene graph detection
            self._detect_goal_with_scenegraph(observations)
        else:
            # Large objects: use GLIP/FastSAM navigation detector
            self._detect_goal_with_navigation_detector(observations)

    def _detect_with_glip(self, observations):
        """Run GLIP detector on current observation."""
        from utils.utils_glip import object_captions
        import cv2
        import os

        # Save original image before detection
        image_bgr = observations["rgb"][:,:,[2,1,0]].copy()
        self._save_detection_images(image_bgr, observations, before_detection=True)

        self.agent.current_obj_predictions = self.agent.glip_demo.inference(
            image_bgr,
            object_captions
        )
        new_labels = self._get_glip_real_label(self.agent.current_obj_predictions)
        self.agent.current_obj_predictions.add_field("labels", new_labels)

        # Print detection details
        num_detections = len(self.agent.current_obj_predictions)

        if num_detections > 0:
            scores = self.agent.current_obj_predictions.get_field('scores').cpu().numpy()
            bboxes = self.agent.current_obj_predictions.bbox.cpu().numpy()


            # Save image with bboxes after detection
            self._save_detection_images(image_bgr, observations, before_detection=False,
                                       bboxes=bboxes, labels=new_labels, scores=scores)

    def _detect_with_fastsam(self, observations):
        """Run FastSAM detector on current observation."""
        from configs.detector_config import CATEGORIES_21
        from utils.utils_glip import categories_21_origin

        image_rgb = observations["rgb"][:,:,[2,1,0]]  # Convert BGR to RGB
        results = self.agent.detector.detect(image_rgb, categories_21_origin)

        # Convert FastSAM results to GLIP-like format for compatibility
        class MockPredictions:
            def __init__(self, bboxes, labels, scores):
                self.bbox = torch.tensor(bboxes) if not isinstance(bboxes, torch.Tensor) else bboxes
                # Convert labels to tensor if it's a list (FastSAM returns list of strings)
                if isinstance(labels, list):
                    # Create a wrapper that has .tolist() method
                    class LabelWrapper:
                        def __init__(self, label_list):
                            self.labels = label_list
                        def tolist(self):
                            return self.labels
                        def __len__(self):
                            return len(self.labels)
                        def __getitem__(self, idx):
                            return self.labels[idx]
                    labels = LabelWrapper(labels)
                self._fields = {'labels': labels, 'scores': torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores}

            def get_field(self, name):
                return self._fields.get(name)

            def add_field(self, name, value):
                self._fields[name] = value

        self.agent.current_obj_predictions = MockPredictions(
            results['bboxes'],
            results['labels'],
            results['scores']
        )

        obj_labels = results['labels']
        print(f"[FastSAM Detection] Found {len(obj_labels)} objects")
        print(f"  bbox shape: {self.agent.current_obj_predictions.bbox.shape}")
        print(f"  scores shape: {self.agent.current_obj_predictions.get_field('scores').shape}")
        print(f"  labels: {obj_labels[:5]}...")  # First 5 labels

    def _process_all_object_detections(self, observations):
        """
        Process detections for all object categories and update obj_locations.

        Args:
            observations: Current observations
        """
        from utils.utils_glip import categories_21_origin

        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        category_to_index = {
            category_name: idx for idx, category_name in enumerate(categories_21_origin)
        }

        for j, label in enumerate(obj_labels):
            if label in category_to_index:
                confidence = self.agent.current_obj_predictions.get_field("scores")[j]
                bbox = self.agent.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.agent.depth[center_point[1],center_point[0],0]


                if temp_distance >= self.agent.distance_threshold:
                    continue

                obj_gps = self.agent.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(self.agent.map_size_cm/10 - obj_gps[1]*100/self.agent.resolution)
                y = int(self.agent.map_size_cm/10 + obj_gps[0]*100/self.agent.resolution)
                self.agent.obj_locations[category_to_index[label]].append([confidence, x, y])

    def _detect_goal_with_scenegraph(self, observations):
        """
        Detect goal using GroundingDINO scene graph (for small objects).

        Args:
            observations: Current observations
        """
        # Using GroundingDINO scene graph detection for small objects
        self.agent.active_detection_path = 'scenegraph'

        self.agent.segment_num = len(self.agent.scenegraph.segment2d_results)
        goal_mask = []

        if self.agent.segment_num > self.agent.last_segment_num:
            self.agent.last_segment_num = self.agent.segment_num
            segment2d_result = self.agent.scenegraph.segment2d_results[-1]
            indices = []

            for index, element in enumerate(segment2d_result['caption']):
                if self.agent.obj_goal_sg in element.split(' '):
                    for node in self.agent.scenegraph.nodes:
                        if node.is_goal_node and node.object['image_idx'][-1] == len(self.agent.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                            indices.append(index)

            goal_mask = [segment2d_result['mask'][index] for index in indices]

        if len(goal_mask) > 0:
            self._process_scenegraph_goal_masks(goal_mask, observations, segment2d_result)
        else:
            # No goal detected, reset if previously found
            if self.agent.found_goal:
                self.agent.reset_goal_detection()

    def _process_scenegraph_goal_masks(self, goal_mask, observations, segment2d_result):
        """
        Process goal masks from scene graph detection.

        Args:
            goal_mask: List of goal masks
            observations: Current observations
            segment2d_result: Scene graph segment result
        """
        possible_goal_detected_before = copy.deepcopy(self.agent.found_possible_goal)
        shortest_distance = 120
        shortest_distance_angle = 0

        for mask in goal_mask:
            center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
            center_point = torch.tensor([center_point[1], center_point[0]])
            temp_direction = (center_point[0] - 320) * 79 / 640
            temp_distance = self.agent.depth[center_point[1],center_point[0],0]
            # Find valid depth if current is invalid
            k = 0
            pos_neg = 1
            while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                pos_neg *= -1
                k += 0.5
                temp_distance = max(
                    self.agent.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                    self.agent.depth[center_point[1],center_point[0]+int(pos_neg*k),0]
                )
            if temp_distance >= self.agent.distance_threshold:
                self.agent.found_possible_goal = True
                self.agent.explanation = f"Potential goal detected: {self.agent.obj_goal} at {temp_distance:.2f}m away (beyond {self.agent.distance_threshold}m threshold)"
                # Update angle for GPS calculation even when goal is far
                if temp_distance < shortest_distance:
                    shortest_distance = temp_distance
                    shortest_distance_angle = temp_direction
            else:
                # Goal found within threshold
                self._handle_scenegraph_goal_found(
                    center_point, temp_direction, temp_distance,
                    observations, segment2d_result
                )

                # Update shortest distance for GPS calculation
                if temp_distance < shortest_distance:
                    shortest_distance = temp_distance
                    shortest_distance_angle = temp_direction

        # Update goal GPS if found
        if self.agent.found_goal:
            self.agent.goal_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            self.agent.explanation = f"Goal found: {self.agent.obj_goal} at {shortest_distance:.2f}m away (detection #{self.agent.found_goal_times})"
        elif self.agent.found_possible_goal:
            # Update possible goal GPS whenever we have a valid possible goal
            self.agent.possible_goal_temp_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            print(f"[SceneGraph GPS Update] Updated possible_goal_temp_gps with angle={shortest_distance_angle:.2f}°, distance={shortest_distance:.2f}m -> gps={self.agent.possible_goal_temp_gps}")

    def _handle_scenegraph_goal_found(self, center_point, temp_direction, temp_distance, observations, segment2d_result):
        """
        Handle goal found from scene graph detection (including RAG verification).

        Args:
            center_point: Center point of goal mask
            temp_direction: Direction to goal
            temp_distance: Distance to goal
            observations: Current observations
            segment2d_result: Scene graph segment result
        """
        # Calculate goal GPS and map coordinates for false positive marking
        goal_gps = self.agent.get_goal_gps(observations, temp_direction, temp_distance)
        map_x = int(self.agent.map_size_cm/10 - goal_gps[1]*100/self.agent.resolution)
        map_y = int(self.agent.map_size_cm/10 + goal_gps[0]*100/self.agent.resolution)


        # Check if this position is marked as false positive (value == -1)
        if 0 <= map_y < self.agent.map_size and 0 <= map_x < self.agent.map_size:
            if self.agent.goal_gps_map[map_y, map_x] == -1:
                print(f"[False Positive Skip] SceneGraph position map({map_y},{map_x}) is marked as false positive, skipping detection")
                return  # Skip this detection entirely

        # Extract crop for RAG
        bbox_int = [
            int(center_point[0] - 50), int(center_point[1] - 50),
            int(center_point[0] + 50), int(center_point[1] + 50)
        ]
        bbox_int[0] = max(0, bbox_int[0])
        bbox_int[1] = max(0, bbox_int[1])
        bbox_int[2] = min(observations["rgb"].shape[1], bbox_int[2])
        bbox_int[3] = min(observations["rgb"].shape[0], bbox_int[3])
        crop = observations["rgb"][bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]

        # Get detected caption for debugging
        detected_caption = segment2d_result['caption'][0] if 'caption' in segment2d_result else self.agent.obj_goal_sg
        initial_conf = 0.9

        print(f"[RAG Window] Adding scenegraph detection: detected_caption='{detected_caption}', using caption='{self.agent.obj_goal}' (goal object)")

        # Update sliding window - CRITICAL: use obj_goal as caption
        detection_dict = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detected caption
            'confidence': initial_conf,
            'distance': temp_distance,
            'step': self.agent.total_steps
        }
        self.agent.rag_sliding_window.append(detection_dict)

        # Keep only latest N detections
        if len(self.agent.rag_sliding_window) > self.agent.rag_sliding_window_size:
            self.agent.rag_sliding_window.pop(0)

        # Update found_goal state
        if self.agent.found_goal:
            if temp_distance < self.agent.distance_threshold:
                self.agent.found_goal_times = self.agent.found_goal_times + 1

        self.agent.found_goal = True
        self.agent.found_possible_goal = False

        # Update latest goal detection - also use obj_goal and include map coordinates
        self.agent.rag_latest_goal_detection = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detected caption
            'confidence': initial_conf,
            'distance': temp_distance,
            'verified': True,
            'map_x': map_x,  # Store map coordinates for false positive marking
            'map_y': map_y
        }

        # RAG verification
        if self.agent.rag_check_enabled and self.agent.rag_latest_goal_detection is not None:
            self._verify_detection_with_rag()

    def _detect_goal_with_navigation_detector(self, observations):
        """
        Detect goal using GLIP/FastSAM navigation detector (for large objects).

        Args:
            observations: Current observations
        """
        # Using GLIP/FastSAM navigation detector for large objects
        self.agent.active_detection_path = 'navigation'

        # Find goal bboxes
        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        goal_bbox = []
        goal_bbox_indices = []

        for j, label in enumerate(obj_labels):
            # Exact match or word-level match (avoid substring false positives)
            label_lower = label.lower() if isinstance(label, str) else str(label).lower()
            goal_lower = self.agent.obj_goal.lower()

            # Check if goal matches (exact match or as separate word)
            is_match = (goal_lower == label_lower or
                       goal_lower in label_lower.split() or
                       label_lower in goal_lower.split())

            if is_match:
                print(f"[Goal Match] Matched '{label}' as goal '{self.agent.obj_goal}'")
                goal_bbox.append(self.agent.current_obj_predictions.bbox[j])
                goal_bbox_indices.append(j)
            elif self.agent.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.agent.current_obj_predictions.bbox[j])
                goal_bbox_indices.append(j)

        if DETECTOR_TYPE in ['fastsam_clip', 'fastsam_text']:
            print(f"[Goal Matching] obj_goal='{self.agent.obj_goal}', found {len(goal_bbox)} matches out of {len(obj_labels)} detections")
            if len(goal_bbox) > 0:
                matched_labels = [obj_labels[idx] if hasattr(obj_labels, '__getitem__') else obj_labels.tolist()[idx] for idx in goal_bbox_indices]
                print(f"  Matched labels: {matched_labels}")

        if len(goal_bbox) > 0:
            self._process_navigation_goal_bboxes(goal_bbox, goal_bbox_indices, observations)

    def _process_navigation_goal_bboxes(self, goal_bbox, goal_bbox_indices, observations):
        """
        Process goal bboxes from navigation detector.

        Args:
            goal_bbox: List of goal bboxes
            goal_bbox_indices: Original indices in predictions
            observations: Current observations
        """
        possible_goal_detected_before = copy.deepcopy(self.agent.found_possible_goal)
        goal_prediction = copy.deepcopy(self.agent.current_obj_predictions)
        goal_prediction.bbox = torch.stack(goal_bbox)

        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        shortest_distance = 120
        shortest_distance_angle = 0

        for idx_box, box in enumerate(goal_prediction.bbox):
            box = box.to(torch.int64)
            original_idx = goal_bbox_indices[idx_box] if idx_box < len(goal_bbox_indices) else idx_box

            # Get center point with occlusion handling
            center_point, is_occluded, true_depth = self.agent.get_unoccluded_center_point(box, original_idx, self.agent.depth)

            temp_direction = (center_point[0] - 320) * 79 / 640
            temp_distance = true_depth
            goal_gps = self.agent.get_goal_gps(observations, temp_direction, temp_distance)

            # Fallback depth search if invalid
            k = 0
            pos_neg = 1
            while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                pos_neg *= -1
                k += 0.5
                temp_distance = max(
                    self.agent.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                    self.agent.depth[center_point[1],center_point[0]+int(pos_neg*k),0]
                )
            obj_labels = self.agent.current_obj_predictions.get_field("labels")
            detected_label = obj_labels[original_idx] if original_idx < len(obj_labels) else self.agent.obj_goal

            if temp_distance >= self.agent.distance_threshold:
                self.agent.found_possible_goal = True
                self.agent.explanation = f"Potential goal detected: {self.agent.obj_goal} at {temp_distance:.2f}m away (beyond {self.agent.distance_threshold}m threshold)"
                print(f"[距离检查] 检测到 {detected_label}, 距离={temp_distance:.2f}m >= 阈值{self.agent.distance_threshold}m, "
                      f"❌ 太远，仅设置found_possible_goal，不记录GPS")
            else:
                # Goal found within threshold
                print(f"[距离检查] 检测到 {detected_label}, 距离={temp_distance:.2f}m < 阈值{self.agent.distance_threshold}m, "
                      f"✓ 距离合适，记录GPS位置...")
                self._handle_navigation_goal_found(
                    box, original_idx, goal_gps, temp_distance, observations
                )

            # Update shortest distance
            if temp_distance < shortest_distance:
                shortest_distance = temp_distance
                shortest_distance_angle = temp_direction

        # Update found_goal state based on detection count
        self.agent.found_goal_times = self.agent.goal_gps_map.max()
        detection_threshold = self.agent.get_goal_detection_threshold()

        if len(goal_bbox) > 0:
            print(f"\n[目标计数] goal_gps_map最大计数={int(self.agent.found_goal_times)}/{detection_threshold}, "
                  f"需要>={detection_threshold}次才能确认目标")
            if self.agent.found_goal_times > 0:
                # Show where detections are in the map
                detection_positions = np.where(self.agent.goal_gps_map > 0)
                num_positions = len(detection_positions[0])
                print(f"[目标计数] 检测到{num_positions}个不同位置的目标候选点:")
                for i in range(min(num_positions, 5)):  # Show top 5
                    y, x = detection_positions[0][i], detection_positions[1][i]
                    count = int(self.agent.goal_gps_map[y, x])
                    print(f"  位置{i+1}: map({y},{x}), 计数={count}")
            else:
                print(f"[目标计数] ⚠️ 没有有效的GPS记录（所有检测距离可能都超过阈值）")
        else:
            print(f"[目标计数] 当前帧未检测到目标物体")

        if self.agent.found_goal_times >= detection_threshold:
            self.agent.found_goal = True

        # Update goal GPS if found
        if self.agent.found_goal:
            # Transfer temp detection to latest detection BEFORE RAG verification
            if hasattr(self.agent, '_temp_latest_detection'):
                self.agent.rag_latest_goal_detection = self.agent._temp_latest_detection
                # Don't delete yet - will be cleaned up in _select_most_recent_goal_position

            # RAG verification
            if self.agent.rag_check_enabled and self.agent.rag_latest_goal_detection is not None:
                self._verify_detection_with_rag()

            # Set goal GPS if still found after RAG check
            if self.agent.found_goal:
                self._select_most_recent_goal_position(detection_threshold)
        elif self.agent.found_goal_times > 0:
            self.agent.found_possible_goal = True
            self.agent.possible_goal_temp_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
        elif not possible_goal_detected_before:
            self.agent.possible_goal_temp_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)

    def _handle_navigation_goal_found(self, box, original_idx, goal_gps, temp_distance, observations):
        """
        Handle goal found from navigation detector (including RAG data collection).

        Args:
            box: Goal bbox
            original_idx: Original index in predictions
            goal_gps: Goal GPS coordinates
            temp_distance: Distance to goal
            observations: Current observations
        """
        # Calculate map coordinates (use correct conversion formula)
        map_x = int(self.agent.map_size_cm/10 - goal_gps[1]*100/self.agent.resolution)
        map_y = int(self.agent.map_size_cm/10 + goal_gps[0]*100/self.agent.resolution)

        # Check if this position is marked as false positive (value == -1)
        if 0 <= map_y < self.agent.map_size and 0 <= map_x < self.agent.map_size:
            if self.agent.goal_gps_map[map_y, map_x] == -1:
                print(f"[False Positive Skip] Position map({map_y},{map_x}) is marked as false positive, skipping detection update")
                return  # Skip this detection entirely

        # Update goal_gps_map
        thres = int(self.agent.goal_merge_threshold * 100 / self.agent.map_resolution)

        if 0 <= map_x < self.agent.map_size and 0 <= map_y < self.agent.map_size:

            goal_gps_map_local = self.agent.goal_gps_map[
                max(map_y - thres, 0):
                min(map_y + thres, self.agent.map_size - 1),
                max(map_x - thres, 0):
                min(map_x + thres, self.agent.map_size - 1)
            ]
            timestamp_map_local = self.agent.goal_gps_timestamp_map[
                max(map_y - thres, 0):
                min(map_y + thres, self.agent.map_size - 1),
                max(map_x - thres, 0):
                min(map_x + thres, self.agent.map_size - 1)
            ]

            if goal_gps_map_local.max() > 0:
                # Update existing detection
                max_y, max_x = np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], \
                               np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]
                old_count = int(goal_gps_map_local[max_y, max_x])
                goal_gps_map_local[max_y, max_x] = goal_gps_map_local[max_y, max_x] + 1
                timestamp_map_local[max_y, max_x] = self.agent.total_steps
                print(f"[GPS记录] 目标位置map({map_y},{map_x}), 距离={temp_distance:.2f}m, "
                      f"在0.8m范围内找到已有检测, 计数: {old_count}→{old_count+1}")
            else:
                # First detection at this location
                self.agent.goal_gps_map[map_y, map_x] = 1
                self.agent.goal_gps_timestamp_map[map_y, map_x] = self.agent.total_steps
                print(f"[GPS记录] 目标位置map({map_y},{map_x}), 距离={temp_distance:.2f}m, "
                      f"首次检测到此位置, 计数: 0→1")

        self.agent.found_possible_goal = False

        # Extract crop for RAG
        box_np = box.cpu().numpy()
        crop = observations["rgb"][int(box_np[1]):int(box_np[3]), int(box_np[0]):int(box_np[2])]

        # Get detected label for debugging
        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        detected_label = obj_labels[original_idx] if original_idx < len(obj_labels) else self.agent.obj_goal

        print(f"[RAG Window] Adding detection: detected_label='{detected_label}', using caption='{self.agent.obj_goal}' (goal object)")

        # Get confidence
        original_scores = self.agent.current_obj_predictions.get_field("scores")
        initial_confidence = float(original_scores[original_idx]) if original_idx < len(original_scores) else 0.8
        capped_confidence = min(initial_confidence, 0.9)

        # Update sliding window - CRITICAL: use obj_goal as caption, not detector's label
        # This ensures we only save correct goal object names to the knowledge base
        detection_dict = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detector label
            'confidence': capped_confidence,
            'distance': temp_distance,
            'step': self.agent.total_steps
        }
        self.agent.rag_sliding_window.append(detection_dict)

        # Keep only latest N detections
        if len(self.agent.rag_sliding_window) > self.agent.rag_sliding_window_size:
            self.agent.rag_sliding_window.pop(0)

        # Store for potential latest detection update (after confirmation)
        # Also use obj_goal for consistency
        self.agent._temp_latest_detection = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detector label
            'confidence': capped_confidence,
            'distance': temp_distance,
            'verified': True,
            'map_x': map_x,  # Store map coordinates for false positive marking
            'map_y': map_y
        }

    def _select_most_recent_goal_position(self, detection_threshold):
        """
        Select the most recent goal position from goal_gps_map.

        Args:
            detection_threshold: Minimum detections required
        """
        positions_above_threshold = np.where(self.agent.goal_gps_map >= detection_threshold)

        if len(positions_above_threshold[0]) > 0:
            # Get timestamps for all positions above threshold
            timestamps = self.agent.goal_gps_timestamp_map[positions_above_threshold]

            # Find most recent position
            most_recent_idx = np.argmax(timestamps)
            goal_y_map = positions_above_threshold[0][most_recent_idx]  # row in map
            goal_x_map = positions_above_threshold[1][most_recent_idx]  # col in map

            # Reverse transform from map coordinates to GPS coordinates
            # Forward: map_x = map_size/10 - goal_gps[1]*100/resolution
            # Reverse: goal_gps[0] = (map_y - map_size/10) * resolution / 100
            #          goal_gps[1] = (map_size/10 - map_x) * resolution / 100
            goal_gps_0 = (goal_y_map - self.agent.map_size_cm/10) * self.agent.resolution / 100
            goal_gps_1 = (self.agent.map_size_cm/10 - goal_x_map) * self.agent.resolution / 100

            self.agent.goal_gps = np.array([goal_gps_0, goal_gps_1], dtype=float)
            print(f"[Goal Selection] Selected most recent detection at map position ({goal_y_map}, {goal_x_map}), "
                  f"count={int(self.agent.goal_gps_map[goal_y_map, goal_x_map])}, "
                  f"last_seen_step={int(self.agent.goal_gps_timestamp_map[goal_y_map, goal_x_map])}, "
                  f"threshold={detection_threshold}")
        else:
            # Fallback: use max count
            max_positions = np.where(self.agent.goal_gps_map == self.agent.goal_gps_map.max())
            goal_y_map = max_positions[0][0]
            goal_x_map = max_positions[1][0]

            # Reverse transform
            goal_gps_0 = (goal_y_map - self.agent.map_size_cm/10) * self.agent.resolution / 100
            goal_gps_1 = (self.agent.map_size_cm/10 - goal_x_map) * self.agent.resolution / 100
            self.agent.goal_gps = np.array([goal_gps_0, goal_gps_1], dtype=float)
            print(f"[Goal Selection] WARNING: No position meets threshold, using max count fallback at map({goal_y_map}, {goal_x_map})")

        self.agent.explanation = f"Goal found: {self.agent.obj_goal} (detection #{int(self.agent.found_goal_times)})"

        # Clean up temp detection (already transferred to rag_latest_goal_detection before RAG check)
        if hasattr(self.agent, '_temp_latest_detection'):
            delattr(self.agent, '_temp_latest_detection')

    def _verify_detection_with_rag(self):
        """
        Verify current goal detection using RAG manager.
        May reset goal detection if RAG rejects the detection.
        """
        is_verified, adjusted_conf, explanation, comparison_doc = self.agent.rag_manager.verify_detection(
            obj_category=self.agent.obj_goal,
            crop=self.agent.rag_latest_goal_detection['crop'],
            caption=self.agent.rag_latest_goal_detection['caption'],
            confidence=self.agent.rag_latest_goal_detection['confidence']
        )

        # Store comparison document for visualization
        self.agent.rag_comparison_doc = comparison_doc

        # Update confidence based on RAG verification
        self.agent.rag_latest_goal_detection['confidence'] = adjusted_conf
        self.agent.rag_latest_goal_detection['verified'] = is_verified

        print(f"[RAG Check] {explanation}")

        # If RAG rejects, check adjusted confidence against threshold
        if not is_verified:
            # Special handling for false positive rejection
            if adjusted_conf == 0.0:
                print(f"[RAG Check] 🚫 FALSE POSITIVE DETECTED! Resetting goal detection and re-entering FBE.")

                # Mark this position on goal_gps_map as false positive (-1) to prevent future detections
                if 'map_x' in self.agent.rag_latest_goal_detection and 'map_y' in self.agent.rag_latest_goal_detection:
                    map_x = self.agent.rag_latest_goal_detection['map_x']
                    map_y = self.agent.rag_latest_goal_detection['map_y']

                    if 0 <= map_x < self.agent.map_size and 0 <= map_y < self.agent.map_size:
                        # Mark this position and nearby area as false positive
                        thres = int(self.agent.goal_merge_threshold * 100 / self.agent.map_resolution)
                        y_start = max(map_y - thres, 0)
                        y_end = min(map_y + thres + 1, self.agent.map_size)
                        x_start = max(map_x - thres, 0)
                        x_end = min(map_x + thres + 1, self.agent.map_size)

                        self.agent.goal_gps_map[y_start:y_end, x_start:x_end] = -1
                        print(f"[False Positive Map] Marked position map({map_y},{map_x}) and nearby area "
                              f"({y_end-y_start}x{x_end-x_start} pixels) as false positive (value=-1)")

                self.agent.reset_goal_detection()
            else:
                # Regular rejection - check against threshold
                detector_threshold = self.agent.get_detector_confidence_threshold()
                if adjusted_conf < detector_threshold:
                    print(f"[RAG Check] ❌ Detection rejected: adjusted conf {adjusted_conf:.3f} < threshold {detector_threshold:.3f}")
                    self.agent.reset_goal_detection()
                else:
                    print(f"[RAG Check] ⚠️  RAG rejected but keeping goal: adjusted conf {adjusted_conf:.3f} >= threshold {detector_threshold:.3f}")
        else:
            print(f"[RAG Check] ✓ Detection accepted (adjusted conf: {adjusted_conf:.3f})")

    def _get_glip_real_label(self, prediction):
        """
        Convert GLIP prediction indices to string labels.

        Args:
            prediction: GLIP prediction object

        Returns:
            List of string labels
        """
        labels = prediction.get_field("labels").tolist()
        new_labels = []

        # Check if labels are already strings (FastSAM) or numeric indices (GLIP)
        if labels and isinstance(labels[0], str):
            # FastSAM already returns string labels, no conversion needed
            return labels

        # GLIP returns numeric indices, convert to string labels
        if self.agent.glip_demo and self.agent.glip_demo.entities and self.agent.glip_demo.plus:
            for i in labels:
                if i <= len(self.agent.glip_demo.entities):
                    new_labels.append(self.agent.glip_demo.entities[i - self.agent.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]

        return new_labels

    def _save_detection_images(self, image_bgr, observations, before_detection=True,
                               bboxes=None, labels=None, scores=None):
        """
        Save detection images (before and after).

        Args:
            image_bgr: Image in BGR format
            observations: Current observations
            before_detection: If True, save original image; if False, save with bboxes
            bboxes: Bounding boxes (numpy array)
            labels: Object labels (list of strings)
            scores: Confidence scores (numpy array)
        """
        import cv2
        import os
        from pathlib import Path

        # Create output directory
        output_dir = Path(f"data/glip_detections/episode_{self.agent.count_episodes}")
        output_dir.mkdir(parents=True, exist_ok=True)

        step = self.agent.total_steps

        if before_detection:
            # Save original image
            save_path = output_dir / f"step_{step:04d}_before.jpg"
            cv2.imwrite(str(save_path), image_bgr)

        else:
            # Draw bboxes and save
            image_with_bbox = image_bgr.copy()

            # Define colors for different categories (BGR format)
            np.random.seed(42)  # Fixed seed for consistent colors
            colors = [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(100)]

            for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                x1, y1, x2, y2 = bbox.astype(int)
                color = colors[i % len(colors)]

                # Draw rectangle
                cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, 2)

                # Prepare label text
                label_text = f"{label}: {score:.2f}"

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    image_with_bbox,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1  # Filled
                )

                # Draw text
                cv2.putText(
                    image_with_bbox,
                    label_text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1,
                    cv2.LINE_AA
                )

            # Save image with bboxes
            save_path = output_dir / f"step_{step:04d}_after.jpg"
            cv2.imwrite(str(save_path), image_with_bbox)
