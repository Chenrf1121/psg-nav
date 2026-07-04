import torch
import supervision as sv
import numpy as np
from collections import Counter
import math
from sklearn.cluster import DBSCAN

from ..node_class import ObjectNode, GroupNode, RoomNode
from ..edge_class import Edge

from utils.utils_sg_uq.merge_utils import filter_objects, gobs_to_detection_list, compute_spatial_similarities, merge_detections_to_objects
from itertools import product


class CoreFunc():
    def __init__(self):
        pass

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes
        
    def segment2d(self):
        if self.sam_variant == 'sam' or self.sam_variant == 'groundedsam':
            with torch.no_grad():
                result = self.get_sam_segmentation_dense(self.sam_variant, self.mask_generator, self.image_rgb)

                # Handle both old (4 values) and new (5 values) return formats
                if len(result) == 5:
                    mask, xyxy, conf, caption, distributions = result
                else:
                    mask, xyxy, conf, caption = result
                    distributions = None

                self.seg_xyxy = xyxy
                self.seg_caption = caption

            if caption is None:
                return
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            image_appear_efficiency = [''] * len(mask)

            result_dict = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                # "image_crops": image_crops,
                # "image_feats": image_feats,
                # "text_feats": text_feats,
                "image_appear_efficiency": image_appear_efficiency,
                "image_rgb": self.image_rgb,
                "caption": caption,
            }

            # Add distributions if available
            if distributions is not None:
                result_dict["distributions"] = distributions

            self.segment2d_results.append(result_dict)

        
    def caption_distribution(self, caps):
        if not caps:
            return {
                "mode": "object",
                "counts": {"object": 1},
                "probs": {"object": 1.0},
                "entropy": 0.0,
                "topk": ["object"]
            }
        counts = Counter([c.strip() for c in caps if c and c.strip()])
        N = sum(counts.values())
        probs = {k: v / N for k, v in counts.items()}
        entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
        mode = max(counts, key=counts.get)
        topk = [k for k, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        return {"mode": mode, "counts": dict(counts), "probs": probs, "entropy": entropy, "topk": topk}

    def mapping3d(self):
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]
        gobs = self.segment2d_results[-1]
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = self.pose_matrix,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
        )
        
        if len(fg_detection_list) == 0:
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            return
                
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        self.objects_post = filter_objects(self.cfg, self.objects)
    def get_caption(self, debug=False):
        if self.sam_variant == 'groundedsam':
            use_distribution = getattr(self, 'use_caption_distribution', False)

            for i, obj in enumerate(self.objects_post):
                if use_distribution:
                    # NEW METHOD: Use single-frame probability distribution
                    # Get the most recent detection's distribution
                    latest_frame_idx = obj["image_idx"][-1]
                    latest_mask_idx = obj["mask_idx"][-1]
                    result = self.segment2d_results[latest_frame_idx]

                    if 'distributions' in result and result['distributions'] is not None:
                        # Use the precomputed distribution from GroundingDINO
                        dist = result['distributions'][latest_mask_idx]
                    else:
                        # Fallback to old method if distributions not available
                        caps = []
                        for t in range(len(obj["image_idx"])):
                            cap = self.segment2d_results[obj["image_idx"][t]]['caption'][obj["mask_idx"][t]]
                            caps.append(cap)
                        dist = self.caption_distribution(caps)
                else:
                    # OLD METHOD: Multi-frame accumulation
                    caps = []
                    for t in range(len(obj["image_idx"])):
                        cap = self.segment2d_results[obj["image_idx"][t]]['caption'][obj["mask_idx"][t]]
                        caps.append(cap)
                    dist = self.caption_distribution(caps)

                obj['captions'] = [dist["mode"]]
                obj['caption_counts'] = dist.get("counts", {dist["mode"]: 1})
                obj['caption_probs'] = dist["probs"]
                obj['caption_entropy'] = dist["entropy"]
                obj['captions_sorted'] = dist["topk"]

                if debug:
                    print(f"\n[{i}] Object:")
                    if use_distribution:
                        print("  Method: Single-frame distribution")
                        print("  Distribution:", dist)
                    else:
                        print("  Method: Multi-frame accumulation")
                        print("  Captions:", caps)
                        print("  Counts:", dist.get("counts", {}))
                    print("  Mode:", dist["mode"])
                    print("  Entropy:", round(dist["entropy"], 4))
                    print("  Probabilities:", {k: round(v, 3) for k, v in dist['probs'].items()})

    def update_node(self):
        # Ensure all existing nodes have IDs
        for i, node in enumerate(self.nodes):
            if 'id' not in node.object:
                node.object['id'] = i

        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            caption_new = node.object['captions_sorted'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        for new_object in new_objects:
            new_node = ObjectNode()
            caption = new_object['captions'][0]
            new_node.set_caption(caption)

            # IMPORTANT: Add unique ID to object if not present
            if 'id' not in new_object:
                new_object['id'] = len(self.nodes)  # Use index as ID

            new_node.set_object(new_object)
            self.nodes.append(new_node)

        # get node.center and node.room
        # DEBUG: Track room assignments
        room_assignment_debug = []

        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)

            # Convert world coordinates to map pixel coordinates
            x = int(center[0] * 100 / self.map_resolution)
            y = int(center[1] * 100 / self.map_resolution)
            y = self.map_size - 1 - y

            node.set_center([x, y])

            # DEBUG: Check room_map values
            room_label = 0
            room_probs_sum = 0.0
            room_probs_max = 0.0

            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                # IMPORTANT: y has been flipped once during coordinate conversion (y = map_size - 1 - y_raw)
                # But for querying room_map, we need to use the unflipped coordinate
                # because room_map uses the original coordinate system
                y_unflipped = self.map_size - 1 - y

                room_probs = self.room_map[0, :, y_unflipped, x]
                room_probs_sum = room_probs.sum().item()
                room_probs_max = room_probs.max().item() if room_probs_sum > 0 else 0.0


                if room_probs_sum == 0:
                    room_label = 0  # Default to 'unknown' (index 0) if no room info
                else:
                    # argmax returns index in [0-8] for 9 detected room types
                    # Add 1 because 'unknown' is at index 0, detected rooms start at index 1
                    room_label = room_probs.argmax().item() + 1

                # DEBUG: Store for later printing
                if len(room_assignment_debug) < 5:  # Only store first 5 for debugging
                    room_assignment_debug.append({
                        'caption': node.caption,
                        'center': (x, y),
                        'room_probs_sum': room_probs_sum,
                        'room_probs_max': room_probs_max,
                        'room_label': room_label,
                        'room_name': self.rooms[room_label]
                    })
            else:
                room_label = 0  # Default to 'unknown'

            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
            if node.caption in self.obj_goal_sg:
                node.is_goal_node = True

    
    def update_edge(self):
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        for new_edge in new_edges:
            image = self.get_joint_image(new_edge.node1, new_edge.node2)
            if image is not None:
                prompt = self.prompt_relation.format(new_edge.node1.caption, new_edge.node2.caption)
                response = self.get_vlm_response(prompt=prompt, image=image)
                # Use intelligent extraction instead of simple cleaning
                new_edge.set_relation(response)
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.get_llm_response(prompt=prompt)
            relations = relations.split('\n')
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
            # discriminate all relation proposals
            self.free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
            for i, new_edge in enumerate(new_edges):
                if new_edge.relation == None or not self.discriminate_relation(new_edge):
                    new_edge.delete()
        self.edge_list = self.get_edges()

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)
                clusters = dbscan.fit_predict(centers)
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)


    def print_scenegraph_hierarchy(self):
        """
        Print the complete scene graph hierarchy showing:
        - All RoomNodes
        - All GroupNodes under each room
        - All ObjectNodes under each group
        - All edges between objects
        """
        print("\n" + "="*100)
        print("🌳 SCENE GRAPH HIERARCHY")
        print("="*100)

        # Statistics
        # Note: Use get_edges() instead of edge_list because edge_list is not updated
        all_edges = self.edge_list
        total_rooms = len(self.room_nodes)
        total_objects = len(self.nodes)
        total_edges = len(all_edges)
        total_groups = sum(len(room.group_nodes) for room in self.room_nodes)

        print(f"\n📊 STATISTICS:")
        print(f"   Total RoomNodes: {total_rooms}")
        print(f"   Total ObjectNodes: {total_objects}")
        print(f"   Total GroupNodes: {total_groups}")
        print(f"   Total Edges: {total_edges}")
        print()

        # Print hierarchy for each room
        for room_idx, room_node in enumerate(self.room_nodes):
            # Skip empty rooms
            if len(room_node.nodes) == 0:
                continue

            print("─" * 100)
            print(f"\n🏠 ROOM #{room_idx}: {room_node.caption.upper()}")
            print(f"   Total Objects: {len(room_node.nodes)}")
            print(f"   Total Groups: {len(room_node.group_nodes)}")

            # Print GroupNodes in this room
            if len(room_node.group_nodes) > 0:
                print(f"\n   📦 GROUP NODES ({len(room_node.group_nodes)}):")

                for group_idx, group_node in enumerate(room_node.group_nodes):
                    print(f"\n   ├─ GroupNode #{group_idx}")
                    print(f"   │  ├─ Number of objects: {len(group_node.nodes)}")
                    print(f"   │  ├─ Center: [{group_node.center[0]:.1f}, {group_node.center[1]:.1f}]")
                    print(f"   │  ├─ Center node: {group_node.center_node.caption if group_node.center_node else 'None'}")
                    print(f"   │  ├─ Probability: {group_node.probability:.3f}")

                    # Truncate caption if too long
                    caption_display = group_node.caption
                    if len(caption_display) > 80:
                        caption_display = caption_display[:77] + "..."
                    print(f"   │  ├─ Caption: {caption_display}")

                    # Print objects in this group
                    print(f"   │  │")
                    print(f"   │  ├─ 🔹 OBJECTS ({len(group_node.nodes)}):")
                    for obj_idx, obj_node in enumerate(group_node.nodes):
                        is_last_obj = (obj_idx == len(group_node.nodes) - 1)
                        prefix = "   │  │  └─" if is_last_obj else "   │  │  ├─"

                        print(f"{prefix} [{obj_idx+1}] {obj_node.caption}")
                        print(f"   │  │  {'   ' if is_last_obj else '│  '}   Position: [{obj_node.center[0]:.1f}, {obj_node.center[1]:.1f}]")
                        print(f"   │  │  {'   ' if is_last_obj else '│  '}   New node: {obj_node.is_new_node}")
                        print(f"   │  │  {'   ' if is_last_obj else '│  '}   Goal node: {obj_node.is_goal_node}")
                        print(f"   │  │  {'   ' if is_last_obj else '│  '}   # of edges: {len(obj_node.edges)}")

                    # Print edges within this group
                    print(f"   │  │")
                    print(f"   │  └─ 🔗 EDGES IN GROUP ({len(group_node.edges)}):")
                    if len(group_node.edges) > 0:
                        for edge_idx, edge in enumerate(list(group_node.edges)):
                            print(f"   │     [{edge_idx+1}] {edge.node1.caption} --[{edge.relation}]--> {edge.node2.caption}")
                    else:
                        print(f"   │     (no edges)")
                    print(f"   │")
            else:
                # Room has objects but no groups
                print(f"\n   ⚠️  No GroupNodes (objects not grouped)")
                print(f"\n   🔹 UNGROUPED OBJECTS ({len(room_node.nodes)}):")
                for obj_idx, obj_node in enumerate(room_node.nodes):
                    print(f"   ├─ [{obj_idx+1}] {obj_node.caption}")
                    print(f"   │  Position: [{obj_node.center[0]:.1f}, {obj_node.center[1]:.1f}]")
                    print(f"   │  # of edges: {len(obj_node.edges)}")

        # Print all edges in the scene graph
        print("\n" + "─" * 100)
        print(f"\n🔗 ALL EDGES IN SCENE GRAPH ({len(all_edges)}):")
        print("─" * 100)

        if len(all_edges) > 0:
            # Group edges by room for better readability
            edges_by_room = {}
            cross_room_edges = []

            for edge in all_edges:
                room1 = edge.node1.room_node
                room2 = edge.node2.room_node

                if room1 == room2 and room1 is not None:
                    # Both nodes in same room
                    room_name = room1.caption
                    if room_name not in edges_by_room:
                        edges_by_room[room_name] = []
                    edges_by_room[room_name].append(edge)
                else:
                    # Cross-room edge or node without room
                    cross_room_edges.append(edge)

            # Print edges by room
            for room_name, edges in edges_by_room.items():
                print(f"\n🏠 {room_name.upper()} ({len(edges)} edges):")
                for idx, edge in enumerate(edges):
                    relation = edge.relation if edge.relation else "None"
                    print(f"   [{idx+1}] {edge.node1.caption:20s} --[{relation:15s}]--> {edge.node2.caption}")

            # Print cross-room edges
            if len(cross_room_edges) > 0:
                print(f"\n🔀 CROSS-ROOM EDGES ({len(cross_room_edges)}):")
                for idx, edge in enumerate(cross_room_edges):
                    room1_name = edge.node1.room_node.caption if edge.node1.room_node else "Unknown"
                    room2_name = edge.node2.room_node.caption if edge.node2.room_node else "Unknown"
                    relation = edge.relation if edge.relation else "None"
                    print(f"   [{idx+1}] {edge.node1.caption} ({room1_name}) --[{relation}]--> {edge.node2.caption} ({room2_name})")
        else:
            print("   (no edges in scene graph)")

        print("\n" + "="*100)
        print("✅ SCENE GRAPH HIERARCHY PRINTED")
        print("="*100 + "\n")

    def _verify_no_duplicate_nodes_in_groups(self, room_node):
        """Verify that no ObjectNode appears in multiple GroupNodes."""
        seen_nodes = set()
        duplicates = []

        for group_idx, group_node in enumerate(room_node.group_nodes):
            for node in group_node.nodes:
                node_id = id(node)
                if node_id in seen_nodes:
                    duplicates.append((node, group_idx))
                seen_nodes.add(node_id)

        if duplicates:
            print("\n⚠️  WARNING: Found duplicate ObjectNodes in different groups!")
            for node, group_idx in duplicates:
                print(f"   Node '{node.caption}' at {node.center} appears in multiple groups including GroupNode {group_idx}")
            print()
        else:
            # Count total nodes in all groups
            total_in_groups = sum(len(gn.nodes) for gn in room_node.group_nodes)
            print(f"✅ Verification passed: No duplicate ObjectNodes across groups")
            print(f"   Total nodes in room: {len(room_node.nodes)}")
            print(f"   Total nodes in all groups: {total_in_groups}")
            if total_in_groups != len(room_node.nodes):
                print(f"   ⚠️  Mismatch: {len(room_node.nodes) - total_in_groups} nodes not in any group")
            print()

    def enumerate_group_node(self, group_node, min_prob_threshold=0.05):
        """
        Enumerate all possible caption combinations for a single GroupNode.

        This function handles the uncertainty in object recognition by generating all
        possible combinations of object captions within a group. Each ObjectNode has
        a caption distribution (e.g., bed: 0.8, sofa: 0.2), and this function creates
        one GroupNode instance for each valid combination.

        Args:
            group_node: The GroupNode to enumerate, containing multiple ObjectNodes
            min_prob_threshold: Minimum probability threshold. Object captions with
                               probability < threshold will be ignored. Default: 0.1

        Returns:
            List of GroupNode instances, each with a different caption combination
            and associated probability. Sorted by probability (descending order).

        Example:
            If group_node has 2 ObjectNodes:
                ObjectNode1: {bed: 0.8, sofa: 0.15, desk: 0.05}
                ObjectNode2: {table: 0.6, chair: 0.4}

            With min_prob_threshold=0.1:
                After filtering: ObjectNode1 -> [bed: 0.8, sofa: 0.15] * norm
                                ObjectNode2 -> [table: 0.6, chair: 0.4]

            Returns 4 GroupNodes:
                1. GroupNode([bed, table], prob=0.8*0.6=0.48)
                2. GroupNode([bed, chair], prob=0.8*0.4=0.32)
                3. GroupNode([sofa, table], prob=0.15*0.6=0.09)
                4. GroupNode([sofa, chair], prob=0.15*0.4=0.06)
        """
        if len(group_node.nodes) == 0:
            return [group_node]

        # Extract caption distributions for each ObjectNode in the group
        all_captions = []  # List of caption lists, one per ObjectNode
        all_probs = []     # List of probability lists, one per ObjectNode

        for node in group_node.nodes:
            # Try to get caption distribution from the object
            if hasattr(node, 'object') and node.object is not None:
                if 'captions_sorted' in node.object and 'caption_probs' in node.object:
                    # Extract sorted captions and their probabilities
                    captions_sorted = node.object['captions_sorted']
                    probs_dict = node.object['caption_probs']

                    # Filter captions by probability threshold
                    filtered_captions = []
                    filtered_probs = []
                    for caption in captions_sorted:
                        prob = probs_dict.get(caption, 0.0)
                        if prob >= min_prob_threshold:
                            filtered_captions.append(caption)
                            filtered_probs.append(prob)

                    # If all captions filtered out, keep at least the top one
                    if len(filtered_captions) == 0:
                        filtered_captions = [captions_sorted[0]]
                        filtered_probs = [probs_dict.get(captions_sorted[0], 1.0)]

                    # Renormalize probabilities to sum to 1.0 (DISABLED - keep raw probabilities)
                    # Normalization removed to preserve absolute confidence information.
                    all_captions.append(filtered_captions)
                    all_probs.append(filtered_probs)

                elif 'captions' in node.object:
                    # Fallback: uniform probability distribution
                    captions = node.object['captions']
                    if len(captions) > 0:
                        uniform_prob = 1.0 / len(captions)

                        # Filter by threshold
                        if uniform_prob >= min_prob_threshold:
                            all_captions.append(captions)
                            all_probs.append([uniform_prob] * len(captions))
                        else:
                            # Keep at least one caption
                            all_captions.append([captions[0]])
                            all_probs.append([uniform_prob])
                    else:
                        all_captions.append([node.caption])
                        all_probs.append([1.0])
                else:
                    # No caption distribution info, use current caption
                    all_captions.append([node.caption])
                    all_probs.append([1.0])
            else:
                # No object info, use current caption with probability 1.0
                all_captions.append([node.caption])
                all_probs.append([1.0])

        # Generate all combinations of captions across ObjectNodes
        caption_combinations = list(product(*all_captions))
        prob_combinations = list(product(*all_probs))

        # Create a GroupNode for each combination
        enumerated_groups = []
        for caption_combo, prob_combo in zip(caption_combinations, prob_combinations):
            # Calculate combined probability (product of individual probabilities)
            combined_prob = 1.0
            for p in prob_combo:
                combined_prob *= p

            # Skip combinations with very low combined probability
            if combined_prob < min_prob_threshold:
                continue

            # Create a copy of the original group node
            new_group = group_node.copy()

            # Update each ObjectNode in the group with the new caption
            for i, (node, new_caption) in enumerate(zip(new_group.nodes, caption_combo)):
                new_node = node.copy()
                new_node.caption = new_caption
                new_group.nodes[i] = new_node

            new_group.probability = combined_prob

            # Regenerate group caption and edges based on new node captions
            new_group.get_graph()

            enumerated_groups.append(new_group)

        # Sort by probability (descending order)
        enumerated_groups.sort(key=lambda g: g.probability, reverse=True)

        return enumerated_groups

    def enumerate_all_group_nodes(self, room_node, top_k_per_group=None, min_prob_threshold=0.1):
        """
        Enumerate all possible combinations of GroupNodes within a single room.

        This function generates the Cartesian product of all GroupNode enumerations in the room.
        Each combination represents one possible interpretation of all groups in the room,
        where each group_idx selects exactly one enumeration.

        Args:
            room_node: The RoomNode containing multiple GroupNodes
            top_k_per_group: Optional, keep only top-k enumerations per GroupNode before
                           computing Cartesian product (to avoid combinatorial explosion)
            min_prob_threshold: Minimum probability threshold for filtering

        Returns:
            List of dicts, each containing:
                - 'group_combo': List of GroupNodes (one per group_idx)
                - 'probability': Joint probability (product of individual probabilities)
                - 'room_caption': Caption of the room
            Sorted by probability (descending)

        Example:
            bedroom has 2 GroupNodes:
                - GroupNode[idx=0]: 3 enumerations
                    - [bed, nightstand] prob=0.6
                    - [bed, table] prob=0.3
                    - [sofa, nightstand] prob=0.1
                - GroupNode[idx=1]: 2 enumerations
                    - [lamp, clock] prob=0.8
                    - [lamp, book] prob=0.2

            Returns 6 combinations (3 × 2):
                1. {group_combo: [[bed,nightstand], [lamp,clock]], prob: 0.6*0.8=0.48}
                2. {group_combo: [[bed,nightstand], [lamp,book]], prob: 0.6*0.2=0.12}
                3. {group_combo: [[bed,table], [lamp,clock]], prob: 0.3*0.8=0.24}
                4. {group_combo: [[bed,table], [lamp,book]], prob: 0.3*0.2=0.06}
                5. {group_combo: [[sofa,nightstand], [lamp,clock]], prob: 0.1*0.8=0.08}
                6. {group_combo: [[sofa,nightstand], [lamp,book]], prob: 0.1*0.2=0.02}

            Note: Each combination has exactly one selection from each group_idx
        """
        if len(room_node.group_nodes) == 0:
            return []

        # Step 1: Enumerate each GroupNode independently
        all_group_enumerations = []  # List of lists
        for group_node in room_node.group_nodes:
            # Get all possible caption combinations for this group
            enumerated = self.enumerate_group_node(group_node, min_prob_threshold=min_prob_threshold)

            # Optionally keep only top-k per group to prevent combinatorial explosion
            if top_k_per_group is not None:
                enumerated = enumerated[:top_k_per_group]

            all_group_enumerations.append(enumerated)

        # Step 2: Generate Cartesian product across all GroupNodes
        # Each element in the product is a tuple of GroupNodes, one from each group_idx
        all_combinations = list(product(*all_group_enumerations))

        # Step 3: Create result structures with metadata
        combinations_with_metadata = []
        for combo_tuple in all_combinations:
            # Calculate joint probability (assuming independence between groups)
            joint_prob = 1.0
            for group in combo_tuple:
                joint_prob *= group.probability

            # Filter out very low probability combinations
            if joint_prob < min_prob_threshold:
                continue

            combination_dict = {
                'room_caption': room_node.caption,
                'group_combo': list(combo_tuple),  # List of GroupNodes
                'probability': joint_prob,
                'num_groups': len(combo_tuple),
            }
            combinations_with_metadata.append(combination_dict)

        # Step 4: Sort by joint probability (descending)
        combinations_with_metadata.sort(key=lambda x: x['probability'], reverse=True)

        return combinations_with_metadata

    def debug_enumerate_group_nodes(self, room_node, top_k_per_group=3):
        """
        Debug function to print enumerated GroupNodes.

        Args:
            room_node: RoomNode to enumerate
            top_k_per_group: Number of top combinations to show per group
        """
        print("\n" + "="*80)
        print(f"🔢 ENUMERATING GroupNodes for ROOM: {room_node.caption}")
        print("="*80)

        if len(room_node.group_nodes) == 0:
            print("No GroupNodes to enumerate.")
            return

        for group_idx, group_node in enumerate(room_node.group_nodes):
            print(f"\n📦 Original GroupNode {group_idx}:")
            print(f"   Caption: {group_node.caption}")
            print(f"   Objects: {[node.caption for node in group_node.nodes]}")

            # Show caption distributions for each object
            print(f"   Caption distributions:")
            for node_idx, node in enumerate(group_node.nodes):
                if hasattr(node, 'object') and node.object is not None:
                    if 'caption_probs' in node.object:
                        probs = node.object['caption_probs']
                        print(f"      Object {node_idx}: {probs}")
                    else:
                        print(f"      Object {node_idx}: {{'{node.caption}': 1.0}}")
                else:
                    print(f"      Object {node_idx}: {{'{node.caption}': 1.0}}")

            # Enumerate this group
            enumerated = self.enumerate_group_node(group_node)
            print(f"\n   💡 Enumerated {len(enumerated)} possible combinations:")

            # Show top-k
            for enum_idx, enum_group in enumerate(enumerated[:top_k_per_group]):
                captions = [node.caption for node in enum_group.nodes]
                print(f"      {enum_idx+1}. {captions} (prob: {enum_group.probability:.4f})")
                print(f"         Caption: {enum_group.caption}")

            if len(enumerated) > top_k_per_group:
                print(f"      ... and {len(enumerated) - top_k_per_group} more combinations")

        # Now show combined enumerations across all groups
        print(f"\n🌐 Combined enumerations across all {len(room_node.group_nodes)} GroupNodes:")
        all_combos = self.enumerate_all_group_nodes(room_node, top_k=2)  # Top-2 per group

        total_combos = len(all_combos)
        print(f"   Total combinations: {total_combos}")
        print(f"   Top 5 most probable:")

        for combo_idx, (group_list, combined_prob) in enumerate(all_combos[:5]):
            print(f"      {combo_idx+1}. Combined prob: {combined_prob:.4f}")
            for group_idx, group in enumerate(group_list):
                captions = [node.caption for node in group.nodes]
                print(f"         Group {group_idx}: {captions} (prob: {group.probability:.4f})")

        print("\n" + "="*80 + "\n")

