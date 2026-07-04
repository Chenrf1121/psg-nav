import numpy as np
from typing import List, Tuple, Optional
from utils.utils_landmark import *
from scipy.spatial import Voronoi
from scipy.spatial._qhull import QhullError
from scipy.ndimage import binary_erosion
import cv2
import networkx as nx
from sklearn.cluster import DBSCAN


class LandmarkMap:
    """Maintain a sparse landmark graph extracted from frontier points.

    Nodes are stored as (row, col) coordinates (map grid indices).
    Edges are stored as pairs of node coordinate tuples: ((r1,c1),(r2,c2)).

    Methods:
      - update(points): build nodes/edges from Nx2 frontier points
      - get_nodes()/get_edges()
      - draw_on_image(img, left, top): draw landmarks/edges onto an image with given top-left offset
    """

    def __init__(self, max_nodes: int = 60, min_dist: float = 8.0, knn: int = 3, method: str = "fast", save_visualization: bool = True, save_steps: bool = False):
        self.max_nodes = int(max_nodes)
        self.min_dist = float(min_dist)
        self.knn = int(knn)
        self.method = method
        self.save_visualization = save_visualization  # Control whether to auto-save visualizations
        self.save_steps = save_steps  # Control whether to save intermediate steps
        self.step_counter = 0  # Counter for step visualization
        self.nodes: Optional[np.ndarray] = None
        self.edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        # Semantic information for each landmark
        self.semantic_info: List[dict] = []
        # Statistics
        self.stats = {'total_generated': 0, 'filtered_isolated': 0}

        # Topological graph (for Voronoi method)
        self.graph: Optional[nx.Graph] = None  # NetworkX graph for topological navigation
        self.node_id_to_landmark_idx: dict = {}  # Map from graph node ID to landmark index
        self.landmark_idx_to_node_id: dict = {}  # Map from landmark index to graph node ID

    def update(self, frontiers, fea_map, obs_map, traversible, agent_position=None, agent_trajectory_map=None):
        self.nodes = None
        self.edges = []
        self.semantic_info = []

        try:
            pts = np.asarray(frontiers)
        except Exception:
            return None

        if pts is None or pts.size == 0:
            # nothing to do
            self.nodes = np.zeros((0, 2), dtype=float)
            return None

        # determine map size from obs_map or fea_map
        def to_numpy(a):
            if a is None:
                return None
            if isinstance(a, np.ndarray):
                return a
            try:
                # handle torch tensors
                if hasattr(a, "cpu"):
                    na = a.cpu().numpy()
                    return na
            except Exception:
                pass
            return np.asarray(a)

        obs = to_numpy(obs_map)
        fea = to_numpy(fea_map)

        if self.method == "voronoi":
            return self._update_voronoi(pts, fea, obs, traversible, agent_position, agent_trajectory_map)
        else:
            return self._update_fast(pts, traversible, agent_position, agent_trajectory_map, fea)

    def _update_voronoi(self, pts, fea_map, obs_map, traversible, agent_position=None, agent_trajectory_map=None):
        """Voronoi-based landmark generation following CogNav approach.

        IMPORTANT: Only generate landmarks in explored free space (浅灰色区域 / light gray area).
        - fea_map (fbe_free_map): Explored free space where landmarks should be generated
        - obs_map (full_map): Internal obstacles
        - traversible: Theoretical navigable region (larger than explored free space)

        This method implements the Reduced Voronoi Diagram (RVD) approach:
        1. Extract boundaries from EXPLORED FREE SPACE (not traversible)
        2. Generate Generalized Voronoi Diagram (GVD) from boundaries
        3. Create NetworkX graph from Voronoi vertices and edges
        4. Simplify graph by removing degree-2 nodes (RVD generation)
        5. Keep only intersection nodes (degree > 2) and leaf nodes (degree = 1)

        Args:
            pts: Frontier points (for visualization and fallback)
            fea_map: fbe_free_map - EXPLORED free space map (浅灰色区域)
            obs_map: full_map - Internal obstacle map
            traversible: Binary map where 1=free space, 0=obstacle (theoretical navigable region)
            agent_position: Current agent position [row, col]
            agent_trajectory_map: Map of visited areas

        Returns:
            None (updates self.nodes and self.edges)
        """
        # Reset nodes and edges
        self.nodes = None
        self.edges = []

        # Ensure obs_map and fea_map are 2D numpy arrays (squeeze extra dimensions if needed)
        if obs_map is not None:
            while obs_map.ndim > 2:
                obs_map = obs_map[0]  # Remove batch/channel dimensions
        if fea_map is not None:
            while fea_map.ndim > 2:
                fea_map = fea_map[0]  # Remove batch/channel dimensions

        if fea_map is None or obs_map is None:
            self.nodes = np.zeros((0, 2), dtype=float)
            return None

        try:
            # CRITICAL: Use fbe_free_map (fea_map) as the PRIMARY map
            # This ensures landmarks are ONLY generated in explored free space

            # Create free space map and path map (free space - obstacles)
            free_space_map = (fea_map > 0.5).astype(np.uint8)

            if free_space_map.sum() == 0:
                self.nodes = np.zeros((0, 2), dtype=float)
                return None

            # Dilate obstacles for safety margin and create path map
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            obs_map_2d = obs_map.astype(np.uint8)
            obs_dilated = cv2.dilate(obs_map_2d, kernel)
            path_map = free_space_map.copy()
            path_map[obs_dilated > 0] = 0


            if path_map.sum() == 0:
                self.nodes = np.zeros((0, 2), dtype=float)
                return None

            # Find largest connected component and extract its boundary points
            from scipy.ndimage import label as scipy_label
            labeled_path_map, num_components = scipy_label(path_map)

            if num_components == 0:
                self.nodes = np.zeros((0, 2), dtype=float)
                return None

            # Keep only the largest connected component
            component_sizes = np.bincount(labeled_path_map.ravel())
            component_sizes[0] = 0  # Ignore background
            largest_component_label = component_sizes.argmax()
            path_map = (labeled_path_map == largest_component_label).astype(np.uint8)

            # Extract boundary points via erosion
            eroded = binary_erosion(path_map, iterations=1).astype(np.uint8)
            boundary_map = (path_map - eroded).astype(np.uint8)
            rows, cols = np.where(boundary_map == 1)
            boundaries = np.array(list(zip(cols, rows)))  # (x, y) = (col, row)

            boundary_sample_rate = 9  # Adjust to control density (2 = keep 1/2 of points)
            if len(boundaries) > 100:
                boundaries = boundaries[::boundary_sample_rate]


            if len(boundaries) < 9:
                # Not enough points for Voronoi diagram - use frontier points as fallback
                if pts is not None and len(pts) > 0:
                    # Filter frontier points to only keep those in explored free space
                    valid_frontiers = []
                    h, w = free_space_map.shape
                    for pt in pts:
                        r, c = int(pt[0]), int(pt[1])
                        if 0 <= r < h and 0 <= c < w and free_space_map[r, c] == 1:
                            valid_frontiers.append(pt)

                    if len(valid_frontiers) > 0:
                        self.nodes = np.array(valid_frontiers, dtype=float)
                    else:
                        self.nodes = np.zeros((0, 2), dtype=float)
                    return None
                else:
                    self.nodes = np.zeros((0, 2), dtype=float)
                    return None

            # Generate Voronoi diagram and filter invalid vertices
            try:
                vor = Voronoi(boundaries.astype(float))
                # Filter vertices: keep only those in explored free space and away from obstacles
                vor_filtered = self._filter_voronoi_vertices_strict(
                    vor, path_map, obs_dilated, free_space_map
                )

            except QhullError as e:
                # Fallback to frontier points if Voronoi generation fails
                if pts is not None and len(pts) > 0:
                    valid_frontiers = []
                    h, w = free_space_map.shape
                    for pt in pts:
                        r, c = int(pt[0]), int(pt[1])
                        if 0 <= r < h and 0 <= c < w and free_space_map[r, c] == 1:
                            valid_frontiers.append(pt)
                    self.nodes = np.array(valid_frontiers, dtype=float) if len(valid_frontiers) > 0 else np.zeros((0, 2), dtype=float)
                else:
                    self.nodes = np.zeros((0, 2), dtype=float)
                return None

            if len(vor_filtered['vertices']) == 0 or len(vor_filtered['edges']) == 0:
                if pts is not None and len(pts) > 0:
                    valid_frontiers = []
                    h, w = free_space_map.shape
                    for pt in pts:
                        r, c = int(pt[0]), int(pt[1])
                        if 0 <= r < h and 0 <= c < w and free_space_map[r, c] == 1:
                            valid_frontiers.append(pt)

                    if len(valid_frontiers) > 0:
                        self.nodes = np.array(valid_frontiers, dtype=float)
                    else:
                        self.nodes = np.zeros((0, 2), dtype=float)
                    return None
                else:
                    self.nodes = np.zeros((0, 2), dtype=float)
                    return None

            # Build NetworkX graph, remove obstacle edges, and simplify by removing degree-2 nodes
            G = nx.Graph()
            for i, vertex in enumerate(vor_filtered['vertices']):
                G.add_node(i, pos=(vertex[0], vertex[1]))
            G.add_edges_from(vor_filtered['edges'])

            # Remove edges that pass through obstacles
            G = self._remove_obstacle_edges_strict(G, obs_dilated, path_map)

            if len(G.nodes()) == 0:
                if pts is not None and len(pts) > 0:
                    h, w = free_space_map.shape
                    valid_frontiers = [pt for pt in pts if 0 <= int(pt[0]) < h and 0 <= int(pt[1]) < w and free_space_map[int(pt[0]), int(pt[1])] == 1]
                    self.nodes = np.array(valid_frontiers, dtype=float) if valid_frontiers else np.zeros((0, 2), dtype=float)
                else:
                    self.nodes = np.zeros((0, 2), dtype=float)
                return None

            # Simplify graph: remove degree-2 nodes to generate RVD (Reduced Voronoi Diagram)
            G_simplified = self._simplify_graph_remove_degree2(G)


            if len(G_simplified.nodes()) == 0:
                if pts is not None and len(pts) > 0:
                    valid_frontiers = []
                    h, w = free_space_map.shape
                    for pt in pts:
                        r, c = int(pt[0]), int(pt[1])
                        if 0 <= r < h and 0 <= c < w and free_space_map[r, c] == 1:
                            valid_frontiers.append(pt)

                    if len(valid_frontiers) > 0:
                        self.nodes = np.array(valid_frontiers, dtype=float)
                    else:
                        self.nodes = np.zeros((0, 2), dtype=float)
                    return None
                else:
                    self.nodes = np.zeros((0, 2), dtype=float)
                    return None

            # Extract intersection nodes (degree >= 3) as final landmarks
            # Note: Allowing disconnected topology - all components are kept
            G_largest = G_simplified  # Use all components (not filtering to largest)

            landmarks = []
            landmark_metadata = []

            for node_id in G_largest.nodes():
                if 'pos' not in G_largest.nodes[node_id]:
                    continue

                node_pos = G_largest.nodes[node_id]['pos']
                degree = G_largest.degree(node_id)

                # Keep intersection nodes (degree >= 3) AND leaf nodes (degree == 1)
                # Skip only degree-2 (chain) nodes
                if degree == 2:
                    continue

                # Convert (col, row) to (row, col)
                landmarks.append([int(node_pos[1]), int(node_pos[0])])

                # Determine landmark type based on degree
                if degree == 1:
                    landmark_type = 'leaf'
                else:  # degree >= 3
                    landmark_type = 'intersection'

                landmark_metadata.append({
                    'type': landmark_type,
                    'degree': degree,
                    'node_id': node_id
                })

            if len(landmarks) == 0:
                self.nodes = np.zeros((0, 2), dtype=int)
                return None

            self.nodes = np.array(landmarks, dtype=int)
            self.semantic_info = landmark_metadata

            # Create mapping between landmark indices and graph node IDs
            self.node_id_to_landmark_idx = {}
            self.landmark_idx_to_node_id = {}
            for landmark_idx, meta in enumerate(self.semantic_info):
                node_id = meta['node_id']
                self.node_id_to_landmark_idx[node_id] = landmark_idx
                self.landmark_idx_to_node_id[landmark_idx] = node_id

            # Store the complete graph (all components)
            self.graph = G_largest.copy()

            # Extract edges from graph for visualization
            self.edges = []
            pos = nx.get_node_attributes(G_largest, 'pos')
            for edge in G_largest.edges():
                node1, node2 = edge
                if 'pos' in G_largest.nodes[node1] and 'pos' in G_largest.nodes[node2]:
                    pos1 = G_largest.nodes[node1]['pos']
                    pos2 = G_largest.nodes[node2]['pos']
                    # Convert (col, row) to (row, col) for consistency
                    edge_tuple = ((pos1[1], pos1[0]), (pos2[1], pos2[0]))
                    self.edges.append(edge_tuple)

            # Track statistics
            self.stats['total_generated'] = len(self.nodes)
            self.stats['leaf_nodes'] = sum(1 for meta in self.semantic_info if meta['type'] == 'leaf')
            self.stats['intersection_nodes'] = sum(1 for meta in self.semantic_info if meta['type'] == 'intersection')

            # Save visualization after each update (if enabled)
            if self.save_visualization:
                # Use free_space_map as traversible (explored free space)
                # Use obs_dilated as obstacle map
                self._save_voronoi_visualization(
                    traversible=free_space_map,
                    obs_map=obs_dilated,
                    agent_position=agent_position,
                    frontier_points=pts
                )

            return None

        except Exception as e:
            print(f"Error in Voronoi landmark generation: {e}")
            import traceback
            traceback.print_exc()
            self.nodes = np.zeros((0, 2), dtype=float)
            return None

    def _filter_voronoi_vertices_strict(self, vor, path_map, obstacles_map, free_space_map):
        """Filter Voronoi vertices with STRICT constraints - only keep vertices in explored free space.

        Args:
            vor: scipy.spatial.Voronoi object
            path_map: Binary map of navigable area (free space - obstacles)
            obstacles_map: Binary obstacle map
            free_space_map: Binary map of explored free space (fbe_free_map > 0.5)

        Returns:
            Dict with 'vertices' (list of filtered vertices) and 'edges' (list of valid edge indices)
        """
        vertices = vor.vertices
        ridges = vor.ridge_vertices

        h, w = path_map.shape

        # Filter vertices: keep ONLY those in explored free space
        valid_indices = []
        for i, vertex in enumerate(vertices):
            x, y = vertex[0], vertex[1]

            # Convert to integer coordinates
            x_int, y_int = int(round(x)), int(round(y))

            # Check if within map bounds
            if not (0 <= x_int < w and 0 <= y_int < h):
                continue

            # CRITICAL CHECK 1: Must be in explored free space
            if free_space_map[y_int, x_int] != 1:
                continue

            # CRITICAL CHECK 2: Must be in navigable area (path_map)
            if path_map[y_int, x_int] != 1:
                continue

            # CRITICAL CHECK 3: Check distance to obstacles
            # Ensure vertex is at least 4 pixels away from any obstacle
            y_min, y_max = max(0, y_int - 2), min(h, y_int + 3)
            x_min, x_max = max(0, x_int - 2), min(w, x_int + 3)
            if np.any(obstacles_map[y_min:y_max, x_min:x_max] > 0):
                continue

            valid_indices.append(i)

        # Create mapping from old to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

        # Filter vertices
        filtered_vertices = vertices[valid_indices]

        # Filter edges: keep only edges where both vertices are valid
        filtered_edges = []
        for ridge in ridges:
            if ridge[0] >= 0 and ridge[1] >= 0:  # Ignore infinite edges
                if ridge[0] in index_map and ridge[1] in index_map:
                    new_edge = (index_map[ridge[0]], index_map[ridge[1]])
                    filtered_edges.append(new_edge)

        return {'vertices': filtered_vertices, 'edges': filtered_edges}

    def _remove_obstacle_edges_strict(self, G, obstacles_map, free_space_map):
        """Remove edges that pass through obstacles or leave explored free space.

        Args:
            G: NetworkX graph with 'pos' attributes (col, row)
            obstacles_map: Binary obstacle map
            free_space_map: Binary map of explored free space

        Returns:
            Graph with invalid edges removed
        """
        pos = nx.get_node_attributes(G, 'pos')
        edges_to_remove = []

        for edge in G.edges():
            node1_pos = pos[edge[0]]
            node2_pos = pos[edge[1]]

            # Check if line between nodes passes through obstacle or leaves free space
            # Use Bresenham's line algorithm via skimage
            from skimage.draw import line

            x1, y1 = int(node1_pos[0]), int(node1_pos[1])
            x2, y2 = int(node2_pos[0]), int(node2_pos[1])

            rr, cc = line(y1, x1, y2, x2)

            # Check if any point on the line crosses an obstacle
            if np.any(obstacles_map[rr, cc] > 0):
                edges_to_remove.append(edge)
                continue

            # CRITICAL: Check if any point leaves explored free space
            if not np.all(free_space_map[rr, cc] == 1):
                edges_to_remove.append(edge)

        # Remove identified edges
        G.remove_edges_from(edges_to_remove)

        return G

    def _simplify_graph_remove_degree2(self, G):
        """Simplify graph by removing degree-2 nodes (chain nodes).

        This is the core RVD generation step from CogNav.
        Only keeps intersection nodes (degree >= 3) and leaf nodes (degree == 1).

        Args:
            G: NetworkX graph

        Returns:
            Simplified graph with degree-2 nodes removed
        """
        H = G.copy()

        # Find all degree-2 nodes
        degree2_nodes = [node for node in H.nodes() if H.degree(node) == 2]

        # Iteratively remove degree-2 nodes
        for node in degree2_nodes:
            # Check if node still has degree 2 (might have changed during iteration)
            if node not in H.nodes() or H.degree(node) != 2:
                continue

            neighbors = list(H.neighbors(node))

            if len(neighbors) == 2:
                # Connect the two neighbors directly
                H.add_edge(neighbors[0], neighbors[1])

                # Remove the degree-2 node
                H.remove_node(node)

        # Remove isolated nodes (degree 0)
        isolated_nodes = [node for node in H.nodes() if H.degree(node) == 0]
        H.remove_nodes_from(isolated_nodes)

        return H

    def _update_fast(self, pts, traversible, agent_position, agent_trajectory_map=None, fbe_free_map=None):
        """Fast update method: Filter noise, compute midpoints of boundary chains, and then cluster them."""

        # Reset nodes and edges
        self.nodes = None
        self.edges = []

        if pts is None or pts.size == 0:
            self.nodes = np.zeros((0, 2), dtype=float)
            return None

        # Step 1: Create a binary map and find connected components
        h, w = np.max(pts, axis=0) + 1  # Determine the size of the map
        binary_map = np.zeros((h, w), dtype=np.uint8)
        binary_map[pts[:, 0], pts[:, 1]] = 1

        # Find connected components (boundary chains)
        num_labels, labels = cv2.connectedComponents(binary_map)

        midpoints = []
        for label in range(1, num_labels):  # Skip the background label (0)
            chain_points = np.argwhere(labels == label)

            # Filter out chains with fewer than 2 points (noise)
            if chain_points.shape[0] < 4:
                continue

            # Compute the midpoint of the chain
            midpoint = np.mean(chain_points, axis=0)
            midpoints.append(midpoint)

        if len(midpoints) == 0:
            self.nodes = np.zeros((0, 2), dtype=float)
            return None

        midpoints = np.array(midpoints)

        # Step 2: Cluster midpoints using DBSCAN
        clustering = DBSCAN(eps=self.min_dist, min_samples=1).fit(midpoints)
        labels = clustering.labels_

        # Step 3: Collect representative points for each cluster
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise points

        representative_points = []
        for label in unique_labels:
            cluster_points = midpoints[labels == label]
            if cluster_points.size == 0:
                continue

            # Choose the centroid as the representative point
            centroid = np.mean(cluster_points, axis=0)
            representative_points.append((float(centroid[0]), float(centroid[1])))

        # Combine isolated midpoints and clustered centroids
        isolated_midpoints = midpoints[labels == -1]
        all_landmarks = np.vstack([isolated_midpoints, np.asarray(representative_points, dtype=float)])

        # Store the landmarks as nodes
        self.nodes = all_landmarks

        # Track statistics and filter isolated landmarks
        self.stats['total_generated'] = len(self.nodes)
        if agent_position is not None and traversible is not None:
            self.nodes = self._filter_isolated_landmarks(self.nodes, agent_position, traversible)
            self.stats['filtered_isolated'] = self.stats['total_generated'] - len(self.nodes)

        # Filter landmarks near agent trajectory (visited areas)
        if agent_trajectory_map is not None and self.nodes is not None and len(self.nodes) > 0:
            before_count = len(self.nodes)
            self.nodes = self._filter_trajectory_landmarks(self.nodes, agent_trajectory_map)
            self.stats['filtered_trajectory'] = before_count - len(self.nodes)

        # Filter landmarks that are too far from explored free areas
        if fbe_free_map is not None and self.nodes is not None and len(self.nodes) > 0:
            before_count = len(self.nodes)
            self.nodes = self._filter_distant_landmarks(self.nodes, fbe_free_map, max_distance=30)
            self.stats['filtered_distant'] = before_count - len(self.nodes)

        return None

    def _filter_isolated_landmarks(self, landmarks, agent_position, traversible):
        """Filter out landmarks in disconnected regions from agent.

        Args:
            landmarks: Nx2 array of landmark positions (row, col)
            agent_position: [row, col] of agent position
            traversible: Binary map where 1=free space, 0=obstacle

        Returns:
            Filtered array of landmarks that are connected to agent's position
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros((0, 2), dtype=float)

        try:
            from scipy.ndimage import label, binary_erosion

            # Label connected components in traversible map
            # traversible values > 0 indicate free space
            binary_map = (traversible > 0).astype(int)

            # Apply erosion to remove thin connections (1-2 pixel wide lines)
            # This ensures only truly navigable paths are considered connected
            eroded_map = binary_erosion(binary_map, iterations=3).astype(int)

            labeled, num_features = label(eroded_map)

            # Get agent's connected component label
            agent_r, agent_c = int(agent_position[0]), int(agent_position[1])

            # Handle boundary case: agent_position might need +1 offset if traversible has boundary
            # Check if traversible has boundary by comparing shape
            h, w = traversible.shape
            if agent_r + 1 < h and agent_c + 1 < w:
                # Try with +1 offset first (in case traversible has boundary)
                agent_label = labeled[agent_r + 1, agent_c + 1]
            else:
                # Use direct coordinates
                agent_label = labeled[min(agent_r, h-1), min(agent_c, w-1)]

            # Filter landmarks to keep only those in the same connected component
            reachable_landmarks = []
            for lm in landmarks:
                lm_r, lm_c = int(lm[0]), int(lm[1])

                # Apply same offset logic for landmarks
                if lm_r + 1 < h and lm_c + 1 < w:
                    lm_label = labeled[lm_r + 1, lm_c + 1]
                else:
                    lm_label = labeled[min(lm_r, h-1), min(lm_c, w-1)]

                if lm_label == agent_label and lm_label > 0:
                    reachable_landmarks.append(lm)

            if len(reachable_landmarks) == 0:
                return np.zeros((0, 2), dtype=float)

            return np.array(reachable_landmarks, dtype=float)

        except Exception as e:
            # If scipy is not available or any error occurs, return all landmarks
            print(f"Warning: Could not filter isolated landmarks: {e}")
            return landmarks

    def _filter_trajectory_landmarks(self, landmarks, agent_trajectory_map, exclusion_radius=3):
        """Filter out landmarks near agent's visited trajectory.

        This prevents the agent from repeatedly selecting landmarks in areas it has already explored.

        A landmark is excluded if ANY cell within its surrounding region (radius=exclusion_radius)
        has been visited by the agent (i.e., agent_trajectory_map value is not 0).

        Args:
            landmarks: Nx2 array of landmark positions (row, col)
            agent_trajectory_map: Binary map where 1=agent has visited, 0=not visited
            exclusion_radius: Radius (in pixels) around landmark to check for visited areas (default: 3)

        Returns:
            Filtered array of landmarks whose surrounding areas are completely unvisited
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros((0, 2), dtype=float)

        if agent_trajectory_map is None:
            return landmarks

        try:
            h, w = agent_trajectory_map.shape
            valid_landmarks = []

            for lm in landmarks:
                lm_r, lm_c = int(lm[0]), int(lm[1])

                # Define the region around this landmark
                r_min = max(0, lm_r - exclusion_radius)
                r_max = min(h, lm_r + exclusion_radius + 1)
                c_min = max(0, lm_c - exclusion_radius)
                c_max = min(w, lm_c + exclusion_radius + 1)

                # Extract the region around the landmark
                region = agent_trajectory_map[r_min:r_max, c_min:c_max]

                # Check if ANY cell in this region has been visited (value != 0)
                if np.sum(region) == 0:
                    # All cells in the region are 0 (unvisited), keep this landmark
                    valid_landmarks.append(lm)
                # else: landmark is near visited area, exclude it

            if len(valid_landmarks) == 0:
                return np.zeros((0, 2), dtype=float)

            return np.array(valid_landmarks, dtype=float)

        except Exception as e:
            # If any error occurs, return all landmarks
            print(f"Warning: Could not filter trajectory landmarks: {e}")
            return landmarks

    def _filter_distant_landmarks(self, landmarks, fbe_free_map, max_distance=20):
        """Filter out landmarks that are too far from the largest explored free area.

        Landmarks that require traversing a long distance through unexplored areas
        are less immediately reachable and should be deprioritized. This method only
        considers the largest connected free region to avoid noise from small fragments.

        Args:
            landmarks: Nx2 array of landmark positions (row, col)
            fbe_free_map: Tensor or array where >0.5 indicates explored free space
            max_distance: Maximum allowed distance (in pixels) from the largest free area (default: 20)

        Returns:
            Filtered array of landmarks within max_distance of the largest explored free area
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros((0, 2), dtype=float)

        if fbe_free_map is None:
            return landmarks

        try:
            from scipy.ndimage import distance_transform_edt, label

            # Convert fbe_free_map to numpy if it's a tensor
            if hasattr(fbe_free_map, 'cpu'):
                # PyTorch tensor: extract first batch and channel [0, 0, :, :]
                free_map = fbe_free_map.cpu().numpy()[0, 0]
            else:
                free_map = np.asarray(fbe_free_map)
                if len(free_map.shape) == 4:
                    free_map = free_map[0, 0]
                elif len(free_map.shape) == 3:
                    free_map = free_map[0]

            # Create binary mask of explored free space
            free_mask = (free_map > 0.5).astype(np.uint8)

            # Find all connected components in the free space
            labeled_map, num_features = label(free_mask)

            if num_features == 0:
                # No free space found, return empty
                return np.zeros((0, 2), dtype=float)

            # Find the largest connected component
            component_sizes = np.bincount(labeled_map.ravel())
            component_sizes[0] = 0  # Ignore background (label 0)
            largest_component_label = component_sizes.argmax()

            # Create mask containing only the largest free region
            largest_free_region = (labeled_map == largest_component_label).astype(np.uint8)

            # Compute distance transform: distance from each pixel to nearest point in largest free region
            # Invert the mask so largest_free_region = 0, other areas = 1
            inverted_mask = 1 - largest_free_region
            distance_map = distance_transform_edt(inverted_mask)

            h, w = distance_map.shape
            valid_landmarks = []

            for lm in landmarks:
                lm_r, lm_c = int(lm[0]), int(lm[1])

                # Boundary check
                if not (0 <= lm_r < h and 0 <= lm_c < w):
                    continue

                # Get distance to nearest point in the largest explored free region
                dist_to_largest_free = distance_map[lm_r, lm_c]

                # Keep landmark if it's within max_distance of the largest free area
                if dist_to_largest_free <= max_distance:
                    valid_landmarks.append(lm)

            if len(valid_landmarks) == 0:
                return np.zeros((0, 2), dtype=float)

            return np.array(valid_landmarks, dtype=float)

        except Exception as e:
            # If scipy is not available or any error occurs, return all landmarks
            print(f"Warning: Could not filter distant landmarks: {e}")
            return landmarks

    def get_nodes(self) -> Optional[np.ndarray]:
        if self.nodes is None or self.nodes.size == 0:
            return self.nodes
        # Convert float coordinates to integers by rounding
        return np.round(self.nodes).astype(int)

    def get_edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return self.edges

    def set_semantic_info(self, semantic_info_list: List[dict]):
        """Set semantic information for all landmarks.

        Args:
            semantic_info_list: List of dicts, one per landmark node.
                Each dict can contain keys like:
                - 'nearby_objects': list of nearby object info
                - 'nearby_rooms': list of nearby room info
                - 'nearby_edges': list of nearby relationships
                - 'object_distances': distances to each nearby object
        """
        self.semantic_info = semantic_info_list

    def get_semantic_info(self, node_idx: Optional[int] = None):
        """Get semantic information for a specific landmark or all landmarks.

        Args:
            node_idx: Index of the landmark node. If None, return all semantic info.

        Returns:
            Dict of semantic info for the specified node, or list of all semantic info.
        """
        if node_idx is None:
            return self.semantic_info
        if 0 <= node_idx < len(self.semantic_info):
            return self.semantic_info[node_idx]
        return {}

    def add_semantic_info_for_node(self, node_idx: int, semantic_data: dict):
        """Add or update semantic information for a specific landmark node.

        Args:
            node_idx: Index of the landmark node
            semantic_data: Dict containing semantic information
        """
        # Ensure the list is large enough
        while len(self.semantic_info) <= node_idx:
            self.semantic_info.append({})

        self.semantic_info[node_idx] = semantic_data

    # Topological Navigation Methods

    def find_nearest_landmark(self, position: np.ndarray) -> Optional[int]:
        """Find the nearest landmark to a given position.

        Args:
            position: [row, col] position in map coordinates

        Returns:
            Index of the nearest landmark, or None if no landmarks exist
        """
        if self.nodes is None or len(self.nodes) == 0:
            return None

        # Calculate Euclidean distance to all landmarks
        distances = np.linalg.norm(self.nodes - position, axis=1)
        nearest_idx = np.argmin(distances)

        return int(nearest_idx)

    def get_topological_path(self, start_pos: np.ndarray, goal_landmark_idx: int) -> Optional[List[int]]:
        """Compute topological path from start position to goal landmark.

        Args:
            start_pos: [row, col] current agent position
            goal_landmark_idx: Index of the goal landmark

        Returns:
            List of landmark indices forming the path, e.g., [3, 2, 7, 5]
            Returns None if no path exists or graph is not available
        """
        if self.graph is None or self.nodes is None or len(self.nodes) == 0:
            return None

        # Step 1: Find nearest landmark to start position
        start_landmark_idx = self.find_nearest_landmark(start_pos)
        if start_landmark_idx is None:
            return None

        # Step 2: Check if goal landmark index is valid
        if goal_landmark_idx < 0 or goal_landmark_idx >= len(self.nodes):
            return None

        # Step 3: Get corresponding graph node IDs
        if start_landmark_idx not in self.landmark_idx_to_node_id:
            return None
        if goal_landmark_idx not in self.landmark_idx_to_node_id:
            return None

        start_node_id = self.landmark_idx_to_node_id[start_landmark_idx]
        goal_node_id = self.landmark_idx_to_node_id[goal_landmark_idx]

        # Step 4: Compute shortest path in the graph
        try:
            node_path = nx.shortest_path(self.graph, source=start_node_id, target=goal_node_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path exists between start and goal
            return None

        # Step 5: Convert node IDs to landmark indices
        landmark_path = []
        for node_id in node_path:
            if node_id in self.node_id_to_landmark_idx:
                landmark_idx = self.node_id_to_landmark_idx[node_id]
                landmark_path.append(landmark_idx)

        return landmark_path if len(landmark_path) > 0 else None

    def get_path_length(self, landmark_path: List[int]) -> float:
        """Calculate the total Euclidean distance of a landmark path.

        Args:
            landmark_path: List of landmark indices

        Returns:
            Total path length in pixels
        """
        if self.nodes is None or len(landmark_path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(landmark_path) - 1):
            idx1 = landmark_path[i]
            idx2 = landmark_path[i + 1]

            if idx1 < len(self.nodes) and idx2 < len(self.nodes):
                pos1 = self.nodes[idx1]
                pos2 = self.nodes[idx2]
                total_length += np.linalg.norm(pos2 - pos1)

        return total_length

    def get_landmark_neighbors(self, landmark_idx: int) -> List[int]:
        """Get the indices of neighboring landmarks.

        Args:
            landmark_idx: Index of the landmark

        Returns:
            List of neighboring landmark indices
        """
        if self.graph is None or landmark_idx not in self.landmark_idx_to_node_id:
            return []

        node_id = self.landmark_idx_to_node_id[landmark_idx]
        neighbor_node_ids = list(self.graph.neighbors(node_id))

        neighbor_indices = []
        for neighbor_node_id in neighbor_node_ids:
            if neighbor_node_id in self.node_id_to_landmark_idx:
                neighbor_indices.append(self.node_id_to_landmark_idx[neighbor_node_id])

        return neighbor_indices

    def is_topological_navigation_available(self) -> bool:
        """Check if topological navigation is available.

        Returns:
            True if graph structure is available for topological navigation
        """
        return (self.graph is not None and
                len(self.node_id_to_landmark_idx) > 0 and
                self.nodes is not None and
                len(self.nodes) > 0)

    def _save_voronoi_visualization(self, traversible=None, obs_map=None,
                                     agent_position=None, frontier_points=None):
        """Save Voronoi landmark visualization automatically after each update.

        Uses occupancy map visualization style (matching visualizations/visualizer.py):
        - Unknown areas: White (#FFFFFF)
        - Free space: Light gray (#E7E7E7)
        - Obstacles: Medium gray (#A2A2A2)

        Saves two visualizations:
        1. Map without landmarks (base occupancy map only)
        2. Map with landmarks (base map + landmarks + edges)

        Args:
            traversible: Binary traversible map (1=free, 0=obstacle)
            obs_map: Obstacle map
            agent_position: Current agent position [row, col]
            frontier_points: Frontier points to overlay (Nx2 array)
        """
        # Disabled: No longer saving voronoi visualizations
        return

    def visualize(self, traversible=None, obs_map=None, agent_position=None,
                  frontier_points=None, selected_landmark_idx=None,
                  path=None, save_path=None, show=True, crop_to_valid_area=True):
        """Visualize the landmark map with optional overlays.

        Args:
            traversible: Binary traversible map (1=free, 0=obstacle)
            obs_map: Obstacle map
            agent_position: Current agent position [row, col]
            frontier_points: Frontier points to overlay (Nx2 array)
            selected_landmark_idx: Index of selected landmark to highlight
            path: List of landmark indices representing a path
            save_path: Path to save the visualization (if None, not saved)
            show: Whether to display the plot (default: True)
            crop_to_valid_area: If True, crop visualization to actual valid area (default: True)

        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Calculate crop bounds if requested
        crop_bounds = None
        if crop_to_valid_area and traversible is not None:
            # Convert to numpy if needed
            if hasattr(traversible, 'cpu'):
                trav_np = traversible.cpu().numpy()
            else:
                trav_np = traversible

            # Find the bounding box of the valid (traversible) area
            rows, cols = np.where(trav_np > 0.5)
            if len(rows) > 0 and len(cols) > 0:
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()

                # Add padding for better visualization
                padding = 20
                r_min = max(0, r_min - padding)
                r_max = min(trav_np.shape[0] - 1, r_max + padding)
                c_min = max(0, c_min - padding)
                c_max = min(trav_np.shape[1] - 1, c_max + padding)

                crop_bounds = (r_min, r_max, c_min, c_max)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Draw base map
        if traversible is not None:
            # Convert to numpy if needed
            if hasattr(traversible, 'cpu'):
                traversible = traversible.cpu().numpy()

            # Create a 3-channel visualization:
            # - Dark gray (0.2): Outside traversible region (non-navigable external area)
            # - White (1.0): Inside traversible region (navigable free space)
            # - Black (0.0): Obstacles inside traversible region
            base_map = np.ones((*traversible.shape, 3)) * 0.2  # Dark gray for external area

            # Mark traversible interior as white
            traversible_mask = traversible > 0.5
            base_map[traversible_mask] = [1.0, 1.0, 1.0]  # White for navigable area

            # Overlay internal obstacles in black if provided
            if obs_map is not None:
                if hasattr(obs_map, 'cpu'):
                    obs_map = obs_map.cpu().numpy()

                # Only mark internal obstacles (within traversible region)
                # obs_map might contain both internal obstacles and external area
                # We want to show internal obstacles as black
                internal_obstacles = (obs_map > 0.5) & traversible_mask
                base_map[internal_obstacles] = [0.0, 0.0, 0.0]  # Black for obstacles

            # Apply cropping if bounds are available
            if crop_bounds is not None:
                r_min, r_max, c_min, c_max = crop_bounds
                base_map = base_map[r_min:r_max+1, c_min:c_max+1]
                # Display with correct extent
                ax.imshow(base_map, origin='upper', extent=[c_min, c_max+1, r_max+1, r_min])
            else:
                ax.imshow(base_map, origin='upper')
        else:
            # Just show white background
            if self.nodes is not None and len(self.nodes) > 0:
                max_r = int(np.max(self.nodes[:, 0])) + 50
                max_c = int(np.max(self.nodes[:, 1])) + 50
                ax.set_xlim(0, max_c)
                ax.set_ylim(max_r, 0)
            ax.set_aspect('equal')

        # Draw edges (if available)
        if self.edges is not None and len(self.edges) > 0:
            for edge in self.edges:
                (r1, c1), (r2, c2) = edge
                ax.plot([c1, c2], [r1, r2], 'b-', alpha=0.4, linewidth=1.5, label='Edge' if edge == self.edges[0] else '')

        # Draw frontier points (if provided, for comparison)
        if frontier_points is not None and len(frontier_points) > 0:
            if hasattr(frontier_points, 'cpu'):
                frontier_points = frontier_points.cpu().numpy()
            ax.scatter(frontier_points[:, 1], frontier_points[:, 0],
                      c='lightgreen', s=10, alpha=0.5, marker='x', label='Frontier Points')

        # Draw landmarks
        if self.nodes is not None and len(self.nodes) > 0:
            # Color landmarks by type if semantic info is available
            if self.semantic_info and len(self.semantic_info) == len(self.nodes):
                leaf_nodes = []
                intersection_nodes = []
                other_nodes = []

                for i, meta in enumerate(self.semantic_info):
                    if meta.get('type') == 'leaf':
                        leaf_nodes.append(self.nodes[i])
                    elif meta.get('type') == 'intersection':
                        intersection_nodes.append(self.nodes[i])
                    else:
                        other_nodes.append(self.nodes[i])

                if len(leaf_nodes) > 0:
                    leaf_nodes = np.array(leaf_nodes)
                    ax.scatter(leaf_nodes[:, 1], leaf_nodes[:, 0],
                             c='cyan', s=100, marker='o', edgecolors='blue', linewidths=2,
                             label=f'Leaf Landmarks ({len(leaf_nodes)})', zorder=5)

                if len(intersection_nodes) > 0:
                    intersection_nodes = np.array(intersection_nodes)
                    ax.scatter(intersection_nodes[:, 1], intersection_nodes[:, 0],
                             c='yellow', s=150, marker='D', edgecolors='orange', linewidths=2,
                             label=f'Intersection Landmarks ({len(intersection_nodes)})', zorder=5)

                if len(other_nodes) > 0:
                    other_nodes = np.array(other_nodes)
                    ax.scatter(other_nodes[:, 1], other_nodes[:, 0],
                             c='red', s=100, marker='s', edgecolors='darkred', linewidths=2,
                             label=f'Other Landmarks ({len(other_nodes)})', zorder=5)
            else:
                # Simple visualization without types
                ax.scatter(self.nodes[:, 1], self.nodes[:, 0],
                         c='red', s=100, marker='o', edgecolors='darkred', linewidths=2,
                         label=f'Landmarks ({len(self.nodes)})', zorder=5)

            # Add landmark indices as text
            for i, node in enumerate(self.nodes):
                ax.text(node[1], node[0], str(i), fontsize=8,
                       ha='center', va='center', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                       zorder=6)

        # Highlight selected landmark
        if selected_landmark_idx is not None and self.nodes is not None:
            if 0 <= selected_landmark_idx < len(self.nodes):
                selected = self.nodes[selected_landmark_idx]
                circle = patches.Circle((selected[1], selected[0]), 15,
                                       linewidth=3, edgecolor='lime', facecolor='none',
                                       label='Selected Landmark', zorder=7)
                ax.add_patch(circle)

        # Draw path if provided
        if path is not None and self.nodes is not None and len(path) > 1:
            path_coords = [self.nodes[idx] for idx in path if 0 <= idx < len(self.nodes)]
            if len(path_coords) > 1:
                path_coords = np.array(path_coords)
                ax.plot(path_coords[:, 1], path_coords[:, 0],
                       'g-', linewidth=3, alpha=0.8, label='Topological Path', zorder=4)
                # Mark waypoints
                ax.scatter(path_coords[:, 1], path_coords[:, 0],
                         c='lime', s=150, marker='*', edgecolors='green', linewidths=2,
                         label='Waypoints', zorder=5)

        # Draw agent position
        if agent_position is not None:
            if isinstance(agent_position, (list, tuple)) and len(agent_position) == 2:
                ax.scatter(agent_position[1], agent_position[0],
                         c='magenta', s=200, marker='^', edgecolors='purple', linewidths=3,
                         label='Agent Position', zorder=8)

        # Add title and legend
        title = f"Landmark Map Visualization ({self.method} method)\n"
        if self.nodes is not None:
            title += f"Total Landmarks: {len(self.nodes)}\n"
        title += "Map: Dark Gray=External | White=Navigable | Black=Obstacles"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('Column (x)', fontsize=12)
        ax.set_ylabel('Row (y)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Save if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        # Show if requested
        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def _save_step_visualization(self, step_name, map_data, points=None, edges=None,
                                  vor_vertices=None, vor_ridges=None, graph=None,
                                  agent_position=None, title_suffix="", node_id_to_landmark_idx=None):
        """Save visualization of intermediate step during Voronoi landmark generation.

        Args:
            step_name: Name of the step (e.g., "step1_free_space", "step5_voronoi")
            map_data: 2D numpy array to visualize as background
            points: Optional Nx2 array of points to overlay (e.g., boundary points, landmarks)
            edges: Optional list of edge tuples to draw
            vor_vertices: Optional Voronoi vertices
            vor_ridges: Optional Voronoi ridge vertices
            graph: Optional NetworkX graph to visualize
            agent_position: Optional agent position [row, col]
            title_suffix: Additional text for title
        """
        # # Disabled: No longer saving step visualizations
        # return
        import matplotlib.patches as patches
        import os

        # Create output directory
        output_dir = "voronoi_steps"
        os.makedirs(output_dir, exist_ok=True)

        # Crop to valid area (non-zero region) for better visualization
        crop_bounds = None
        if map_data is not None:
            rows, cols = np.where(map_data > 0)
            if len(rows) > 0 and len(cols) > 0:
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()

                # Add padding for context
                padding = 20
                r_min = max(0, r_min - padding)
                r_max = min(map_data.shape[0] - 1, r_max + padding)
                c_min = max(0, c_min - padding)
                c_max = min(map_data.shape[1] - 1, c_max + padding)

                crop_bounds = (r_min, r_max, c_min, c_max)

        # Create larger figure for better visibility (20x20 inches)
        fig, ax = plt.subplots(figsize=(20, 20))

        # Display map data
        if map_data is not None:
            ax.imshow(map_data, cmap='gray', origin='upper', interpolation='nearest')

            # Crop to valid area
            if crop_bounds is not None:
                r_min, r_max, c_min, c_max = crop_bounds
                ax.set_xlim(c_min, c_max)
                ax.set_ylim(r_max, r_min)  # Inverted for image coordinates

        # Draw Voronoi diagram if provided
        if vor_vertices is not None and vor_ridges is not None:
            # Draw Voronoi edges (thicker for visibility)
            for ridge in vor_ridges:
                if ridge[0] >= 0 and ridge[1] >= 0:  # Ignore infinite edges
                    v1 = vor_vertices[ridge[0]]
                    v2 = vor_vertices[ridge[1]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]],
                           'b-', linewidth=3, alpha=0.6, label='Voronoi Edges' if ridge == vor_ridges[0] else "")

            # Draw Voronoi vertices (larger for visibility)
            ax.scatter(vor_vertices[:, 0], vor_vertices[:, 1],
                      c='blue', s=120, marker='o', alpha=0.8, label='Voronoi Vertices', zorder=5, edgecolors='darkblue', linewidths=2)

        # Draw NetworkX graph if provided
        if graph is not None:
            pos = nx.get_node_attributes(graph, 'pos')

            # Draw edges (thicker for visibility)
            for edge in graph.edges():
                node1_pos = pos[edge[0]]
                node2_pos = pos[edge[1]]
                ax.plot([node1_pos[0], node2_pos[0]], [node1_pos[1], node2_pos[1]],
                       'g-', linewidth=4, alpha=0.7, zorder=3)

            # Draw nodes with different colors based on degree (larger for visibility)
            for node_id in graph.nodes():
                if 'pos' in graph.nodes[node_id]:
                    node_pos = graph.nodes[node_id]['pos']
                    degree = graph.degree(node_id)

                    if degree == 1:
                        color = 'yellow'
                        marker = 's'
                        size = 300
                        label = 'Leaf Node' if node_id == list(graph.nodes())[0] else ""
                    elif degree >= 3:
                        color = 'red'
                        marker = 'D'
                        size = 350
                        label = 'Intersection Node' if node_id == list(graph.nodes())[0] else ""
                    else:
                        color = 'orange'
                        marker = 'o'
                        size = 250
                        label = 'Degree-2 Node' if node_id == list(graph.nodes())[0] else ""

                    ax.scatter(node_pos[0], node_pos[1],
                             c=color, s=size, marker=marker,
                             edgecolors='black', linewidths=3,
                             label=label, zorder=6)

                    # Draw landmark index if mapping is provided
                    if node_id_to_landmark_idx is not None and node_id in node_id_to_landmark_idx:
                        landmark_idx = node_id_to_landmark_idx[node_id]
                        ax.text(node_pos[0], node_pos[1], f'L{landmark_idx}',
                               fontsize=12, fontweight='bold', color='white',
                               ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'),
                               zorder=7)

        # Draw custom edges if provided
        elif edges is not None:
            for edge in edges:
                p1, p2 = edge
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                       'g-', linewidth=4, alpha=0.7, label='Edges' if edge == edges[0] else "")

        # Draw points if provided (larger for visibility)
        if points is not None and len(points) > 0:
            # Determine point type based on step name
            if 'boundary' in step_name.lower() or 'boundaries' in step_name.lower():
                # Boundaries: (col, row) format - scatter(col, row) same as Voronoi vertices
                ax.scatter(points[:, 0], points[:, 1],
                          c='cyan', s=20, marker='.', alpha=0.8, label='Boundary Points', zorder=4)
            elif 'landmark' in step_name or 'final' in step_name:
                # Landmarks: (row, col) format - swap for display
                ax.scatter(points[:, 1], points[:, 0],
                          c='red', s=400, marker='*',
                          edgecolors='yellow', linewidths=3,
                          label='Landmarks', zorder=7)
            else:
                # Other points: (row, col) format - swap for display
                ax.scatter(points[:, 1], points[:, 0],
                          c='red', s=200, marker='o', alpha=0.8,
                          edgecolors='darkred', linewidths=2,
                          label='Points', zorder=5)

        # Draw agent position (larger for visibility)
        if agent_position is not None:
            ax.scatter(agent_position[1], agent_position[0],
                      c='magenta', s=500, marker='^',
                      edgecolors='purple', linewidths=4,
                      label='Agent Position', zorder=8)

        # Set title and labels (larger fonts for readability)
        title = f"{step_name.replace('_', ' ').title()}"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=16, framealpha=0.9)
        ax.set_xlabel('Column (x)', fontsize=18)
        ax.set_ylabel('Row (y)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)

        # Save figure with high DPI for clarity
        save_path = os.path.join(output_dir, f"{self.step_counter:02d}_{step_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  💾 Saved step visualization: {save_path}")
        self.step_counter += 1
