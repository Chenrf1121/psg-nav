"""
Information Gain Scorer: Simplified version for landmark evaluation

This module computes Information Gain utility scores for landmarks to encourage
exploration of unexplored areas and avoid wandering in known regions.

Simplified IG formula:
    U_IG(landmark_k) = w1 × Novelty + w2 × FrontierDensity - w3 × VisitPenalty

Components:
- Novelty: How unexplored is the area around this landmark?
- FrontierDensity: How many unknown regions (frontiers) are nearby?
- VisitPenalty: Has this landmark been visited before? (Binary: 0 or 1)
"""

import numpy as np


class SimplifiedIGScorer:
    """
    Compute simplified Information Gain scores for landmarks.

    This scorer encourages the robot to:
    1. Visit unexplored areas (high novelty)
    2. Move toward regions with many frontiers (high exploration potential)
    3. Avoid revisiting landmarks (visit penalty)
    """

    def __init__(self, visit_tracker, map_resolution=5.0,
                 w_novelty=0.4, w_frontier_density=0.4, w_visit_penalty=0.2):
        """
        Initialize the IG scorer.

        Args:
            visit_tracker (LandmarkVisitTracker): Tracker for visited landmarks
            map_resolution (float): Map resolution in cm/pixel (default: 5.0)
            w_novelty (float): Weight for novelty component (default: 0.4)
            w_frontier_density (float): Weight for frontier density (default: 0.4)
            w_visit_penalty (float): Weight for visit penalty (default: 0.2)
        """
        self.visit_tracker = visit_tracker
        self.map_resolution = map_resolution

        # Weights (should sum to 1.0)
        self.w_novelty = w_novelty
        self.w_frontier_density = w_frontier_density
        self.w_visit_penalty = w_visit_penalty

        # Normalize weights
        total_weight = w_novelty + w_frontier_density + w_visit_penalty
        if total_weight > 0:
            self.w_novelty /= total_weight
            self.w_frontier_density /= total_weight
            self.w_visit_penalty /= total_weight

        # Frontier density parameters
        self.frontier_search_radius = 100  # pixels (~5 meters with 5cm resolution)
        self.max_frontier_count = 50  # Normalization constant

    def compute_ig_score(self, landmark_position, landmark_idx,
                        frontier_locations=None, fbe_free_map=None):
        """
        Compute Information Gain score for a single landmark.

        Args:
            landmark_position: [row, col] position of the landmark
            landmark_idx (int): Index of the landmark
            frontier_locations: Nx2 array of frontier points (optional)
            fbe_free_map: Free space map (optional, not used in simplified version)

        Returns:
            float: IG score in range [0, 1]
        """
        # 1. Novelty score (0-1)
        novelty = self.visit_tracker.get_novelty_score(landmark_position)

        # 2. Frontier density score (0-1)
        frontier_density = self._compute_frontier_density(
            landmark_position, frontier_locations
        )

        # 3. Visit penalty (0-1, binary)
        visit_penalty = self.visit_tracker.get_visit_penalty(landmark_idx)

        # Combined IG score
        ig_score = (self.w_novelty * novelty +
                   self.w_frontier_density * frontier_density -
                   self.w_visit_penalty * visit_penalty)

        # Ensure non-negative
        ig_score = max(0.0, ig_score)

        return ig_score

    def _compute_frontier_density(self, landmark_pos, frontier_locations):
        """
        Compute frontier density around a landmark.

        Args:
            landmark_pos: [row, col] position
            frontier_locations: Nx2 array of frontier positions

        Returns:
            float: Normalized frontier density (0-1)
        """
        if frontier_locations is None or len(frontier_locations) == 0:
            return 0.0

        # Ensure numpy array
        if not isinstance(frontier_locations, np.ndarray):
            frontier_locations = np.array(frontier_locations)

        # Ensure 2D array
        if frontier_locations.ndim == 1:
            frontier_locations = frontier_locations.reshape(-1, 2)

        # Compute distances to all frontiers
        landmark_pos = np.array(landmark_pos).reshape(1, 2)
        distances = np.linalg.norm(frontier_locations - landmark_pos, axis=1)

        # Count frontiers within search radius
        nearby_count = np.sum(distances <= self.frontier_search_radius)

        # Normalize by max expected count
        density = min(1.0, nearby_count / self.max_frontier_count)

        return density

    def compute_batch_scores(self, landmark_positions, landmark_indices,
                            frontier_locations=None, fbe_free_map=None):
        """
        Compute IG scores for multiple landmarks in batch.

        Args:
            landmark_positions: Nx2 array of landmark positions
            landmark_indices: N-length array of landmark indices
            frontier_locations: Mx2 array of frontier points
            fbe_free_map: Free space map (optional)

        Returns:
            np.ndarray: Array of IG scores (length N)
        """
        if len(landmark_positions) == 0:
            return np.array([])

        scores = []
        for lm_pos, lm_idx in zip(landmark_positions, landmark_indices):
            score = self.compute_ig_score(
                lm_pos, lm_idx, frontier_locations, fbe_free_map
            )
            scores.append(score)

        return np.array(scores)

    def get_score_breakdown(self, landmark_position, landmark_idx,
                           frontier_locations=None, fbe_free_map=None):
        """
        Get detailed breakdown of IG score components for debugging.

        Args:
            landmark_position: [row, col] position
            landmark_idx (int): Landmark index
            frontier_locations: Frontier points
            fbe_free_map: Free space map

        Returns:
            dict: Breakdown with keys: 'novelty', 'frontier_density',
                  'visit_penalty', 'total_score'
        """
        novelty = self.visit_tracker.get_novelty_score(landmark_position)
        frontier_density = self._compute_frontier_density(
            landmark_position, frontier_locations
        )
        visit_penalty = self.visit_tracker.get_visit_penalty(landmark_idx)

        total_score = (self.w_novelty * novelty +
                      self.w_frontier_density * frontier_density -
                      self.w_visit_penalty * visit_penalty)

        return {
            'novelty': novelty,
            'frontier_density': frontier_density,
            'visit_penalty': visit_penalty,
            'novelty_weighted': self.w_novelty * novelty,
            'frontier_weighted': self.w_frontier_density * frontier_density,
            'penalty_weighted': self.w_visit_penalty * visit_penalty,
            'total_score': max(0.0, total_score),
        }

    def set_weights(self, w_novelty, w_frontier_density, w_visit_penalty):
        """
        Update component weights.

        Args:
            w_novelty: Weight for novelty
            w_frontier_density: Weight for frontier density
            w_visit_penalty: Weight for visit penalty
        """
        self.w_novelty = w_novelty
        self.w_frontier_density = w_frontier_density
        self.w_visit_penalty = w_visit_penalty

        # Normalize
        total = w_novelty + w_frontier_density + w_visit_penalty
        if total > 0:
            self.w_novelty /= total
            self.w_frontier_density /= total
            self.w_visit_penalty /= total

    def __repr__(self):
        return (f"SimplifiedIGScorer("
                f"w_novelty={self.w_novelty:.2f}, "
                f"w_frontier={self.w_frontier_density:.2f}, "
                f"w_penalty={self.w_visit_penalty:.2f})")


class AdaptiveLambdaScheduler:
    """
    Compute adaptive λ (lambda) for balancing Goal Utility and IG Utility.

    λ controls the trade-off:
        Score = λ × U_goal + (1-λ) × U_IG

    Strategies:
    - Time-based: Start with more exploration (low λ), end with more exploitation (high λ)
    - Confidence-based: If goal confidence is high, increase λ (focus on goal)
    - Hybrid: Combine both strategies
    """

    def __init__(self, strategy='time_based', lambda_min=0.3, lambda_max=0.8):
        """
        Initialize the scheduler.

        Args:
            strategy (str): Strategy to use ('time_based', 'confidence_based', 'hybrid')
            lambda_min (float): Minimum lambda value (more exploration)
            lambda_max (float): Maximum lambda value (more exploitation)
        """
        self.strategy = strategy
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def get_lambda(self, current_step, max_steps=500, goal_confidence=None):
        """
        Compute adaptive lambda value.

        Args:
            current_step (int): Current navigation step
            max_steps (int): Maximum steps per episode (default: 500)
            goal_confidence (float): Optional confidence in goal location (0-1)

        Returns:
            float: Lambda value in [lambda_min, lambda_max]
        """
        if self.strategy == 'time_based':
            return self._time_based_lambda(current_step, max_steps)
        elif self.strategy == 'confidence_based':
            return self._confidence_based_lambda(goal_confidence)
        elif self.strategy == 'hybrid':
            return self._hybrid_lambda(current_step, max_steps, goal_confidence)
        else:
            # Default: fixed lambda at midpoint
            return (self.lambda_min + self.lambda_max) / 2

    def _time_based_lambda(self, current_step, max_steps):
        """
        Time-based adaptive lambda: increase linearly with steps.

        Early steps: low λ (more exploration)
        Late steps: high λ (more exploitation)
        """
        progress = min(1.0, current_step / max_steps)
        lambda_val = self.lambda_min + (self.lambda_max - self.lambda_min) * progress
        return lambda_val

    def _confidence_based_lambda(self, goal_confidence):
        """
        Confidence-based adaptive lambda.

        High confidence in goal → high λ (focus on reaching goal)
        Low confidence → low λ (keep exploring)
        """
        if goal_confidence is None:
            return (self.lambda_min + self.lambda_max) / 2

        # Linear mapping from confidence to lambda
        lambda_val = self.lambda_min + (self.lambda_max - self.lambda_min) * goal_confidence
        return lambda_val

    def _hybrid_lambda(self, current_step, max_steps, goal_confidence):
        """
        Hybrid strategy: combine time-based and confidence-based.

        Uses weighted average of both strategies.
        """
        time_lambda = self._time_based_lambda(current_step, max_steps)

        if goal_confidence is not None:
            conf_lambda = self._confidence_based_lambda(goal_confidence)
            # Weight: 60% time-based, 40% confidence-based
            lambda_val = 0.6 * time_lambda + 0.4 * conf_lambda
        else:
            lambda_val = time_lambda

        return lambda_val

    def __repr__(self):
        return (f"AdaptiveLambdaScheduler("
                f"strategy={self.strategy}, "
                f"range=[{self.lambda_min}, {self.lambda_max}])")
