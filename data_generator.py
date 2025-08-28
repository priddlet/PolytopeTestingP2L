"""
Unified Polytope Data Generator

This module generates labeled datasets for polytope classification tasks with both
traditional sampling methods and direct balanced generation using ground truth knowledge.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, List
from polytope_generator import PolytopeGenerator


class PolytopeDataGenerator:
    """Unified generator for polytope classification datasets with balance control"""
    
    def __init__(self, dimension: int = 10, random_seed: Optional[int] = None):
        """
        Initialize the data generator
        
        Args:
            dimension: Dimension of the space
            random_seed: Random seed for reproducibility
        """
        self.dimension = dimension
        if random_seed is not None:
            np.random.seed(random_seed)
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
    
    def generate_dataset(self, 
                        polytope_type: str,
                        n_samples: int,
                        sampling_strategy: str = 'uniform',
                        label_type: str = 'binary',
                        target_balance: Optional[float] = None,
                        random_seed: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate a dataset for polytope classification
        
        Args:
            polytope_type: Type of polytope to generate
            n_samples: Number of samples to generate
            sampling_strategy: Strategy for sampling points ('uniform', 'adaptive', 'biased', 'boundary_focused', 'direct_balanced')
            label_type: Type of labels ('binary' or 'signed_distance')
            target_balance: Target ratio of inside points (0.05 to 0.5) for direct_balanced strategy
            random_seed: Random seed for this generation
            
        Returns:
            Tuple of (data, labels, metadata)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create polytope
        polytope_gen = PolytopeGenerator(self.dimension, random_seed=random_seed)
        
        # Handle custom halfspace count for complexity study
        if polytope_type == 'random_halfspaces' and metadata and 'halfspace_count' in metadata:
            A, b, polytope_metadata = polytope_gen.create_random_halfspaces(metadata['halfspace_count'])
        else:
            A, b, polytope_metadata = polytope_gen.create_random_polytope(polytope_type)
        
        # Use direct balanced generation if specified
        if sampling_strategy == 'direct_balanced':
            if target_balance is None:
                raise ValueError("target_balance must be specified for direct_balanced strategy")
            return self._direct_balanced_generation(A, b, polytope_type, target_balance, n_samples, label_type, polytope_metadata)
        
        # Sample points based on strategy
        if sampling_strategy == 'uniform':
            data = self._uniform_sampling(A, b, n_samples)
        elif sampling_strategy == 'adaptive':
            data = self._adaptive_sampling(A, b, n_samples)
        elif sampling_strategy == 'biased':
            data = self._biased_sampling(A, b, n_samples)
        elif sampling_strategy == 'boundary_focused':
            data = self._boundary_focused_sampling(A, b, n_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Generate labels
        if label_type == 'binary':
            labels = self._generate_binary_labels(A, b, data)
        elif label_type == 'signed_distance':
            labels = self._generate_signed_distance_labels(A, b, data)
        else:
            raise ValueError(f"Unknown label type: {label_type}")
        
        # Calculate balance
        balance = np.mean(labels)
        
        metadata = {
            'polytope_type': polytope_type,
            'sampling_strategy': sampling_strategy,
            'label_type': label_type,
            'balance': balance,
            'n_samples': n_samples,
            'polytope_metadata': polytope_metadata
        }
        
        return data, labels, metadata
    
    def _direct_balanced_generation(self, A: np.ndarray, b: np.ndarray, polytope_type: str,
                                  target_balance: float, n_samples: int, label_type: str,
                                  polytope_metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate dataset with exact balance ratio using ground truth knowledge"""
        
        # Calculate exact numbers needed
        n_inside = int(n_samples * target_balance)
        n_outside = n_samples - n_inside
        
        print(f"  Target: {n_inside} inside, {n_outside} outside")
        
        # Generate inside points directly
        inside_points = self._generate_inside_points(A, b, n_inside, polytope_type)
        
        # Generate outside points directly
        outside_points = self._generate_outside_points(A, b, n_outside, polytope_type)
        
        # Combine and shuffle
        data = np.vstack([inside_points, outside_points])
        
        if label_type == 'binary':
            labels = np.concatenate([np.ones(n_inside), np.zeros(n_outside)])
        else:  # signed_distance
            inside_labels = np.array([self._signed_distance_to_polytope(point, A, b) for point in inside_points])
            outside_labels = np.array([self._signed_distance_to_polytope(point, A, b) for point in outside_points])
            labels = np.concatenate([inside_labels, outside_labels])
        
        # Shuffle the data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        # Verify actual balance
        actual_balance = np.mean(labels > 0) if label_type == 'binary' else np.mean(labels < 0)
        
        metadata = {
            'polytope_type': polytope_type,
            'target_balance': target_balance,
            'actual_balance': actual_balance,
            'n_samples': n_samples,
            'n_inside': n_inside,
            'n_outside': n_outside,
            'polytope_metadata': polytope_metadata,
            'sampling_strategy': 'direct_balanced',
            'label_type': label_type
        }
        
        return data, labels, metadata
    
    def _generate_inside_points(self, A: np.ndarray, b: np.ndarray, 
                               n_inside: int, polytope_type: str) -> np.ndarray:
        """Generate points that are guaranteed to be inside the polytope"""
        
        if n_inside == 0:
            return np.array([]).reshape(0, self.dimension)
        
        inside_points = []
        max_attempts = n_inside * 100  # Reasonable limit
        attempts = 0
        
        while len(inside_points) < n_inside and attempts < max_attempts:
            attempts += 1
            
            if polytope_type == 'hypercube':
                # For hypercube, sample from interior
                point = self.rng.uniform(-0.8, 0.8, self.dimension)
            elif polytope_type == 'simplex':
                # For simplex, sample from interior using barycentric coordinates
                point = self._sample_simplex_interior()
            else:
                # For other polytopes, use rejection sampling with tighter bounds
                bounds = self._estimate_bounds(A, b)
                point = self.rng.uniform(bounds[:, 0] * 0.8, bounds[:, 1] * 0.8, self.dimension)
            
            # Check if point is inside
            if self._point_in_polytope(point, A, b):
                inside_points.append(point)
        
        if len(inside_points) < n_inside:
            print(f"Warning: Only generated {len(inside_points)} inside points out of {n_inside} requested")
        
        return np.array(inside_points)
    
    def _generate_outside_points(self, A: np.ndarray, b: np.ndarray, 
                                n_outside: int, polytope_type: str) -> np.ndarray:
        """Generate points that are guaranteed to be outside the polytope"""
        
        if n_outside == 0:
            return np.array([]).reshape(0, self.dimension)
        
        outside_points = []
        max_attempts = n_outside * 100
        attempts = 0
        
        while len(outside_points) < n_outside and attempts < max_attempts:
            attempts += 1
            
            if polytope_type == 'hypercube':
                # For hypercube, sample from outside the unit cube
                point = self.rng.uniform(-2.0, 2.0, self.dimension)
                # Ensure it's outside by pushing it away from center
                if np.all(np.abs(point) <= 1.0):
                    # Push in a random direction
                    direction = self.rng.randn(self.dimension)
                    direction = direction / np.linalg.norm(direction)
                    point = point + direction * 1.5
            else:
                # For other polytopes, use expanded bounds
                bounds = self._estimate_bounds(A, b)
                point = self.rng.uniform(bounds[:, 0] * 1.5, bounds[:, 1] * 1.5, self.dimension)
            
            # Check if point is outside
            if not self._point_in_polytope(point, A, b):
                outside_points.append(point)
        
        if len(outside_points) < n_outside:
            print(f"Warning: Only generated {len(outside_points)} outside points out of {n_outside} requested")
        
        return np.array(outside_points)
    
    def _sample_simplex_interior(self) -> np.ndarray:
        """Sample a point from the interior of a simplex"""
        # Generate barycentric coordinates
        coords = self.rng.exponential(1.0, self.dimension + 1)
        coords = coords / np.sum(coords)
        
        # Convert to Cartesian coordinates
        vertices = []
        for i in range(self.dimension + 1):
            vertex = np.zeros(self.dimension)
            if i < self.dimension:
                vertex[i] = 1.0
            vertices.append(vertex)
        
        point = np.zeros(self.dimension)
        for i, coord in enumerate(coords):
            point += coord * vertices[i]
        
        return point
    
    def _uniform_sampling(self, A: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
        """Uniform sampling from a bounded region"""
        bounds = self._estimate_bounds(A, b)
        return self.rng.uniform(bounds[:, 0], bounds[:, 1], (n_samples, self.dimension))
    
    def _adaptive_sampling(self, A: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
        """Adaptive sampling that focuses on regions near the polytope boundary"""
        bounds = self._estimate_bounds(A, b)
        
        # Sample more points near the boundary
        data = []
        for _ in range(n_samples):
            if self.rng.random() < 0.7:  # 70% near boundary
                # Sample near boundary
                point = self.rng.uniform(bounds[:, 0] * 0.8, bounds[:, 1] * 0.8, self.dimension)
            else:  # 30% far from boundary
                # Sample from expanded region
                point = self.rng.uniform(bounds[:, 0] * 1.5, bounds[:, 1] * 1.5, self.dimension)
            data.append(point)
        
        return np.array(data)
    
    def _biased_sampling(self, A: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
        """Biased sampling that tries to achieve better balance"""
        bounds = self._estimate_bounds(A, b)
        
        data = []
        inside_count = 0
        outside_count = 0
        target_inside = n_samples // 2
        
        for _ in range(n_samples):
            if inside_count < target_inside and self.rng.random() < 0.6:
                # Try to sample inside
                point = self.rng.uniform(bounds[:, 0] * 0.5, bounds[:, 1] * 0.5, self.dimension)
            else:
                # Sample outside
                point = self.rng.uniform(bounds[:, 0] * 1.2, bounds[:, 1] * 1.2, self.dimension)
            
            data.append(point)
            
            # Update counts
            if self._point_in_polytope(point, A, b):
                inside_count += 1
            else:
                outside_count += 1
        
        return np.array(data)
    
    def _boundary_focused_sampling(self, A: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
        """Sampling focused on the boundary region"""
        bounds = self._estimate_bounds(A, b)
        
        data = []
        for _ in range(n_samples):
            # Sample near the boundary
            point = self.rng.uniform(bounds[:, 0] * 0.9, bounds[:, 1] * 0.9, self.dimension)
            data.append(point)
        
        return np.array(data)
    
    def _generate_binary_labels(self, A: np.ndarray, b: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Generate binary labels (1 for inside, 0 for outside)"""
        return np.array([1.0 if self._point_in_polytope(point, A, b) else 0.0 for point in data])
    
    def _generate_signed_distance_labels(self, A: np.ndarray, b: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Generate signed distance labels"""
        return np.array([self._signed_distance_to_polytope(point, A, b) for point in data])
    
    def _point_in_polytope(self, point: np.ndarray, A: np.ndarray, b: np.ndarray) -> bool:
        """Check if a point is inside the polytope"""
        return np.all(A @ point <= b)
    
    def _signed_distance_to_polytope(self, point: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
        """Calculate signed distance to polytope boundary"""
        # Distance to each face
        distances = (b - A @ point) / np.linalg.norm(A, axis=1)
        
        # If point is inside, return negative of minimum distance to boundary
        if np.all(A @ point <= b):
            return -np.min(distances)
        else:
            # If point is outside, return positive distance to boundary
            return np.min(distances)
    
    def _estimate_bounds(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Estimate bounds for sampling based on polytope definition"""
        bounds = np.zeros((self.dimension, 2))
        
        # Simple heuristic: use the constraints to estimate bounds
        for i in range(self.dimension):
            # Find constraints involving this dimension
            relevant_constraints = A[:, i] != 0
            
            if np.any(relevant_constraints):
                # Use constraints to estimate bounds
                A_relevant = A[relevant_constraints, i]
                b_relevant = b[relevant_constraints]
                
                # For positive coefficients: x <= b/a
                pos_mask = A_relevant > 0
                if np.any(pos_mask):
                    upper_bounds = b_relevant[pos_mask] / A_relevant[pos_mask]
                    bounds[i, 1] = np.min(upper_bounds)
                else:
                    bounds[i, 1] = 2.0  # Default upper bound
                
                # For negative coefficients: x >= b/a
                neg_mask = A_relevant < 0
                if np.any(neg_mask):
                    lower_bounds = b_relevant[neg_mask] / A_relevant[neg_mask]
                    bounds[i, 0] = np.max(lower_bounds)
                else:
                    bounds[i, 0] = -2.0  # Default lower bound
            else:
                # No constraints on this dimension, use default bounds
                bounds[i, 0] = -2.0
                bounds[i, 1] = 2.0
        
        return bounds
    
    def generate_polytope_data(self, A: np.ndarray, b: np.ndarray, n_samples: int, 
                              balance_ratio: float = 0.3, key: Optional[jax.Array] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from existing polytope constraints
        
        Args:
            A: Constraint matrix
            b: Constraint vector
            n_samples: Number of samples to generate
            balance_ratio: Target ratio of safe points
            key: JAX random key
            
        Returns:
            Tuple of (data, targets)
        """
        if key is None:
            key = jax.random.key(42)
        
        # Use direct balanced generation
        data, targets, _ = self._direct_balanced_generation(
            A, b, "custom", balance_ratio, n_samples, "binary", {}
        )
        
        return data, targets
    
    def split_data(self, data: np.ndarray, labels: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, random_seed: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/validation/test sets
        
        Args:
            data: Input features
            labels: Target labels
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return {
            'train': (data[train_indices], labels[train_indices]),
            'val': (data[val_indices], labels[val_indices]),
            'test': (data[test_indices], labels[test_indices])
        } 