"""
Polytope Generator for 10-dimensional testing

This module creates various types of polytopes in 10-dimensional space
for testing the pick-to-learn algorithm with known ground truth.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


class PolytopeGenerator:
    """Generator for various types of polytopes in 10-dimensional space"""
    
    def __init__(self, dimension: int = 10, random_seed: Optional[int] = None):
        """
        Initialize the polytope generator
        
        Args:
            dimension: Dimension of the space (default: 10)
            random_seed: Random seed for reproducibility
        """
        self.dimension = dimension
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
    
    def create_random_polytope(self, polytope_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create a random polytope of the specified type or randomly select one
        
        Args:
            polytope_type: Type of polytope to create. If None, randomly selects one.
                          Options: 'hypercube', 'simplex', 'random_halfspaces', 'ellipsoid_approximation'
            
        Returns:
            Tuple of (A, b, metadata) where Ax <= b defines the polytope
        """
        if polytope_type is None:
            polytope_types = ['hypercube', 'simplex', 'random_halfspaces', 'ellipsoid_approximation']
            polytope_type = random.choice(polytope_types)
        
        if polytope_type == 'hypercube':
            return self.create_hypercube()
        elif polytope_type == 'simplex':
            return self.create_simplex()
        elif polytope_type == 'random_halfspaces':
            return self.create_random_halfspaces()
        elif polytope_type == 'ellipsoid_approximation':
            return self.create_ellipsoid_approximation()
        else:
            raise ValueError(f"Unknown polytope type: {polytope_type}")
    
    def create_hypercube(self, center: Optional[np.ndarray] = None, 
                        side_length: float = 2.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create a hypercube polytope
        
        Args:
            center: Center of the hypercube (default: origin)
            side_length: Length of each side
            
        Returns:
            Tuple of (A, b, metadata) where Ax <= b defines the polytope
        """
        if center is None:
            center = np.zeros(self.dimension)
        
        # Create faces for each dimension
        A = []
        b = []
        
        for i in range(self.dimension):
            # Lower face: x_i >= center_i - side_length/2
            face_lower = np.zeros(self.dimension)
            face_lower[i] = -1
            A.append(face_lower)
            b.append(-center[i] + side_length/2)
            
            # Upper face: x_i <= center_i + side_length/2
            face_upper = np.zeros(self.dimension)
            face_upper[i] = 1
            A.append(face_upper)
            b.append(center[i] + side_length/2)
        
        metadata = {
            'type': 'hypercube',
            'center': center,
            'side_length': side_length,
            'volume': side_length ** self.dimension
        }
        
        return np.array(A), np.array(b), metadata
    
    def create_simplex(self, center: Optional[np.ndarray] = None, 
                      radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create a regular simplex polytope
        
        Args:
            center: Center of the simplex (default: origin)
            radius: Radius of the simplex
            
        Returns:
            Tuple of (A, b, metadata) where Ax <= b defines the polytope
        """
        if center is None:
            center = np.zeros(self.dimension)
        
        # Create a regular simplex
        # For a d-dimensional simplex, we need d+1 vertices
        vertices = []
        
        # First vertex at (radius, 0, 0, ..., 0)
        v1 = np.zeros(self.dimension)
        v1[0] = radius
        vertices.append(v1)
        
        # Remaining vertices
        for i in range(1, self.dimension + 1):
            v = np.zeros(self.dimension)
            v[0] = -radius / self.dimension
            if i < self.dimension:
                v[i] = radius * np.sqrt(1 - 1/(self.dimension**2))
            vertices.append(v)
        
        # Convert vertices to half-space representation
        A, b = self._vertices_to_halfspaces(vertices)
        
        # Translate to center
        b = b - A @ center
        
        metadata = {
            'type': 'simplex',
            'center': center,
            'radius': radius,
            'vertices': vertices
        }
        
        return A, b, metadata
    
    def create_random_halfspaces(self, n_halfspaces: int = 20) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create a polytope defined by random halfspaces
        
        Args:
            n_halfspaces: Number of halfspaces to generate
            
        Returns:
            Tuple of (A, b, metadata) where Ax <= b defines the polytope
        """
        A = self.rng.randn(n_halfspaces, self.dimension)
        # Normalize the normal vectors
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
            
        # Generate random offsets
        b = self.rng.uniform(0.5, 2.0, n_halfspaces)
        
        metadata = {
            'type': 'random_halfspaces',
            'n_halfspaces': n_halfspaces
        }
        
        return A, b, metadata
    
    def create_ellipsoid_approximation(self, center: Optional[np.ndarray] = None,
                                     radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create a polytope that approximates an ellipsoid using many halfspaces
        
        Args:
            center: Center of the ellipsoid (default: origin)
            radius: Radius of the ellipsoid
            
        Returns:
            Tuple of (A, b, metadata) where Ax <= b defines the polytope
        """
        if center is None:
            center = np.zeros(self.dimension)
        
        # Generate many random directions to approximate the ellipsoid
        n_halfspaces = 50
        A = self.rng.randn(n_halfspaces, self.dimension)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
            
        # Set the offset to be the radius
        b = radius * np.ones(n_halfspaces)
        
        # Translate to center
        b = b - A @ center
        
        metadata = {
            'type': 'ellipsoid_approximation',
            'center': center,
            'radius': radius,
            'n_halfspaces': n_halfspaces
        }
        
        return A, b, metadata
    
    def _vertices_to_halfspaces(self, vertices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a list of vertices to half-space representation
        
        Args:
            vertices: List of vertex coordinates
            
        Returns:
            Tuple of (A, b) where Ax <= b defines the convex hull
        """
        # This is a simplified version - in practice, you'd use a proper convex hull algorithm
        # For now, we'll create a simple bounding box around the vertices
        vertices = np.array(vertices)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        A = []
        b = []
        
        for i in range(self.dimension):
            # Lower bound
            face_lower = np.zeros(self.dimension)
            face_lower[i] = -1
            A.append(face_lower)
            b.append(-min_coords[i])
            
            # Upper bound
            face_upper = np.zeros(self.dimension)
            face_upper[i] = 1
            A.append(face_upper)
            b.append(max_coords[i])
        
        return np.array(A), np.array(b)
    
    def point_in_polytope(self, point: np.ndarray, A: np.ndarray, b: np.ndarray) -> bool:
        """
        Check if a point is inside the polytope
        
        Args:
            point: Point to check
            A, b: Polytope definition Ax <= b
            
        Returns:
            True if point is inside the polytope
        """
        return np.all(A @ point <= b)
    
    def signed_distance_to_polytope(self, point: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the signed distance from a point to the polytope
        
        Args:
            point: Point to compute distance from
            A, b: Polytope definition Ax <= b
            
        Returns:
            Signed distance (negative if inside, positive if outside)
        """
        # Compute distance to each face
        distances = (A @ point - b) / np.linalg.norm(A, axis=1)
        
        # If all distances are negative, point is inside
        if np.all(distances <= 0):
            # Distance to boundary is the maximum (least negative) distance
            return np.max(distances)
        else:
            # Point is outside, distance is the minimum positive distance
            return np.min(distances[distances > 0]) 