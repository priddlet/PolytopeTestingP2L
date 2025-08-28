"""
Geometric Fidelity Metrics for Polytope Classification

This module provides metrics to evaluate how well a learned classifier
approximates the ground truth polytope geometry.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def estimate_optimal_sample_size(dimension: int, base_samples: int = 10000) -> int:
    """
    Estimate optimal sample size based on dimension to handle curse of dimensionality
    
    Args:
        dimension: Number of dimensions
        base_samples: Base number of samples for 2D
        
    Returns:
        Recommended sample size
    """
    # Exponential scaling: double samples for each dimension increase
    # This helps maintain statistical power as volume ratios decrease
    scaling_factor = 2 ** max(0, dimension - 2)
    return int(base_samples * scaling_factor)


def estimate_polytope_bounds(A: np.ndarray, b: np.ndarray, padding_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate tight bounding box around polytope to improve sampling efficiency
    
    Args:
        A: Constraint matrix
        b: Constraint vector
        padding_factor: Factor to add padding around estimated bounds
        
    Returns:
        Tuple of (min_coords, max_coords) defining tight bounding box
    """
    dim = A.shape[1]
    
    # Initialize bounds
    min_coords = np.full(dim, -np.inf)
    max_coords = np.full(dim, np.inf)
    
    # For each dimension, find constraints that bound it
    for i in range(dim):
        # Find constraints with non-zero coefficient for this dimension
        relevant_constraints = A[:, i] != 0
        
        if np.any(relevant_constraints):
            A_relevant = A[relevant_constraints, i]
            b_relevant = b[relevant_constraints]
            
            # For positive coefficients: x_i <= b/a
            pos_mask = A_relevant > 0
            if np.any(pos_mask):
                upper_bounds = b_relevant[pos_mask] / A_relevant[pos_mask]
                max_coords[i] = np.min(upper_bounds)
            
            # For negative coefficients: x_i >= b/a
            neg_mask = A_relevant < 0
            if np.any(neg_mask):
                lower_bounds = b_relevant[neg_mask] / A_relevant[neg_mask]
                min_coords[i] = np.max(lower_bounds)
    
    # Handle unbounded dimensions
    min_coords = np.where(np.isfinite(min_coords), min_coords, -2.0)
    max_coords = np.where(np.isfinite(max_coords), max_coords, 2.0)
    
    # Add padding
    ranges = max_coords - min_coords
    padding = ranges * padding_factor
    
    min_coords -= padding
    max_coords += padding
    
    return min_coords, max_coords


def monte_carlo_iou(predicted_safe_region, true_polytope, bounding_box, n_samples=10000, key=None, 
                   auto_scale_samples=True, auto_bounds=True):
    """
    Compute Monte Carlo IoU between predicted safe region and true polytope
    
    Args:
        predicted_safe_region: Function that takes points and returns safety predictions
        true_polytope: Function that takes points and returns true safety labels
        bounding_box: Tuple of (min_coords, max_coords) defining sampling region
        n_samples: Number of Monte Carlo samples
        key: JAX random key
        auto_scale_samples: Whether to automatically scale sample size based on dimension
        auto_bounds: Whether to automatically estimate tight bounding box
        
    Returns:
        Dictionary with IoU and related metrics
    """
    if key is None:
        key = jax.random.key(42)
    
    min_coords, max_coords = bounding_box
    dim = len(min_coords)
    
    # Auto-scale sample size if requested
    if auto_scale_samples:
        n_samples = estimate_optimal_sample_size(dim, n_samples)
        print(f"  Auto-scaled to {n_samples:,} samples for {dim}D")
    
    # Auto-estimate bounds if requested
    if auto_bounds and hasattr(true_polytope, 'A') and hasattr(true_polytope, 'b'):
        min_coords, max_coords = estimate_polytope_bounds(true_polytope.A, true_polytope.b)
        print(f"  Auto-estimated bounds: {min_coords} to {max_coords}")
    
    # Generate random samples in bounding box
    key, sample_key = jax.random.split(key)
    samples = jax.random.uniform(sample_key, (n_samples, dim), 
                                minval=min_coords, maxval=max_coords)
    
    # Get predictions from both models
    pred_safe = predicted_safe_region(samples)
    true_safe = true_polytope(samples)
    
    # Convert to boolean masks
    pred_mask = pred_safe > 0.5
    true_mask = true_safe > 0.5
    
    # Compute intersection and union
    intersection = jnp.logical_and(pred_mask, true_mask)
    union = jnp.logical_or(pred_mask, true_mask)
    
    # Compute IoU
    intersection_size = jnp.sum(intersection)
    union_size = jnp.sum(union)
    
    iou = jnp.where(union_size > 0, intersection_size / union_size, 0.0)
    
    # Compute additional metrics
    pred_safe_size = jnp.sum(pred_mask)
    true_safe_size = jnp.sum(true_mask)
    
    # False safe volume (unsafe but predicted safe)
    false_safe_volume = jnp.sum(jnp.logical_and(pred_mask, ~true_mask))
    
    # False unsafe volume (safe but predicted unsafe)
    false_unsafe_volume = jnp.sum(jnp.logical_and(~pred_mask, true_mask))
    
    # Volume ratios
    pred_volume_ratio = pred_safe_size / n_samples
    true_volume_ratio = true_safe_size / n_samples
    
    return {
        'iou': float(iou),
        'intersection_size': int(intersection_size),
        'union_size': int(union_size),
        'pred_safe_size': int(pred_safe_size),
        'true_safe_size': int(true_safe_size),
        'false_safe_volume': int(false_safe_volume),
        'false_unsafe_volume': int(false_unsafe_volume),
        'pred_volume_ratio': float(pred_volume_ratio),
        'true_volume_ratio': float(true_volume_ratio),
        'false_safe_ratio': float(false_safe_volume / n_samples),
        'false_unsafe_ratio': float(false_unsafe_volume / n_samples),
        'n_samples_used': n_samples,
        'bounding_box_used': (min_coords, max_coords)
    }


def create_polytope_classifier(model, scaler=None):
    """
    Create a function that classifies points as safe/unsafe using a trained model
    
    Args:
        model: Trained classifier model
        scaler: Optional scaler for input normalization
        
    Returns:
        Function that takes points and returns safety predictions
    """
    def classify_points(points):
        if scaler is not None:
            points_scaled = scaler.transform(points)
        else:
            points_scaled = points
            
        if hasattr(model, 'predict_proba'):  # sklearn model
            probabilities = model.predict_proba(points_scaled)[:, 1]
        else:  # JAX model
            logits = model(points_scaled, deterministic=True)
            probabilities = jax.nn.sigmoid(logits).flatten()
            
        return probabilities
    
    return classify_points


def create_true_polytope_classifier(A, b):
    """
    Create a function that classifies points using the true polytope constraints
    
    Args:
        A: Constraint matrix (n_constraints x n_dimensions)
        b: Constraint vector (n_constraints,)
        
    Returns:
        Function that takes points and returns true safety labels
    """
    def classify_points(points):
        # Check if points satisfy all constraints: Ax <= b
        constraints = jnp.dot(points, A.T)  # Shape: (n_points, n_constraints)
        satisfied = jnp.all(constraints <= b, axis=1)  # Shape: (n_points,)
        return satisfied.astype(jnp.float32)
    
    # Add A and b as attributes for auto-bounds estimation
    classify_points.A = A
    classify_points.b = b
    
    return classify_points


def compute_geometric_metrics(model, A, b, test_data, test_labels, scaler=None, 
                            bounding_box=None, n_samples=10000, key=None,
                            auto_scale_samples=True, auto_bounds=True):
    """
    Compute comprehensive geometric fidelity metrics
    
    Args:
        model: Trained classifier model
        A: True polytope constraint matrix
        b: True polytope constraint vector
        test_data: Test data points
        test_labels: Test labels
        scaler: Optional scaler for input normalization
        bounding_box: Optional bounding box for Monte Carlo sampling
        n_samples: Number of Monte Carlo samples
        key: JAX random key
        auto_scale_samples: Whether to automatically scale sample size based on dimension
        auto_bounds: Whether to automatically estimate tight bounding box
        
    Returns:
        Dictionary with all geometric metrics
    """
    if key is None:
        key = jax.random.key(42)
    
    # Create classifier functions
    pred_classifier = create_polytope_classifier(model, scaler)
    true_classifier = create_true_polytope_classifier(A, b)
    
    # Compute test set metrics
    test_predictions = pred_classifier(test_data)
    test_pred_binary = (test_predictions > 0.5).astype(jnp.float32)
    
    # Standard classification metrics
    test_accuracy = jnp.mean(test_pred_binary == test_labels)
    
    # Geometric metrics on test set
    test_true_safe = true_classifier(test_data)
    test_geometric_accuracy = jnp.mean(test_pred_binary == test_true_safe)
    
    # Monte Carlo metrics if bounding box provided or auto-bounds enabled
    mc_metrics = {}
    if bounding_box is not None or auto_bounds:
        if bounding_box is None and auto_bounds:
            # Auto-estimate bounding box
            min_coords, max_coords = estimate_polytope_bounds(A, b)
            bounding_box = (min_coords, max_coords)
            print(f"Auto-estimated bounding box for {A.shape[1]}D polytope")
        
        # Geometric metrics for all models
        mc_metrics = monte_carlo_iou(
            pred_classifier, true_classifier, bounding_box, n_samples, key,
            auto_scale_samples, auto_bounds
        )
    
    return {
        'test_accuracy': float(test_accuracy),
        'test_geometric_accuracy': float(test_geometric_accuracy),
        'test_predictions': np.array(test_predictions),
        'test_pred_binary': np.array(test_pred_binary),
        'monte_carlo_metrics': mc_metrics
    }


def visualize_geometric_comparison(model, A, b, test_data, test_labels, scaler=None,
                                 bounding_box=None, n_samples=1000, key=None):
    """
    Create visualization comparing predicted and true safe regions
    
    Args:
        model: Trained classifier model
        A: True polytope constraint matrix
        b: True polytope constraint vector
        test_data: Test data points
        test_labels: Test labels
        scaler: Optional scaler for input normalization
        bounding_box: Optional bounding box for visualization
        n_samples: Number of samples for visualization
        key: JAX random key
        
    Returns:
        matplotlib figure
    """
    if key is None:
        key = jax.random.key(42)
    
    # For 2D visualization
    if test_data.shape[1] != 2:
        print("Warning: Geometric visualization only supports 2D data")
        return None
    
    # Create classifier functions
    pred_classifier = create_polytope_classifier(model, scaler)
    true_classifier = create_true_polytope_classifier(A, b)
    
    # Generate grid for visualization
    if bounding_box is None:
        # Use test data bounds
        min_coords = np.min(test_data, axis=0)
        max_coords = np.max(test_data, axis=0)
        # Add some padding
        padding = (max_coords - min_coords) * 0.1
        min_coords -= padding
        max_coords += padding
    else:
        min_coords, max_coords = bounding_box
    
    # Create meshgrid
    x = np.linspace(min_coords[0], max_coords[0], int(np.sqrt(n_samples)))
    y = np.linspace(min_coords[1], max_coords[1], int(np.sqrt(n_samples)))
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Get predictions
    pred_probs = pred_classifier(grid_points)
    true_safe = true_classifier(grid_points)
    
    # Reshape for plotting
    pred_probs_grid = pred_probs.reshape(X.shape)
    true_safe_grid = true_safe.reshape(X.shape)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Predicted safe region
    im1 = ax1.contourf(X, Y, pred_probs_grid, levels=20, cmap='RdYlBu')
    ax1.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, 
               cmap='RdYlBu', edgecolors='black', s=20, alpha=0.7)
    ax1.set_title('Predicted Safe Region')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: True safe region
    im2 = ax2.contourf(X, Y, true_safe_grid, levels=2, cmap='RdYlBu')
    ax2.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, 
               cmap='RdYlBu', edgecolors='black', s=20, alpha=0.7)
    ax2.set_title('True Safe Region')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Difference
    diff = pred_probs_grid - true_safe_grid
    im3 = ax3.contourf(X, Y, diff, levels=20, cmap='RdBu')
    ax3.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, 
               cmap='RdYlBu', edgecolors='black', s=20, alpha=0.7)
    ax3.set_title('Difference (Pred - True)')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    return fig

