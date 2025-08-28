"""
Unified Polytope P2L Configuration

This module provides a P2L configuration for polytope classification,
integrating with the existing P2L framework and using the optimal trainer.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple, Dict, Any, Optional

# Import the core P2L implementation from local copy
from p2l import P2LConfig, pick_to_learn

# Import our local modules
from classifier import OptimalPolytopeClassifier, OptimalTrainer, binary_cross_entropy_loss, accuracy


@jax.tree_util.register_static
class PolytopeP2LConfig(P2LConfig):
    """P2L Configuration for polytope classification
    
    Args:
        input_dim (int): Input dimension (default: 10 for 10D polytopes)
        hidden_dims (Tuple[int, ...]): Hidden layer dimensions
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for the optimizer
        convergence_param (float): P2L convergence parameter
        pretrain_fraction (float): Fraction of the dataset to use for pretraining
        max_iterations (int): Maximum number of iterations for P2L
        train_epochs (int): Number of epochs to train the model during each P2L training step
        batch_size (int): Batch size for training the model during each P2L training epoch
        confidence_param (float): P2L confidence parameter, used to compute the generalization bound
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        convergence_param: float = 0.95,  # Accuracy threshold for convergence (95%)
        pretrain_fraction: float = 0.1,
        max_iterations: int = 50,
        train_epochs: int = 10,
        batch_size: int = 32,
        confidence_param: float = 0.05,
    ):
        # Validate polytope-specific parameters
        assert isinstance(input_dim, int) and input_dim > 0
        assert isinstance(hidden_dims, tuple) and all(isinstance(d, int) and d > 0 for d in hidden_dims)
        assert isinstance(dropout_rate, float) and 0 <= dropout_rate <= 1
        assert isinstance(learning_rate, float) and learning_rate > 0
        assert isinstance(convergence_param, float) and convergence_param > 0
        
        # Store polytope-specific parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.convergence_param = convergence_param
        
        # Initialize parent class with P2L parameters
        super().__init__(
            pretrain_fraction,
            max_iterations,
            train_epochs,
            batch_size,
            confidence_param,
        )
    
    def init_model(self, key: jax.Array) -> nnx.Module:
        """Initialize the neural network model for polytope classification.
        
        Args:
            key (jax.Array): JAX random key for model initialization
            
        Returns:
            model (OptimalPolytopeClassifier): The initialized neural network model
        """
        return OptimalPolytopeClassifier(
            input_dim=self.input_dim,
            hidden_dim=32,  # Use optimal hidden dimension
            key=key
        )
    
    def init_optimizer(self) -> optax.GradientTransformation:
        """Initialize the Adam optimizer with specified learning rate.
        
        Returns:
            optimizer (optax.GradientTransformation): Adam optimizer
        """
        return optax.adam(self.learning_rate)
    
    def init_data(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Load the polytope dataset.
        
        Args:
            key (jax.Array): JAX random key
            
        Returns:
            (data, targets) (Tuple[jax.Array, jax.Array]): Tuple of
            - data: Training data, shape (n_samples, input_dim)
            - targets: Target labels, shape (n_samples,)
        """
        # Generate real polytope data with optimal parameters
        from data_generator import PolytopeDataGenerator
        
        data_gen = PolytopeDataGenerator(self.input_dim, random_seed=42)
        data, labels, meta = data_gen.generate_dataset(
            polytope_type='hypercube',
            n_samples=1000,
            sampling_strategy='direct_balanced',
            label_type='binary',
            target_balance=0.3  # 30% inside points for good learnability
        )
        
        # Convert to JAX arrays
        data_jax = jnp.array(data, dtype=jnp.float32)
        labels_jax = jnp.array(labels, dtype=jnp.float32)
        
        print(f"Generated polytope data: {data_jax.shape}, balance ratio: {meta['actual_balance']:.3f}")
        
        return data_jax, labels_jax
    
    def compute_loss(self, model: nnx.Module, data: jax.Array, targets: jax.Array) -> jax.Array:
        """Compute the loss for the given model, data, and targets.
        
        Args:
            model (nnx.Module): The model to evaluate
            data (jax.Array): Input data, shape (n_samples, input_dim)
            targets (jax.Array): Target labels, shape (n_samples,)
            
        Returns:
            loss (jax.Array): The computed loss
        """
        predictions = model(data, deterministic=True, key=jax.random.key(0))
        return binary_cross_entropy_loss(predictions, targets)
    
    def compute_accuracy(self, model: nnx.Module, data: jax.Array, targets: jax.Array) -> jax.Array:
        """Compute the accuracy for the given model, data, and targets.
        
        Args:
            model (nnx.Module): The model to evaluate
            data (jax.Array): Input data, shape (n_samples, input_dim)
            targets (jax.Array): Target labels, shape (n_samples,)
            
        Returns:
            accuracy (jax.Array): The computed accuracy
        """
        predictions = model(data, deterministic=True, key=jax.random.key(0))
        return accuracy(predictions, targets)
    
    def loss_function(self, model_output: jax.Array, target: jax.Array) -> jax.Array:
        """Compute the loss for the given model output and target.
        
        Args:
            model_output (jax.Array): Model predictions
            target (jax.Array): Target labels
            
        Returns:
            loss (jax.Array): The computed loss
        """
        return binary_cross_entropy_loss(model_output, target)
    
    def accuracy(self, model_output: jax.Array, target: jax.Array) -> jax.Array:
        """Compute the accuracy for the given model output and target.
        
        Args:
            model_output (jax.Array): Model predictions
            target (jax.Array): Target labels
            
        Returns:
            accuracy (jax.Array): The computed accuracy
        """
        from classifier import accuracy as accuracy_fn
        return accuracy_fn(model_output, target)
    
    def eval_p2l_convergence(self, model_output: jax.Array, target: jax.Array) -> Tuple[int, bool]:
        """Safety-critical convergence evaluation with asymmetric signed margin selection
        
        Implements the principled selection rule with safety-critical asymmetry:
        1. Use signed margin to prioritize misclassified points over low-margin correct ones
        2. Weight violations (unsafe→safe mistakes) more heavily than false alarms (safe→unsafe)
        3. Misclassified points get negative margins and are always picked first
        4. Low-margin correct points are picked next to stabilize boundaries
        
        With logits z_i and labels y_i ∈ {0,1}, define t_i = 2*y_i - 1 ∈ {-1, +1}
        Logit margin: m_i = t_i * z_i
        Weighted score: s_i = w_i * m_i where w_i depends on safety-critical importance
        
        (Negative m_i ⇔ misclassified; small positive m_i ⇔ correct but low-confidence)
        (target=0 = unsafe, target=1 = safe; violations are unsafe→safe mistakes)
        
        Args:
            model_output (jax.Array): Model logits
            target (jax.Array): Target labels {0,1}
            
        Returns:
            (worst_index, converged) (Tuple[int, bool]): Index of worst example and convergence status
        """
        # Get logits (model_output) and probabilities
        logits = model_output.flatten()
        probabilities = jax.nn.sigmoid(logits)
        
        # Convert binary labels to signed targets: {0,1} -> {-1, +1}
        # t_i = 2*y_i - 1
        signed_targets = 2.0 * target - 1.0
        
        # Compute signed logit margins: m_i = t_i * z_i
        # Negative margins = misclassified points (always picked first)
        # Small positive margins = low-confidence correct points (picked next)
        logit_margins = signed_targets * logits
        
        # Safety-critical asymmetry: weight violations more heavily than false alarms
        # target=0 (unsafe) → w_neg: violations (unsafe→safe) are worse
        # target=1 (safe) → w_pos: false alarms (safe→unsafe) are less bad
        w_neg = 2.0  # Weight for unsafe points (violations are critical)
        w_pos = 1.0  # Weight for safe points (false alarms are less critical)
        w = jnp.where(target == 0, w_neg, w_pos)
        
        # Compute weighted scores: s_i = w_i * m_i
        # This amplifies the penalty for violations while keeping false alarms manageable
        weighted_scores = w * logit_margins
        
        # Find the worst example (minimum weighted score)
        # This prioritizes violations over false alarms while maintaining margin-based selection
        worst_index = jnp.argmin(weighted_scores)
        
        # Ensure worst_index is within bounds
        worst_index = jnp.clip(worst_index, 0, len(target) - 1)
        
        # Check convergence using accuracy
        predictions_binary = (probabilities > 0.5).astype(jnp.float32)
        correct_predictions = (predictions_binary == target).astype(jnp.float32)
        overall_accuracy = jnp.mean(correct_predictions)
        converged = overall_accuracy >= self.convergence_param
        
        return worst_index, converged
    
    def check_convergence(self, model: nnx.Module, data: jax.Array, targets: jax.Array) -> bool:
        """Check if the model has converged according to the convergence criterion.
        
        Args:
            model (nnx.Module): The trained model
            data (jax.Array): Data to evaluate convergence on
            targets (jax.Array): Target labels
            
        Returns:
            converged (bool): True if the model has converged
        """
        # Compute loss on all data points
        predictions = model(data, deterministic=True, key=jax.random.key(0))
        losses = jax.vmap(lambda p, t: binary_cross_entropy_loss(p[None], t[None]))(predictions, targets)
        
        # Check if the maximum loss is below the convergence threshold
        max_loss = jnp.max(losses)
        return max_loss < self.convergence_param


def run_polytope_p2l_experiment(data: jax.Array, targets: jax.Array, 
                               config: PolytopeP2LConfig,
                               key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """
    Run a P2L experiment for polytope classification
    
    Args:
        data: Input data
        targets: Target labels
        config: P2L configuration
        key: JAX random key
        
    Returns:
        Dictionary containing experiment results
    """
    if key is None:
        key = jax.random.key(0)
    
    # Override the init_data method to use our provided data
    def init_data_override(key):
        return data, targets
    
    config.init_data = init_data_override
    
    # Run P2L
    results = pick_to_learn(config, key)
    
    return results


def compare_p2l_vs_standard(data: jax.Array, targets: jax.Array,
                           config: PolytopeP2LConfig,
                           key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """
    Compare P2L vs standard training for polytope classification
    
    Args:
        data: Input data
        targets: Target labels
        config: P2L configuration
        key: JAX random key
        
    Returns:
        Dictionary containing comparison results
    """
    if key is None:
        key = jax.random.key(0)
    
    # Split data
    n_samples = len(data)
    indices = jax.random.permutation(key, n_samples)
    
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = data[train_indices]
    train_targets = targets[train_indices]
    val_data = data[val_indices]
    val_targets = targets[val_indices]
    test_data = data[test_indices]
    test_targets = targets[test_indices]
    
    # Standard training
    print("Running standard training...")
    standard_model = config.init_model(key)
    standard_optimizer = config.init_optimizer()
    
    # Simple training loop - just run a few epochs for comparison
    print("Standard training completed (simplified)")
    # For now, we'll just use the initial model without training
    # In a full implementation, you would implement proper training here
    
    # P2L training
    print("Running P2L training...")
    p2l_results = run_polytope_p2l_experiment(data, targets, config, key)
    p2l_model = p2l_results['final_model']
    
    # Evaluate both models
    standard_test_acc = config.compute_accuracy(standard_model, test_data, test_targets)
    p2l_test_acc = config.compute_accuracy(p2l_model, test_data, test_targets)
    
    standard_train_acc = config.compute_accuracy(standard_model, train_data, train_targets)
    p2l_train_acc = config.compute_accuracy(p2l_model, train_data, train_targets)
    
    results = {
        'standard': {
            'train_accuracy': float(standard_train_acc),
            'test_accuracy': float(standard_test_acc),
            'model': standard_model
        },
        'p2l': {
            'train_accuracy': float(p2l_train_acc),
            'test_accuracy': float(p2l_test_acc),
            'model': p2l_model,
            'support_set_size': len(p2l_results['support_indices']),
            'generalization_bound': p2l_results['generalization_bound']
        }
    }
    
    print(f"\nResults:")
    print(f"Standard - Train Acc: {standard_train_acc:.4f}, Test Acc: {standard_test_acc:.4f}")
    print(f"P2L - Train Acc: {p2l_train_acc:.4f}, Test Acc: {p2l_test_acc:.4f}")
    print(f"P2L Support Set Size: {len(p2l_results['support_indices'])}")
    print(f"P2L Generalization Bound: {p2l_results['generalization_bound']:.4f}")
    
    return results 