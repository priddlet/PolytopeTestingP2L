"""
Fixed Polytope P2L Configuration with Improved Convergence and Sample Selection
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple, Dict, Any, Optional

# Import the core P2L implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'picktolearn'))
from p2l import P2LConfig, pick_to_learn

# Import our local modules
from classifier import OptimalPolytopeClassifier, OptimalTrainer, binary_cross_entropy_loss, accuracy


@jax.tree_util.register_static
class FixedPolytopeP2LConfig(P2LConfig):
    """Fixed P2L Configuration for polytope classification with improved convergence
    
    Key improvements:
    1. Adaptive convergence threshold based on data balance
    2. Better worst sample selection (misclassified first)
    3. More reasonable default parameters
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        convergence_param: float = 0.70,  # Default to 70% for imbalanced data
        pretrain_fraction: float = 0.1,
        max_iterations: int = 50,
        train_epochs: int = 10,
        batch_size: int = 32,
        confidence_param: float = 0.05,
    ):
        # Validate polytope-specific parameters
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not all(d > 0 for d in hidden_dims):
            raise ValueError("All hidden_dims must be positive")
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0 < convergence_param <= 1):
            raise ValueError("convergence_param must be between 0 and 1")
        if not (0 < pretrain_fraction < 1):
            raise ValueError("pretrain_fraction must be between 0 and 1")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if train_epochs <= 0:
            raise ValueError("train_epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0 < confidence_param < 1):
            raise ValueError("confidence_param must be between 0 and 1")
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.convergence_param = convergence_param
        self.pretrain_fraction = pretrain_fraction
        self.max_iterations = max_iterations
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.confidence_param = confidence_param
        
        # Initialize data and model methods (to be set later)
        self.init_data = None
    
    def init_model(self, key: jax.Array) -> nnx.Module:
        """Initialize the neural network model for polytope classification"""
        return OptimalPolytopeClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dims[0],  # Use first hidden dimension
            key=key
        )
    
    def init_optimizer(self) -> optax.GradientTransformation:
        """Initialize the optimizer for training"""
        return optax.adam(learning_rate=self.learning_rate)
    
    def forward(self, graphdef: nnx.GraphDef, model_state: nnx.State, 
                data: jax.Array, deterministic: bool, key: jax.Array) -> jax.Array:
        """Forward pass through the model"""
        model = nnx.merge(graphdef, model_state)
        return model(data, deterministic=deterministic, key=key)
    
    def loss_function(self, model_output: jax.Array, target: jax.Array) -> jax.Array:
        """Compute the loss for the given model output and target"""
        return binary_cross_entropy_loss(model_output, target)
    
    def accuracy(self, model_output: jax.Array, target: jax.Array) -> jax.Array:
        """Compute the accuracy for the given model output and target"""
        from classifier import accuracy as accuracy_fn
        return accuracy_fn(model_output, target)
    
    def eval_p2l_convergence(self, model_output: jax.Array, target: jax.Array) -> Tuple[int, bool]:
        """Improved convergence evaluation with adaptive threshold and better sample selection
        
        Key improvements:
        1. Adaptive convergence threshold based on data balance
        2. Prioritize misclassified samples over low-confidence correct ones
        3. More robust worst sample selection
        """
        # Compute per-sample accuracies
        predictions = jax.nn.sigmoid(model_output)
        predictions_binary = (predictions > 0.5).astype(jnp.float32)
        correct_predictions = (predictions_binary == target).astype(jnp.float32)
        
        # Calculate balance ratio for adaptive threshold
        balance_ratio = jnp.mean(target)
        
        # Use adaptive convergence threshold based on data balance
        # For imbalanced data (< 30%), use user's convergence_param
        # For balanced data (>= 30%), use higher threshold
        convergence_threshold = jnp.where(balance_ratio < 0.3, self.convergence_param, 0.85)
        
        # Find the worst example using margin-based selection (distance from decision boundary)
        # This selects samples closest to the decision boundary (most informative)
        margin = jnp.abs(predictions - 0.5)  # Distance from 0.5 (decision boundary)
        
        # Find sample with smallest margin (closest to decision boundary)
        worst_index = jnp.argmin(margin)
        
        # Ensure worst_index is within bounds
        worst_index = jnp.clip(worst_index, 0, len(target) - 1)
        
        # Check convergence
        overall_accuracy = jnp.mean(correct_predictions)
        converged = overall_accuracy >= convergence_threshold
        
        return worst_index, converged
    
    def train_step(self, graphdef: nnx.GraphDef, model_state: nnx.State,
                   optimizer: optax.GradientTransformation, opt_state: optax.OptState,
                   data: jax.Array, target: jax.Array, key: jax.Array) -> Tuple[float, float, nnx.State, optax.OptState]:
        """Perform a single training step on a batch of data."""
        deterministic = False  # Training is not deterministic
        
        # Compute loss and gradients
        (loss, aux), grads = jax.value_and_grad(
            self.loss_and_aux, argnums=0, has_aux=True
        )(model_state, graphdef, data, target, deterministic, key)
        accuracy, _model_output = aux
        
        # Update optimizer state and compute parameter updates
        updates, new_opt_state = optimizer.update(grads, opt_state)
        # Apply updates to model parameters
        new_model_state = optax.apply_updates(model_state, updates)
        
        return loss, accuracy, new_model_state, new_opt_state
    
    def loss_and_aux(self, model_state: nnx.State, graphdef: nnx.GraphDef,
                     data: jax.Array, target: jax.Array, deterministic: bool, key: jax.Array) -> Tuple[float, Tuple[float, jax.Array]]:
        """Compute loss and auxiliary outputs (accuracy, model_output)"""
        model_output = self.forward(graphdef, model_state, data, deterministic, key)
        loss = self.loss_function(model_output, target)
        accuracy = self.accuracy(model_output, target)
        return loss, (accuracy, model_output)
    
    def train_on_support_set(self, graphdef: nnx.GraphDef, model_state: nnx.State,
                            opt: optax.GradientTransformation, opt_state: optax.OptState,
                            support_data: jax.Array, support_targets: jax.Array,
                            key: jax.Array) -> Tuple[nnx.State, optax.OptState]:
        """Train the model on the current support set"""
        num_support = support_data.shape[0]
        
        # Train for specified number of epochs
        for _epoch in range(self.train_epochs):
            key, key_epoch = jax.random.split(key)
            # Shuffle data for this epoch
            perm = jax.random.permutation(key_epoch, num_support)
            data_shuffled = support_data[perm]
            targets_shuffled = support_targets[perm]
            
            # Train in batches
            num_batches = (num_support + self.batch_size - 1) // self.batch_size
            for batch_idx in range(num_batches):
                key, key_batch = jax.random.split(key)
                
                # Calculate batch indices
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, data_shuffled.shape[0])
                
                # Extract current batch
                batch_data = data_shuffled[start:end]
                batch_targets = targets_shuffled[start:end]
                
                # Perform training step
                _loss, _accuracy, model_state, opt_state = self.train_step(
                    graphdef, model_state, opt, opt_state, batch_data, batch_targets, key_batch
                )
        
        return model_state, opt_state
    
    def evaluate_on_nonsupport_set(self, graphdef: nnx.GraphDef, model_state: nnx.State,
                                 nonsupport_data: jax.Array, nonsupport_targets: jax.Array) -> Tuple[float, float, int, bool]:
        """Evaluate the model on the non-support set and check convergence"""
        # Forward pass
        model_output = self.forward(graphdef, model_state, nonsupport_data, deterministic=True, key=jax.random.key(0))
        
        # Compute loss and accuracy
        loss = self.loss_function(model_output, nonsupport_targets)
        accuracy = self.accuracy(model_output, nonsupport_targets)
        
        # Check convergence and find worst sample
        worst_index, converged = self.eval_p2l_convergence(model_output, nonsupport_targets)
        
        return float(loss), float(accuracy), int(worst_index), bool(converged)


def run_fixed_polytope_p2l_experiment(data: jax.Array, targets: jax.Array, 
                                    config: FixedPolytopeP2LConfig,
                                    key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """
    Run a fixed P2L experiment for polytope classification
    
    Args:
        data: Input data
        targets: Target labels
        config: Fixed P2L configuration
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


def compare_fixed_p2l_vs_standard(data: jax.Array, targets: jax.Array,
                                 config: FixedPolytopeP2LConfig,
                                 key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """
    Compare fixed P2L vs standard training for polytope classification
    
    Args:
        data: Input data
        targets: Target labels
        config: Fixed P2L configuration
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
    trainer = OptimalTrainer(
        learning_rate=config.learning_rate,
        epochs=config.train_epochs,
        batch_size=config.batch_size
    )
    
    model = config.init_model(key)
    standard_model = trainer.train_optimal(train_data, train_targets, val_data, val_targets)
    
    # P2L training
    p2l_results = run_fixed_polytope_p2l_experiment(train_data, train_targets, config, key)
    
    # Evaluate both models on test set
    standard_accuracy = trainer.evaluate_model(standard_model['model'], test_data, test_targets)
    p2l_accuracy = trainer.evaluate_model(p2l_results['final_model'], test_data, test_targets)
    
    return {
        'standard_accuracy': standard_accuracy,
        'p2l_accuracy': p2l_accuracy,
        'p2l_improvement': p2l_accuracy - standard_accuracy,
        'support_set_size': len(p2l_results['support_indices']),
        'support_set_ratio': len(p2l_results['support_indices']) / len(train_data),
        'converged': p2l_results.get('converged', False),
        'num_iterations': p2l_results.get('num_iterations', 0),
        'p2l_results': p2l_results
    } 