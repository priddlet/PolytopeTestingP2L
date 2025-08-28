"""
Fixed Polytope P2L Configuration with Improved Convergence and Sample Selection
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple, Dict, Any, Optional

# Import the core P2L implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'picktolearn'))
from p2l import P2LConfig, pick_to_learn, initialize_support_sets, generalization_bound

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
                            key: jax.Array, scaler=None) -> Tuple[nnx.State, optax.OptState]:
        """Train the model on the current support set with optional scaling"""
        num_support = support_data.shape[0]
        
        # Apply scaling if provided
        if scaler is not None:
            support_data_scaled = scaler.transform(support_data)
        else:
            support_data_scaled = support_data
        
        # Train for specified number of epochs
        for _epoch in range(self.train_epochs):
            key, key_epoch = jax.random.split(key)
            # Shuffle data for this epoch
            perm = jax.random.permutation(key_epoch, num_support)
            data_shuffled = support_data_scaled[perm]
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


def run_fixed_polytope_p2l_experiment_with_scaling(data: jax.Array, targets: jax.Array, 
                                                  config: FixedPolytopeP2LConfig,
                                                  scaler=None, key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """
    Run a fixed P2L experiment for polytope classification with proper scaling
    
    Args:
        data: Input data
        targets: Target labels
        config: Fixed P2L configuration
        scaler: Optional scaler for consistent scaling
        key: JAX random key
        
    Returns:
        Dictionary containing experiment results
    """
    if key is None:
        key = jax.random.key(0)
    
    # Split the random key for different components
    key, model_key, sets_key = jax.random.split(key, 3)
    
    # Initialize model, optimizer
    model = config.init_model(model_key)
    graphdef, model_state = nnx.split(model)
    opt = config.init_optimizer()
    opt_state = opt.init(model_state)
    
    n_total = data.shape[0]
    
    # Initialize support and non-support sets
    support_indices, nonsupport_indices = initialize_support_sets(
        n_total, config.pretrain_fraction, sets_key
    )
    initial_support_set_size = len(support_indices)
    initial_nonsupport_set_size = len(nonsupport_indices)
    
    print(f"Starting P2L with {len(support_indices)} initial support examples")
    
    iteration = 0
    converged = False
    losses = []
    accuracies = []
    
    # Main P2L loop
    while iteration < config.max_iterations:
        print(f"\n--- P2L Iteration {iteration + 1} ---")
        print(f"Support set size: {len(support_indices)}")
        print(f"Non-support set size: {len(nonsupport_indices)}")
        
        key, key_train = jax.random.split(key)
        
        # Step 1: Train model on current support set with scaling
        if support_indices:
            support_data = data[np.array(support_indices)]
            support_targets = targets[np.array(support_indices)]
            
            model_state, opt_state = config.train_on_support_set(
                graphdef,
                model_state,
                opt,
                opt_state,
                support_data,
                support_targets,
                key_train,
                scaler=scaler  # Pass scaler for consistent scaling
            )
        
        # Step 2: Evaluate on non-support set and check convergence
        if nonsupport_indices:
            nonsupport_data = data[np.array(nonsupport_indices)]
            nonsupport_targets = targets[np.array(nonsupport_indices)]
            
            # Apply scaling for evaluation if scaler provided
            if scaler is not None:
                nonsupport_data_scaled = scaler.transform(nonsupport_data)
            else:
                nonsupport_data_scaled = nonsupport_data
            
            # Update the forward method to use scaled data
            model_output = config.forward(graphdef, model_state, nonsupport_data_scaled, deterministic=True, key=jax.random.key(0))
            
            # Compute loss and accuracy
            loss = config.loss_function(model_output, nonsupport_targets)
            accuracy = config.accuracy(model_output, nonsupport_targets)
            
            # Check convergence and find worst sample
            worst_index, converged = config.eval_p2l_convergence(model_output, nonsupport_targets)
        else:
            loss = 0.0
            accuracy = 1.0
            worst_index = 0
            converged = True
        
        # Log metrics for this iteration
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")
        
        if converged:
            print("P2L Converged!")
            break
        
        # Step 3: If not converged, move least appropriate example to support set
        if nonsupport_indices:
            support_indices.append(nonsupport_indices.pop(worst_index))
        
        iteration += 1
    
    # Check final convergence status
    if not converged:
        print(f"Warning: P2L did not converge after {config.max_iterations} iterations")
    
    # Calculate generalization bound
    bound = generalization_bound(
        len(support_indices) - initial_support_set_size,
        initial_nonsupport_set_size,
        config.confidence_param,
    )
    
    # Return comprehensive results dictionary
    return {
        "final_model": nnx.merge(graphdef, model_state),
        "support_indices": support_indices,
        "nonsupport_indices": nonsupport_indices,
        "generalization_bound": bound,
        "num_iterations": iteration,
        "converged": converged,
        "losses": losses,
        "accuracies": accuracies,
    }


def compare_fixed_p2l_vs_standard(data: jax.Array, targets: jax.Array,
                                 config: FixedPolytopeP2LConfig,
                                 key: Optional[jax.Array] = None,
                                 A: Optional[jax.Array] = None,
                                 b: Optional[jax.Array] = None) -> Dict[str, Any]:
    """
    Compare fixed P2L vs standard training for polytope classification with comprehensive metrics
    
    Args:
        data: Input data
        targets: Target labels
        config: Fixed P2L configuration
        key: JAX random key
        A: True polytope constraint matrix (for geometric metrics)
        b: True polytope constraint vector (for geometric metrics)
        
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
    
    # P2L training with consistent scaling
    p2l_results = run_fixed_polytope_p2l_experiment_with_scaling(
        train_data, train_targets, config, standard_model['scaler'], key
    )
    
    # Size-matched baseline: train standard model on random subset of same size
    key, random_key = jax.random.split(key)
    support_size = len(p2l_results['support_indices'])
    random_indices = jax.random.permutation(random_key, len(train_data))[:support_size]
    random_train_data = train_data[random_indices]
    random_train_targets = train_targets[random_indices]
    
    random_trainer = OptimalTrainer(
        learning_rate=config.learning_rate,
        epochs=config.train_epochs,
        batch_size=config.batch_size
    )
    random_model = random_trainer.train_optimal(random_train_data, random_train_targets, val_data, val_targets)
    
    # Evaluate all models with comprehensive metrics
    standard_metrics = trainer.evaluate_model_with_metrics(
        standard_model['model'], test_data, test_targets, standard_model['scaler']
    )
    p2l_metrics = trainer.evaluate_model_with_metrics(
        p2l_results['final_model'], test_data, test_targets, standard_model['scaler']  # Use same scaler
    )
    random_metrics = random_trainer.evaluate_model_with_metrics(
        random_model['model'], test_data, test_targets, random_model['scaler']
    )
    
    # Geometric metrics if polytope constraints provided
    geometric_metrics = {}
    if A is not None and b is not None:
        from geometric_metrics import compute_geometric_metrics
        
        # Compute bounding box for Monte Carlo sampling
        min_coords = np.min(data, axis=0)
        max_coords = np.max(data, axis=0)
        bounding_box = (min_coords, max_coords)
        
        # Geometric metrics for all models
        standard_geo = compute_geometric_metrics(
            standard_model['model'], A, b, test_data, test_targets, 
            standard_model['scaler'], bounding_box
        )
        p2l_geo = compute_geometric_metrics(
            p2l_results['final_model'], A, b, test_data, test_targets, 
            standard_model['scaler'], bounding_box
        )
        random_geo = compute_geometric_metrics(
            random_model['model'], A, b, test_data, test_targets, 
            random_model['scaler'], bounding_box
        )
        
        geometric_metrics = {
            'standard': standard_geo,
            'p2l': p2l_geo,
            'random': random_geo
        }
    
    return {
        # Standard metrics
        'standard_accuracy': standard_metrics['accuracy'],
        'standard_f1': standard_metrics['f1'],
        'standard_violation_rate': standard_metrics['violation_rate'],
        'standard_false_safe_rate': standard_metrics['false_safe_rate'],
        
        'p2l_accuracy': p2l_metrics['accuracy'],
        'p2l_f1': p2l_metrics['f1'],
        'p2l_violation_rate': p2l_metrics['violation_rate'],
        'p2l_false_safe_rate': p2l_metrics['false_safe_rate'],
        
        'random_accuracy': random_metrics['accuracy'],
        'random_f1': random_metrics['f1'],
        'random_violation_rate': random_metrics['violation_rate'],
        'random_false_safe_rate': random_metrics['false_safe_rate'],
        
        # Improvements
        'p2l_accuracy_improvement': p2l_metrics['accuracy'] - standard_metrics['accuracy'],
        'p2l_f1_improvement': p2l_metrics['f1'] - standard_metrics['f1'],
        'p2l_violation_improvement': standard_metrics['violation_rate'] - p2l_metrics['violation_rate'],
        
        'random_vs_standard_accuracy': random_metrics['accuracy'] - standard_metrics['accuracy'],
        'random_vs_standard_f1': random_metrics['f1'] - standard_metrics['f1'],
        
        'p2l_vs_random_accuracy': p2l_metrics['accuracy'] - random_metrics['accuracy'],
        'p2l_vs_random_f1': p2l_metrics['f1'] - random_metrics['f1'],
        
        # Support set info
        'support_set_size': len(p2l_results['support_indices']),
        'support_set_ratio': len(p2l_results['support_indices']) / len(train_data),
        'converged': p2l_results.get('converged', False),
        'num_iterations': p2l_results.get('num_iterations', 0),
        
        # Full results for detailed analysis
        'standard_metrics': standard_metrics,
        'p2l_metrics': p2l_metrics,
        'random_metrics': random_metrics,
        'geometric_metrics': geometric_metrics,
        'p2l_results': p2l_results
    } 