#!/usr/bin/env python3
"""
Fixed Experiment 5: Generalization Bounds Analysis
Analyze how generalization bounds are affected by hyperparameter changes and P2L configurations
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import jax
import jax.numpy as jnp

# Configure JAX for better GPU memory management
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
print("Using GPU with very conservative memory management")

# Use local p2l implementation with safety-critical asymmetric margin selection
from p2l import pick_to_learn

# Import our modules
from polytope_generator import PolytopeGenerator
from data_generator import PolytopeDataGenerator
from classifier import OptimalPolytopeClassifier, OptimalTrainer
from fixed_polytope_p2l import FixedPolytopeP2LConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class FixedGeneralizationBoundsExperiment:
    def __init__(self, results_dir="experiment_5_fixed_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment parameters (reduced for speed)
        self.dimensions = [5, 10]  # 2 dimensions instead of 3
        self.polytope_type = "hypercube"
        self.balance_ratio = 0.30
        self.sample_size = 500  # Reduced from 1000 for speed
        
        # P2L configurations for generalization bounds analysis (reduced for speed)
        self.convergence_params = [0.85, 0.90, 0.95]  # 3 values instead of 4
        self.pretrain_fractions = [0.10, 0.15]  # 2 values instead of 4
        self.max_iterations = [50, 75]  # 2 values instead of 4
        
        # Fixed hyperparameters
        self.train_epochs = 30
        self.learning_rate = 0.001
        self.batch_size = 32
    
    def split_data_balanced(self, data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=None):
        """Split data while preserving balance ratios in each split"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Separate inside and outside points
        inside_mask = labels == 1
        outside_mask = labels == 0
        
        inside_data = data[inside_mask]
        inside_labels = labels[inside_mask]
        outside_data = data[outside_mask]
        outside_labels = labels[outside_mask]
        
        # Split inside points
        n_inside = len(inside_data)
        n_inside_train = int(n_inside * train_ratio)
        n_inside_val = int(n_inside * val_ratio)
        
        inside_indices = np.random.permutation(n_inside)
        inside_train_idx = inside_indices[:n_inside_train]
        inside_val_idx = inside_indices[n_inside_train:n_inside_train + n_inside_val]
        inside_test_idx = inside_indices[n_inside_train + n_inside_val:]
        
        # Split outside points
        n_outside = len(outside_data)
        n_outside_train = int(n_outside * train_ratio)
        n_outside_val = int(n_outside * val_ratio)
        
        outside_indices = np.random.permutation(n_outside)
        outside_train_idx = outside_indices[:n_outside_train]
        outside_val_idx = outside_indices[n_outside_train:n_outside_train + n_outside_val]
        outside_test_idx = outside_indices[n_outside_train + n_outside_val:]
        
        # Combine splits
        train_data = np.vstack([inside_data[inside_train_idx], outside_data[outside_train_idx]])
        train_labels = np.concatenate([inside_labels[inside_train_idx], outside_labels[outside_train_idx]])
        
        val_data = np.vstack([inside_data[inside_val_idx], outside_data[outside_val_idx]])
        val_labels = np.concatenate([inside_labels[inside_val_idx], outside_labels[outside_val_idx]])
        
        test_data = np.vstack([inside_data[inside_test_idx], outside_data[outside_test_idx]])
        test_labels = np.concatenate([inside_labels[inside_test_idx], outside_labels[outside_test_idx]])
        
        # Shuffle each split
        train_indices = np.random.permutation(len(train_data))
        val_indices = np.random.permutation(len(val_data))
        test_indices = np.random.permutation(len(test_data))
        
        return (
            train_data[train_indices], train_labels[train_indices],
            val_data[val_indices], val_labels[val_indices],
            test_data[test_indices], test_labels[test_indices]
        )
    
    def compute_generalization_bound(self, support_set_size, sample_size, confidence=0.95):
        """Compute generalization bound based on support set size and sample size"""
        # Using a simplified version of the generalization bound
        # In practice, this would be more sophisticated based on P2L theory
        delta = 1 - confidence
        bound = np.sqrt((np.log(2 * support_set_size) + np.log(1/delta)) / (2 * sample_size))
        return bound
    
    def run_single_experiment(self, dimension, convergence_param, pretrain_fraction, max_iter):
        """Run a single experiment for one P2L configuration"""
        print(f"\nTesting {dimension}D, conv={convergence_param}, pretrain={pretrain_fraction}, max_iter={max_iter}")
        print("=" * 70)
        
        # 1. Generate dataset
        print("1. Generating dataset...")
        data_gen = PolytopeDataGenerator(
            dimension=dimension,
            random_seed=42
        )
        
        data, labels, metadata = data_gen.generate_dataset(
            polytope_type=self.polytope_type,
            n_samples=self.sample_size,
            sampling_strategy='direct_balanced',
            target_balance=self.balance_ratio
        )
        
        print(f"   Created {dimension}D {self.polytope_type}")
        print(f"   Dataset: {data.shape}, balance: {np.mean(labels):.1%}")
        
        # 2. Split data
        print("2. Splitting data...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.split_data_balanced(
            data, labels, random_seed=42
        )
        
        print(f"   Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        # 3. Standard training
        print("3. Running standard training...")
        trainer = OptimalTrainer(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.train_epochs
        )
        
        standard_results = trainer.train_optimal(train_data, train_labels, val_data, val_labels)
        
        # Evaluate on test set
        test_data_scaled = standard_results['scaler'].transform(test_data)
        test_logits = standard_results['model'](test_data_scaled, deterministic=True)
        test_predictions = (jax.nn.sigmoid(test_logits) > 0.5).astype(jnp.float32)
        standard_accuracy = np.mean(test_predictions.flatten() == test_labels)
        standard_f1 = f1_score(test_labels, test_predictions.flatten(), average='binary')
        
        print(f"   Standard - Accuracy: {standard_accuracy:.3f}, F1: {standard_f1:.3f}")
        
        # 4. P2L training
        print("4. Running P2L training...")
        config = FixedPolytopeP2LConfig(
            input_dim=dimension,
            convergence_param=convergence_param,
            pretrain_fraction=pretrain_fraction,
            max_iterations=max_iter,
            train_epochs=self.train_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size
        )
        
        # Initialize data for P2L
        config.init_data = lambda key: (train_data, train_labels)
        
        p2l_results = pick_to_learn(config, jax.random.key(42))
        
        # Extract P2L metrics
        p2l_accuracy = p2l_results['final_accuracy']
        p2l_f1 = p2l_results['final_f1_score']
        support_set_size = p2l_results['support_set_size']
        convergence_iterations = p2l_results['convergence_iterations']
        
        print(f"   P2L - Accuracy: {p2l_accuracy:.3f}, F1: {p2l_f1:.3f}")
        print(f"   Support set size: {support_set_size}, Convergence: {convergence_iterations}")
        
        # 5. Compute generalization bounds
        print("5. Computing generalization bounds...")
        generalization_bound = self.compute_generalization_bound(support_set_size, self.sample_size)
        bound_tightness = p2l_accuracy - generalization_bound
        
        print(f"   Generalization bound: {generalization_bound:.3f}")
        print(f"   Bound tightness: {bound_tightness:.3f}")
        
        # 6. Compile results
        result = {
            'dimension': dimension,
            'convergence_param': convergence_param,
            'pretrain_fraction': pretrain_fraction,
            'max_iterations': max_iter,
            'standard_accuracy': standard_accuracy,
            'standard_f1': standard_f1,
            'p2l_accuracy': p2l_accuracy,
            'p2l_f1': p2l_f1,
            'p2l_improvement': p2l_accuracy - standard_accuracy,
            'support_set_size': support_set_size,
            'support_set_ratio': support_set_size / self.sample_size,
            'convergence_iterations': convergence_iterations,
            'generalization_bound': generalization_bound,
            'bound_tightness': bound_tightness,
            'polytope_type': self.polytope_type,
            'balance_ratio': self.balance_ratio,
            'sample_size': self.sample_size,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_all_experiments(self):
        """Run experiments for all P2L configurations"""
        print(f"Starting Fixed Generalization Bounds Experiment")
        print(f"Dimensions: {self.dimensions}")
        print(f"Convergence params: {self.convergence_params}")
        print(f"Pretrain fractions: {self.pretrain_fractions}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Polytope type: {self.polytope_type}")
        print(f"Balance ratio: {self.balance_ratio:.2f}, Sample size: {self.sample_size}")
        print("=" * 80)
        
        all_results = []
        
        # Run experiments for each dimension and configuration
        for dimension in self.dimensions:
            for convergence_param in self.convergence_params:
                for pretrain_fraction in self.pretrain_fractions:
                    for max_iter in self.max_iterations:
                        try:
                            result = self.run_single_experiment(
                                dimension, convergence_param, pretrain_fraction, max_iter
                            )
                            all_results.append(result)
                            
                            # Save intermediate results
                            if len(all_results) % 10 == 0:
                                self.save_intermediate_results(all_results)
                                
                        except Exception as e:
                            print(f"Error in experiment: {e}")
                            continue
        
        # Save final results
        self.save_final_results(all_results)
        
        # Create visualizations
        self.create_paper_style_visualizations(all_results)
        
        return all_results
    
    def save_intermediate_results(self, results):
        """Save intermediate results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intermediate_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Intermediate results saved: {filepath}")
    
    def save_final_results(self, results):
        """Save final results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generalization_bounds_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Final results saved: {filepath}")
        
        # Also save summary statistics
        self.save_summary_statistics(results, timestamp)
    
    def save_summary_statistics(self, results, timestamp):
        """Save summary statistics"""
        df = pd.DataFrame(results)
        
        # Overall statistics
        summary = {
            'total_experiments': len(results),
            'dimensions_tested': list(df['dimension'].unique()),
            'convergence_params_tested': list(df['convergence_param'].unique()),
            'pretrain_fractions_tested': list(df['pretrain_fraction'].unique()),
            'max_iterations_tested': list(df['max_iterations'].unique()),
            'mean_standard_accuracy': df['standard_accuracy'].mean(),
            'mean_p2l_accuracy': df['p2l_accuracy'].mean(),
            'mean_p2l_improvement': df['p2l_improvement'].mean(),
            'mean_support_set_ratio': df['support_set_ratio'].mean(),
            'mean_generalization_bound': df['generalization_bound'].mean(),
            'mean_bound_tightness': df['bound_tightness'].mean(),
            'timestamp': timestamp
        }
        
        filename = f"summary_statistics_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary statistics saved: {filepath}")
    
    def create_paper_style_visualizations(self, results):
        """Create comprehensive paper-style visualizations for generalization bounds"""
        print("\nCreating paper-style visualizations...")
        
        df = pd.DataFrame(results)
        
        # Set up plotting style similar to the paper
        plt.style.use('default')
        fig_width = 12
        fig_height = 8
        
        # 1. Generalization bounds vs support set size
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        fig.suptitle('Generalization Bounds Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Bound vs Support Set Size
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['support_set_size'], df['generalization_bound'], 
                             c=df['dimension'], cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('Support Set Size')
        ax1.set_ylabel('Generalization Bound')
        ax1.set_title('Generalization Bound vs Support Set Size')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Dimension')
        
        # Plot 2: Bound Tightness vs Convergence Parameter
        ax2 = axes[0, 1]
        for dim in df['dimension'].unique():
            dim_data = df[df['dimension'] == dim]
            ax2.scatter(dim_data['convergence_param'], dim_data['bound_tightness'], 
                       label=f'{dim}D', alpha=0.7, s=50)
        ax2.set_xlabel('Convergence Parameter')
        ax2.set_ylabel('Bound Tightness (Accuracy - Bound)')
        ax2.set_title('Bound Tightness vs Convergence Parameter')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Support Set Ratio vs Pretrain Fraction
        ax3 = axes[1, 0]
        for dim in df['dimension'].unique():
            dim_data = df[df['dimension'] == dim]
            ax3.scatter(dim_data['pretrain_fraction'], dim_data['support_set_ratio'], 
                       label=f'{dim}D', alpha=0.7, s=50)
        ax3.set_xlabel('Pretrain Fraction')
        ax3.set_ylabel('Support Set Ratio')
        ax3.set_title('Support Set Efficiency vs Pretrain Fraction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: P2L Improvement vs Max Iterations
        ax4 = axes[1, 1]
        for dim in df['dimension'].unique():
            dim_data = df[df['dimension'] == dim]
            ax4.scatter(dim_data['max_iterations'], dim_data['p2l_improvement'], 
                       label=f'{dim}D', alpha=0.7, s=50)
        ax4.set_xlabel('Max Iterations')
        ax4.set_ylabel('P2L Improvement')
        ax4.set_title('P2L Improvement vs Max Iterations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.results_dir, f"generalization_bounds_analysis_{timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved: {viz_path}")
        return viz_path
    
    def run_comprehensive_experiment(self):
        """Main method to run the complete generalization bounds experiment"""
        print("=" * 80)
        print("FIXED EXPERIMENT 5: GENERALIZATION BOUNDS ANALYSIS")
        print("=" * 80)
        print("This experiment analyzes how generalization bounds are affected by")
        print("P2L hyperparameter changes and configuration settings.")
        print("=" * 80)
        
        results = self.run_all_experiments()
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED")
        print("=" * 80)
        print(f"Total experiments run: {len(results)}")
        print(f"Results saved to: {self.results_dir}")
        print("=" * 80)
        
        return results


if __name__ == "__main__":
    experiment = FixedGeneralizationBoundsExperiment()
    results = experiment.run_comprehensive_experiment() 