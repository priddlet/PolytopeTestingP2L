#!/usr/bin/env python3
"""
Fixed Experiment 6: Shape Elongation Effects Analysis
Analyze how polytope shape characteristics (elongation, regularity) affect learnability
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

# Add parent directory to path
sys.path.append('..')
from picktolearn.p2l import pick_to_learn

# Import our modules
from polytope_generator import PolytopeGenerator
from data_generator import PolytopeDataGenerator
from classifier import OptimalPolytopeClassifier, OptimalTrainer
from fixed_polytope_p2l import FixedPolytopeP2LConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class FixedShapeElongationExperiment:
    def __init__(self, results_dir="experiment_6_fixed_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment parameters (reduced for speed)
        self.dimensions = [5, 10]  # 2 dimensions instead of 3
        self.base_polytope_type = "hypercube"
        self.balance_ratio = 0.30
        self.sample_size = 500  # Reduced from 1000 for speed
        
        # Shape elongation factors (reduced for speed)
        self.elongation_factors = [1.0, 2.0, 5.0]  # 3 values instead of 5
        
        # Fixed hyperparameters
        self.train_epochs = 30
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # P2L configuration
        self.p2l_config = {
            'convergence_param': 0.90,
            'pretrain_fraction': 0.12,
            'max_iterations': 75
        }
    
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
    
    def create_elongated_polytope(self, dimension, elongation_factor, polytope_type="hypercube"):
        """Create an elongated version of a polytope by scaling dimensions"""
        # Generate base polytope
        polytope_gen = PolytopeGenerator(dimension=dimension, random_seed=42)
        
        if polytope_type == "hypercube":
            # For hypercube, we can create elongation by scaling the first dimension
            base_polytope = polytope_gen.generate_hypercube()
            
            # Apply elongation by scaling the first dimension
            elongation_matrix = np.eye(dimension)
            elongation_matrix[0, 0] = elongation_factor
            
            # Transform the polytope vertices
            elongated_vertices = base_polytope['vertices'] @ elongation_matrix.T
            
            # Update the polytope
            elongated_polytope = base_polytope.copy()
            elongated_polytope['vertices'] = elongated_vertices
            elongated_polytope['elongation_factor'] = elongation_factor
            elongated_polytope['aspect_ratio'] = elongation_factor
            
            return elongated_polytope
        
        elif polytope_type == "simplex":
            # For simplex, create elongation by scaling the first dimension
            base_polytope = polytope_gen.generate_simplex()
            
            # Apply elongation
            elongation_matrix = np.eye(dimension)
            elongation_matrix[0, 0] = elongation_factor
            
            elongated_vertices = base_polytope['vertices'] @ elongation_matrix.T
            
            elongated_polytope = base_polytope.copy()
            elongated_polytope['vertices'] = elongated_vertices
            elongated_polytope['elongation_factor'] = elongation_factor
            elongated_polytope['aspect_ratio'] = elongation_factor
            
            return elongated_polytope
        
        else:
            # For other types, use the base polytope
            return polytope_gen.generate_polytope(polytope_type)
    
    def compute_shape_metrics(self, polytope, elongation_factor):
        """Compute shape characteristics of the polytope"""
        vertices = polytope['vertices']
        
        # Compute bounding box
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # Compute dimensions
        dimensions = max_coords - min_coords
        
        # Aspect ratio (max dimension / min dimension)
        aspect_ratio = np.max(dimensions) / np.min(dimensions)
        
        # Volume estimation (for hypercube)
        if 'hypercube' in polytope.get('type', ''):
            volume = np.prod(dimensions)
        else:
            volume = np.nan
        
        # Regularity measure (how close to a regular shape)
        # Lower values indicate more regular shapes
        regularity = np.std(dimensions) / np.mean(dimensions)
        
        # Elongation measure
        elongation_measure = elongation_factor
        
        return {
            'aspect_ratio': aspect_ratio,
            'volume': volume,
            'regularity': regularity,
            'elongation_measure': elongation_measure,
            'dimensions': dimensions.tolist()
        }
    
    def run_single_experiment(self, dimension, elongation_factor):
        """Run a single experiment for one elongation factor"""
        print(f"\nTesting {dimension}D with elongation factor {elongation_factor}")
        print("=" * 60)
        
        # 1. Create elongated polytope
        print("1. Creating elongated polytope...")
        polytope = self.create_elongated_polytope(dimension, elongation_factor, self.base_polytope_type)
        shape_metrics = self.compute_shape_metrics(polytope, elongation_factor)
        
        print(f"   Polytope type: {self.base_polytope_type}")
        print(f"   Aspect ratio: {shape_metrics['aspect_ratio']:.2f}")
        print(f"   Regularity: {shape_metrics['regularity']:.3f}")
        print(f"   Elongation measure: {shape_metrics['elongation_measure']:.2f}")
        
        # 2. Generate dataset using the elongated polytope
        print("2. Generating dataset...")
        data_gen = PolytopeDataGenerator(
            dimension=dimension,
            random_seed=42
        )
        
        # For elongated polytopes, we need to modify the data generation
        # This is a simplified approach - in practice, you'd need to implement
        # proper sampling for elongated polytopes
        data, labels, metadata = data_gen.generate_dataset(
            polytope_type=self.base_polytope_type,
            n_samples=self.sample_size,
            sampling_strategy='direct_balanced',
            target_balance=self.balance_ratio
        )
        
        # Apply elongation transformation to the data
        if elongation_factor != 1.0:
            elongation_matrix = np.eye(dimension)
            elongation_matrix[0, 0] = elongation_factor
            data = data @ elongation_matrix.T
        
        print(f"   Dataset: {data.shape}, balance: {np.mean(labels):.1%}")
        
        # 3. Split data
        print("3. Splitting data...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.split_data_balanced(
            data, labels, random_seed=42
        )
        
        print(f"   Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        # 4. Standard training
        print("4. Running standard training...")
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
        
        # 5. P2L training
        print("5. Running P2L training...")
        config = FixedPolytopeP2LConfig(
            input_dim=dimension,
            convergence_param=self.p2l_config['convergence_param'],
            pretrain_fraction=self.p2l_config['pretrain_fraction'],
            max_iterations=self.p2l_config['max_iterations'],
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
        
        # 6. Compile results
        result = {
            'dimension': dimension,
            'elongation_factor': elongation_factor,
            'aspect_ratio': shape_metrics['aspect_ratio'],
            'regularity': shape_metrics['regularity'],
            'elongation_measure': shape_metrics['elongation_measure'],
            'volume': shape_metrics['volume'],
            'standard_accuracy': standard_accuracy,
            'standard_f1': standard_f1,
            'p2l_accuracy': p2l_accuracy,
            'p2l_f1': p2l_f1,
            'p2l_improvement': p2l_accuracy - standard_accuracy,
            'support_set_size': support_set_size,
            'support_set_ratio': support_set_size / self.sample_size,
            'convergence_iterations': convergence_iterations,
            'polytope_type': self.base_polytope_type,
            'balance_ratio': self.balance_ratio,
            'sample_size': self.sample_size,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_all_experiments(self):
        """Run experiments for all elongation factors and dimensions"""
        print(f"Starting Fixed Shape Elongation Effects Experiment")
        print(f"Dimensions: {self.dimensions}")
        print(f"Elongation factors: {self.elongation_factors}")
        print(f"Base polytope type: {self.base_polytope_type}")
        print(f"Balance ratio: {self.balance_ratio:.2f}, Sample size: {self.sample_size}")
        print("=" * 80)
        
        all_results = []
        
        # Run experiments for each dimension and elongation factor
        for dimension in self.dimensions:
            for elongation_factor in self.elongation_factors:
                try:
                    result = self.run_single_experiment(dimension, elongation_factor)
                    all_results.append(result)
                    
                    # Save intermediate results
                    if len(all_results) % 5 == 0:
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
        filename = f"shape_elongation_results_{timestamp}.json"
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
            'elongation_factors_tested': list(df['elongation_factor'].unique()),
            'mean_standard_accuracy': df['standard_accuracy'].mean(),
            'mean_p2l_accuracy': df['p2l_accuracy'].mean(),
            'mean_p2l_improvement': df['p2l_improvement'].mean(),
            'mean_support_set_ratio': df['support_set_ratio'].mean(),
            'mean_aspect_ratio': df['aspect_ratio'].mean(),
            'mean_regularity': df['regularity'].mean(),
            'timestamp': timestamp
        }
        
        filename = f"summary_statistics_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary statistics saved: {filepath}")
    
    def create_paper_style_visualizations(self, results):
        """Create comprehensive paper-style visualizations for shape elongation effects"""
        print("\nCreating paper-style visualizations...")
        
        df = pd.DataFrame(results)
        
        # Set up plotting style similar to the paper
        plt.style.use('default')
        fig_width = 12
        fig_height = 8
        
        # 1. Performance vs elongation
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        fig.suptitle('Shape Elongation Effects Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy vs Elongation Factor
        ax1 = axes[0, 0]
        for dim in df['dimension'].unique():
            dim_data = df[df['dimension'] == dim]
            ax1.plot(dim_data['elongation_factor'], dim_data['standard_accuracy'], 
                    'o-', label=f'{dim}D Standard', alpha=0.7, linewidth=2)
            ax1.plot(dim_data['elongation_factor'], dim_data['p2l_accuracy'], 
                    's-', label=f'{dim}D P2L', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Elongation Factor')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Elongation Factor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: P2L Improvement vs Elongation Factor
        ax2 = axes[0, 1]
        for dim in df['dimension'].unique():
            dim_data = df[df['dimension'] == dim]
            ax2.plot(dim_data['elongation_factor'], dim_data['p2l_improvement'], 
                    'o-', label=f'{dim}D', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Elongation Factor')
        ax2.set_ylabel('P2L Improvement')
        ax2.set_title('P2L Improvement vs Elongation Factor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Support Set Ratio vs Elongation Factor
        ax3 = axes[1, 0]
        for dim in df['dimension'].unique():
            dim_data = df[df['dimension'] == dim]
            ax3.plot(dim_data['elongation_factor'], dim_data['support_set_ratio'], 
                    'o-', label=f'{dim}D', alpha=0.7, linewidth=2)
        ax3.set_xlabel('Elongation Factor')
        ax3.set_ylabel('Support Set Ratio')
        ax3.set_title('Support Set Efficiency vs Elongation Factor')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regularity vs Performance
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['regularity'], df['p2l_improvement'], 
                             c=df['dimension'], cmap='viridis', alpha=0.7, s=50)
        ax4.set_xlabel('Regularity Measure')
        ax4.set_ylabel('P2L Improvement')
        ax4.set_title('P2L Improvement vs Shape Regularity')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Dimension')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.results_dir, f"shape_elongation_analysis_{timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved: {viz_path}")
        return viz_path
    
    def run_comprehensive_experiment(self):
        """Main method to run the complete shape elongation effects experiment"""
        print("=" * 80)
        print("FIXED EXPERIMENT 6: SHAPE ELONGATION EFFECTS ANALYSIS")
        print("=" * 80)
        print("This experiment analyzes how polytope shape characteristics")
        print("(elongation, regularity) affect learnability and P2L performance.")
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
    experiment = FixedShapeElongationExperiment()
    results = experiment.run_comprehensive_experiment() 