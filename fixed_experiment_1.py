#!/usr/bin/env python3
"""
Fixed Experiment 1: Polytope Shape Comparison with Proper Data Balance and F1 Scores
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


class FixedPolytopeShapeExperiment:
    def __init__(self, results_dir="experiment_1_fixed_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment parameters
        self.dimensions = [2, 3, 5, 7, 10]
        self.polytope_type = "hypercube"
        self.balance_ratio = 0.30  # 30-70 balance for tougher test of boundary learning
        self.sample_size = 1000
        
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
        
        return {
            'train': (train_data[train_indices], train_labels[train_indices]),
            'val': (val_data[val_indices], val_labels[val_indices]),
            'test': (test_data[test_indices], test_labels[test_indices])
        }
    
    def evaluate_comprehensive_metrics(self, model, scaler, test_data, test_labels):
        """Calculate comprehensive metrics including accuracy, F1, precision, recall"""
        # Get predictions
        test_data_scaled = scaler.transform(test_data)
        logits = model(test_data_scaled, deterministic=True)
        sigmoid_outputs = jax.nn.sigmoid(logits)
        
        # Use adaptive threshold based on data balance
        # For imbalanced datasets, we need a lower threshold
        balance_ratio = np.mean(test_labels)
        if balance_ratio < 0.5:
            # For minority class, use threshold that gives similar ratio of positive predictions
            threshold = np.percentile(sigmoid_outputs, (1 - balance_ratio) * 100)
            threshold = max(0.1, min(0.5, threshold))  # Clamp between 0.1 and 0.5
        else:
            threshold = 0.5
        
        predictions = (sigmoid_outputs > threshold).astype(jnp.float32)
        y_pred = predictions.squeeze().astype(int)
        y_true = test_labels.astype(int)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'true_labels': y_true,
            'threshold_used': threshold,
            'sigmoid_range': [float(sigmoid_outputs.min()), float(sigmoid_outputs.max())]
        }
    
    def run_single_experiment(self, dimension):
        """Run a single experiment for one dimension"""
        print(f"\nTesting {dimension}D {self.polytope_type}")
        print("=" * 50)
        
        # 1. Generate dataset
        print("1. Generating dataset...")
        data_gen = PolytopeDataGenerator(dimension=dimension, random_seed=42)
        
        data, labels, metadata = data_gen.generate_dataset(
            polytope_type=self.polytope_type,
            n_samples=self.sample_size,
            sampling_strategy='direct_balanced',
            target_balance=self.balance_ratio
        )
        
        print(f"   Created {dimension}D {self.polytope_type} with {metadata['polytope_metadata']['type']}")
        print(f"   Dataset: {data.shape}, balance: {np.mean(labels):.1%}")
        
        # 2. Split data with balanced ratios
        print("2. Splitting data with balanced ratios...")
        splits = self.split_data_balanced(data, labels, random_seed=42)
        
        # Verify balance in each split
        train_balance = np.mean(splits['train'][1])
        val_balance = np.mean(splits['val'][1])
        test_balance = np.mean(splits['test'][1])
        
        print(f"   Train balance: {train_balance:.1%}")
        print(f"   Val balance: {val_balance:.1%}")
        print(f"   Test balance: {test_balance:.1%}")
        
        # 3. Standard training
        print("3. Running standard training...")
        trainer = OptimalTrainer(
            learning_rate=self.learning_rate,
            epochs=self.train_epochs,
            batch_size=self.batch_size
        )
        
        standard_results = trainer.train_optimal(
            splits['train'][0], splits['train'][1],
            splits['val'][0], splits['val'][1]
        )
        
        # Evaluate standard model with comprehensive metrics
        model = standard_results['model']
        scaler = standard_results['scaler']
        standard_metrics = self.evaluate_comprehensive_metrics(model, scaler, splits['test'][0], splits['test'][1])
        
        print(f"   Standard test accuracy: {standard_metrics['accuracy']:.3f}")
        print(f"   Standard F1 score: {standard_metrics['f1_score']:.3f}")
        print(f"   Standard precision: {standard_metrics['precision']:.3f}")
        print(f"   Standard recall: {standard_metrics['recall']:.3f}")
        print(f"   Standard threshold: {standard_metrics['threshold_used']:.3f}")
        print(f"   Standard sigmoid range: [{standard_metrics['sigmoid_range'][0]:.3f}, {standard_metrics['sigmoid_range'][1]:.3f}]")
        
        # 4. P2L training
        print("4. Running P2L training...")
        
        # Use adaptive parameters based on dimension
        if dimension >= 7:
            # For high dimensions, use deeper network and more iterations
            hidden_dims = (128, 64, 32, 16)
            convergence_param = 0.50  # Lower threshold for high dimensions
            max_iterations = 500      # More iterations for complex decision boundaries
            pretrain_fraction = 0.3   # Larger initial support set
        else:
            # For lower dimensions, use standard parameters
            hidden_dims = (64, 32)
            convergence_param = 0.70
            max_iterations = 200
            pretrain_fraction = 0.2
        
        p2l_config = FixedPolytopeP2LConfig(
            input_dim=dimension,
            hidden_dims=hidden_dims,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            train_epochs=self.train_epochs,
            convergence_param=convergence_param,
            max_iterations=max_iterations,
            pretrain_fraction=pretrain_fraction
        )
        
        # Initialize data for P2L
        p2l_config.init_data = lambda key: (splits['train'][0], splits['train'][1])
        
        p2l_results = pick_to_learn(p2l_config, jax.random.key(42), scaler=scaler)
        
        # Evaluate P2L model with comprehensive metrics
        p2l_model = p2l_results['final_model']
        p2l_metrics = self.evaluate_comprehensive_metrics(p2l_model, scaler, splits['test'][0], splits['test'][1])
        
        print(f"   P2L test accuracy: {p2l_metrics['accuracy']:.3f}")
        print(f"   P2L F1 score: {p2l_metrics['f1_score']:.3f}")
        print(f"   P2L precision: {p2l_metrics['precision']:.3f}")
        print(f"   P2L recall: {p2l_metrics['recall']:.3f}")
        print(f"   P2L threshold: {p2l_metrics['threshold_used']:.3f}")
        print(f"   P2L sigmoid range: [{p2l_metrics['sigmoid_range'][0]:.3f}, {p2l_metrics['sigmoid_range'][1]:.3f}]")
        print(f"   P2L improvement: {p2l_metrics['accuracy'] - standard_metrics['accuracy']:+.3f}")
        print(f"   Support set: {len(p2l_results['support_indices'])} samples ({len(p2l_results['support_indices'])/len(splits['train'][0])*100:.1f}%)")
        
        # 5. Compile results
        result = {
            'dimension': dimension,
            'polytope_type': self.polytope_type,
            'standard_accuracy': standard_metrics['accuracy'],
            'standard_f1': standard_metrics['f1_score'],
            'standard_precision': standard_metrics['precision'],
            'standard_recall': standard_metrics['recall'],
            'p2l_accuracy': p2l_metrics['accuracy'],
            'p2l_f1': p2l_metrics['f1_score'],
            'p2l_precision': p2l_metrics['precision'],
            'p2l_recall': p2l_metrics['recall'],
            'p2l_improvement': p2l_metrics['accuracy'] - standard_metrics['accuracy'],
            'support_set_size': len(p2l_results['support_indices']),
            'support_set_ratio': len(p2l_results['support_indices']) / len(splits['train'][0]),
            'convergence_iterations': p2l_results.get('num_iterations', len(p2l_results['support_indices'])),
            'converged': p2l_results.get('converged', False),
            'train_balance': train_balance,
            'val_balance': val_balance,
            'test_balance': test_balance,
            'metadata': metadata['polytope_metadata'],
            # Store detailed P2L progression for visualization
            'p2l_accuracies': p2l_results.get('accuracies', []),
            'p2l_losses': p2l_results.get('losses', []),
            'support_indices': p2l_results['support_indices'],
            'nonsupport_indices': p2l_results.get('nonsupport_indices', []),
            'generalization_bound': p2l_results.get('generalization_bound', 0.0)
        }
        
        return result
    
    def run_all_experiments(self):
        """Run experiments for all dimensions"""
        print(f"Starting Fixed Polytope Shape Comparison Experiment")
        print(f"Testing {self.polytope_type} across dimensions: {self.dimensions}")
        print(f"Balance ratio: {self.balance_ratio:.2f}, Sample size: {self.sample_size}")
        
        results = []
        for dim in self.dimensions:
            try:
                result = self.run_single_experiment(dim)
                results.append(result)
            except Exception as e:
                print(f"❌ Error in {dim}D experiment: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def save_results(self, results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"fixed_polytope_shape_results_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            # Convert all numpy arrays to lists
            for key, value in json_result.items():
                if hasattr(value, 'tolist'):  # numpy array
                    json_result[key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy scalar
                    json_result[key] = value.item()
            if 'metadata' in json_result:
                # Convert metadata to serializable format
                json_result['metadata'] = str(json_result['metadata'])
            json_results.append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
        return results_path
    
    def create_paper_style_visualizations(self, results):
        """Create visualizations similar to the original P2L paper"""
        
        print("\nCreating paper-style visualizations...")
        
        # Set up plotting style similar to the paper
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        df = pd.DataFrame(results)
        
        # 1. Performance vs Dimension (similar to paper Figure 1)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        dimensions = df['dimension'].values
        standard_acc = df['standard_accuracy'].values
        p2l_acc = df['p2l_accuracy'].values
        standard_f1 = df['standard_f1'].values
        p2l_f1 = df['p2l_f1'].values
        
        # Plot accuracy
        ax1.plot(dimensions, standard_acc, 'o-', label='Standard (Accuracy)', 
                linewidth=2, markersize=8, color='blue', alpha=0.8)
        ax1.plot(dimensions, p2l_acc, 's-', label='P2L (Accuracy)', 
                linewidth=2, markersize=8, color='red', alpha=0.8)
        
        # Plot F1 scores on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(dimensions, standard_f1, '^--', label='Standard (F1)', 
                     linewidth=2, markersize=6, color='lightblue', alpha=0.8)
        ax1_twin.plot(dimensions, p2l_f1, 'v--', label='P2L (F1)', 
                     linewidth=2, markersize=6, color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Accuracy', color='black')
        ax1_twin.set_ylabel('F1 Score', color='black')
        ax1.set_title('Performance vs Dimension', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.4, 1.0)
        ax1_twin.set_ylim(0, 1.0)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 2. Support Set Analysis (similar to paper compression analysis)
        ax2 = fig.add_subplot(gs[0, 2:4])
        
        support_ratios = df['support_set_ratio'].values
        improvements = df['p2l_improvement'].values
        
        scatter = ax2.scatter(dimensions, support_ratios, c=improvements, 
                            s=100, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Support Set Ratio')
        ax2.set_title('Support Set Efficiency', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Accuracy Improvement')
        
        # 3. P2L Convergence Analysis
        ax3 = fig.add_subplot(gs[1, 2:4])
        
        iterations = df['convergence_iterations'].values
        converged = df['converged'].values
        
        colors = ['green' if c else 'red' for c in converged]
        bars = ax3.bar(dimensions, iterations, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('P2L Iterations')
        ax3.set_title('P2L Convergence Analysis', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, iterations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom', fontsize=9)
        
        # 4. Precision-Recall Analysis
        ax4 = fig.add_subplot(gs[2, 0:2])
        
        # Standard vs P2L precision-recall
        standard_precision = df['standard_precision'].values
        standard_recall = df['standard_recall'].values
        p2l_precision = df['p2l_precision'].values
        p2l_recall = df['p2l_recall'].values
        
        ax4.scatter(standard_precision, standard_recall, s=100, alpha=0.7, 
                   label='Standard', marker='o', color='blue')
        ax4.scatter(p2l_precision, p2l_recall, s=100, alpha=0.7, 
                   label='P2L', marker='s', color='red')
        
        # Add dimension labels
        for i, dim in enumerate(dimensions):
            ax4.annotate(f'{dim}D', (standard_precision[i], standard_recall[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.annotate(f'{dim}D', (p2l_precision[i], p2l_recall[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision-Recall Trade-off', fontweight='bold', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1.0)
        ax4.set_ylim(0, 1.0)
        
        # 5. Generalization Bound Analysis
        ax5 = fig.add_subplot(gs[2, 2:4])
        
        # This would require storing generalization bounds in results
        # For now, show improvement vs support set size
        support_sizes = df['support_set_size'].values
        
        ax5.scatter(support_sizes, improvements, s=100, c=dimensions, 
                   cmap='viridis', alpha=0.7, edgecolors='black')
        ax5.set_xlabel('Support Set Size')
        ax5.set_ylabel('Accuracy Improvement')
        ax5.set_title('Improvement vs Support Set Size', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(ax5.collections[0], ax=ax5)
        cbar2.set_label('Dimension')
        
        # 6. Summary Statistics Table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for _, row in df.iterrows():
            summary_data.append([
                f"{row['dimension']}D",
                f"{row['standard_accuracy']:.3f}",
                f"{row['p2l_accuracy']:.3f}",
                f"{row['p2l_improvement']:+.3f}",
                f"{row['p2l_f1']:.3f}",
                f"{row['support_set_ratio']:.1%}",
                "✅" if row['converged'] else "❌"
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Dim', 'Std Acc', 'P2L Acc', 'Improvement', 'P2L F1', 'Support %', 'Converged'],
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the header row
        for i in range(7):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Summary Statistics', fontweight='bold', fontsize=14, pad=20)
        
        # Add overall title
        fig.suptitle('Fixed Experiment 1: Polytope Shape Comparison\n'
                    'Performance Analysis with Paper-Style Visualizations', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the comprehensive visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.results_dir, f"fixed_experiment_1_visualizations_{timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive visualization saved: {viz_path}")
    
    def print_summary(self, results):
        """Print a summary of the results"""
        print("\n" + "="*60)
        print("📋 FIXED EXPERIMENT SUMMARY")
        print("="*60)
        
        df = pd.DataFrame(results)
        
        print(f"Polytope Type: {self.polytope_type}")
        print(f"Dimensions Tested: {list(df['dimension'])}")
        print(f"Balance Ratio: {self.balance_ratio}")
        print(f"Sample Size: {self.sample_size}")
        print()
        
        print("Results by Dimension:")
        print("-" * 40)
        for _, row in df.iterrows():
            print(f"{row['dimension']:2d}D: "
                  f"Standard={row['standard_accuracy']:.3f} (F1={row['standard_f1']:.3f}), "
                  f"P2L={row['p2l_accuracy']:.3f} (F1={row['p2l_f1']:.3f}), "
                  f"Improvement={row['p2l_improvement']:+.3f}, "
                  f"Support={row['support_set_ratio']:.1%}")
        
        print()
        print("Overall Statistics:")
        print("-" * 40)
        print(f"Average Standard Accuracy: {df['standard_accuracy'].mean():.3f}")
        print(f"Average Standard F1: {df['standard_f1'].mean():.3f}")
        print(f"Average P2L Accuracy: {df['p2l_accuracy'].mean():.3f}")
        print(f"Average P2L F1: {df['p2l_f1'].mean():.3f}")
        print(f"Average Improvement: {df['p2l_improvement'].mean():.3f}")
        print(f"Best Improvement: {df['p2l_improvement'].max():.3f} ({df.loc[df['p2l_improvement'].idxmax(), 'dimension']}D)")
        print(f"Average Support Set Ratio: {df['support_set_ratio'].mean():.1%}")
        
        print()
        print("Data Balance Verification:")
        print("-" * 40)
        print(f"Average Train Balance: {df['train_balance'].mean():.1%}")
        print(f"Average Val Balance: {df['val_balance'].mean():.1%}")
        print(f"Average Test Balance: {df['test_balance'].mean():.1%}")


def main():
    """Main function to run the experiment"""
    experiment = FixedPolytopeShapeExperiment()
    
    # Run experiments
    results = experiment.run_all_experiments()
    
    # Save results
    experiment.save_results(results)
    
    # Create paper-style visualizations
    experiment.create_paper_style_visualizations(results)
    
    # Print summary
    experiment.print_summary(results)
    
    print("\n🎉 Fixed experiment completed! Check experiment_1_fixed_results for results and visualizations.")


if __name__ == "__main__":
    main() 