#!/usr/bin/env python3
"""
Fixed Experiment 3: Data Balance and Amount Studies with Proper Data Balance and F1 Scores
Test different data balances and sample sizes for 3D, 5D, and 10D hypercubes
to understand how P2L performance varies with data characteristics
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
from polytope_p2l import PolytopeP2LConfig
from sklearn.metrics import f1_score, precision_score, recall_score


class FixedDataBalanceAmountExperiment:
    def __init__(self, results_dir="experiment_3_fixed_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment parameters
        self.dimensions = [3, 5, 10]
        self.balance_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]  # 10% to 50% inside
        self.sample_sizes = [500, 1000, 2000]  # Different data amounts
        
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
        balance_ratio = np.mean(test_labels)
        if balance_ratio < 0.5:
            threshold = np.percentile(sigmoid_outputs, (1 - balance_ratio) * 100)
            threshold = max(0.1, min(0.5, threshold))
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
    
    def run_single_experiment(self, dimension, balance_ratio, sample_size):
        """Run a single experiment for one configuration"""
        print(f"\nTesting {dimension}D, balance={balance_ratio:.1%}, size={sample_size}")
        print("=" * 60)
        
        # 1. Generate dataset
        print("1. Generating dataset...")
        print(f"  Target: {int(sample_size * balance_ratio)} inside, {int(sample_size * (1 - balance_ratio))} outside")
        data_gen = PolytopeDataGenerator(
            dimension=dimension,
            random_seed=42
        )
        
        data, labels, metadata = data_gen.generate_dataset(
            polytope_type='hypercube',
            n_samples=sample_size,
            sampling_strategy='direct_balanced',
            target_balance=balance_ratio
        )
        
        print(f"   Created {dimension}D hypercube")
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
        
        # 4. P2L training with adaptive parameters
        print("4. Running P2L training...")
        
        # Adaptive P2L configuration based on dimension
        if dimension >= 7:
            # For high dimensions, use deeper network and more conservative parameters
            hidden_dims = (128, 64, 32, 16)
            convergence_param = 0.50
            max_iterations = 500
            pretrain_fraction = 0.3
        else:
            # For lower dimensions, use simpler network and standard parameters
            hidden_dims = (64, 32)
            convergence_param = 0.70
            max_iterations = 200
            pretrain_fraction = 0.2
        
        p2l_config = PolytopeP2LConfig(
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
        
        # 5. Compile results with essential data only
        result = {
            'dimension': dimension,
            'balance_ratio': balance_ratio,
            'sample_size': sample_size,
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
            'convergence_iterations': len(p2l_results['support_indices']),
            'converged': p2l_results.get('converged', False),
            'train_balance': train_balance,
            'val_balance': val_balance,
            'test_balance': test_balance,
            'metadata': str(metadata['polytope_metadata'])
        }
        
        return result
    
    def create_paper_style_visualizations(self, results):
        """Create comprehensive paper-style visualizations for experiment 3"""
        print("\nCreating paper-style visualizations...")
        
        df = pd.DataFrame(results)
        
        # Ensure p2l_improvement is numeric to avoid dtype issues
        df['p2l_improvement_numeric'] = pd.to_numeric(df['p2l_improvement'], errors='coerce')
        
        # Set up the figure with 6 panels (2x3 layout)
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Performance by Balance Ratio and Dimension
        ax1 = fig.add_subplot(gs[0, 0])
        balance_dim_data = df.groupby(['balance_ratio', 'dimension'])['p2l_accuracy'].mean().unstack()
        balance_dim_data.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xlabel('Balance Ratio')
        ax1.set_ylabel('P2L Accuracy')
        ax1.set_title('P2L Performance by Balance Ratio and Dimension')
        ax1.legend(title='Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: F1 Score Comparison by Balance Ratio
        ax2 = fig.add_subplot(gs[0, 1])
        f1_comparison = df.groupby('balance_ratio')[['standard_f1', 'p2l_f1']].mean()
        f1_comparison.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xlabel('Balance Ratio')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Comparison by Balance Ratio')
        ax2.legend(['Standard', 'P2L'])
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Support Set Efficiency by Sample Size
        ax3 = fig.add_subplot(gs[0, 2])
        support_efficiency = df.groupby('sample_size')['support_set_ratio'].mean()
        support_efficiency.plot(kind='bar', ax=ax3, width=0.6, color='skyblue')
        ax3.set_xlabel('Sample Size')
        ax3.set_ylabel('Support Set Ratio')
        ax3.set_title('Support Set Efficiency by Sample Size')
        ax3.tick_params(axis='x', rotation=0)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Precision-Recall Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        for dim in sorted(df['dimension'].unique()):
            dim_data = df[df['dimension'] == dim]
            ax4.scatter(dim_data['p2l_precision'], dim_data['p2l_recall'], 
                       s=50, alpha=0.7, label=f'{dim}D')
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision-Recall Analysis by Dimension')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        # Panel 5: P2L Convergence Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        convergence_data = df.groupby('dimension')['convergence_iterations'].mean()
        convergence_data.plot(kind='bar', ax=ax5, width=0.6, color='lightblue')
        ax5.set_xlabel('Dimension')
        ax5.set_ylabel('Average Convergence Iterations')
        ax5.set_title('P2L Convergence by Dimension')
        ax5.tick_params(axis='x', rotation=0)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Improvement vs Balance Ratio
        ax6 = fig.add_subplot(gs[1, 2])
        improvement_data = df.groupby('balance_ratio')['p2l_improvement_numeric'].mean()
        colors = ['red' if x < 0 else 'blue' for x in improvement_data]
        improvement_data.plot(kind='bar', ax=ax6, width=0.6, color=colors)
        ax6.set_xlabel('Balance Ratio')
        ax6.set_ylabel('P2L Improvement')
        ax6.set_title('P2L Improvement by Balance Ratio')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 7: Sample Size Impact on Performance
        ax7 = fig.add_subplot(gs[2, 0])
        size_performance = df.groupby('sample_size')[['standard_accuracy', 'p2l_accuracy']].mean()
        size_performance.plot(kind='bar', ax=ax7, width=0.8)
        ax7.set_xlabel('Sample Size')
        ax7.set_ylabel('Accuracy')
        ax7.set_title('Performance by Sample Size')
        ax7.legend(['Standard', 'P2L'])
        ax7.tick_params(axis='x', rotation=0)
        ax7.grid(True, alpha=0.3)
        
        # Panel 8: Balance Ratio Heatmap
        ax8 = fig.add_subplot(gs[2, 1])
        heatmap_data = df.groupby(['balance_ratio', 'dimension'])['p2l_improvement_numeric'].mean().unstack()
        # Ensure heatmap data is numeric and fill NaN values
        heatmap_data = heatmap_data.astype(float).fillna(0)
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax8)
        ax8.set_xlabel('Dimension')
        ax8.set_ylabel('Balance Ratio')
        ax8.set_title('P2L Improvement Heatmap')
        
        # Panel 9: Summary Statistics Table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('tight')
        ax9.axis('off')
        
        # Calculate summary statistics
        summary_stats = [
            ['Metric', 'Standard', 'P2L', 'Improvement'],
            ['Accuracy', f"{df['standard_accuracy'].mean():.3f}", f"{df['p2l_accuracy'].mean():.3f}", f"{df['p2l_improvement_numeric'].mean():.3f}"],
            ['F1 Score', f"{df['standard_f1'].mean():.3f}", f"{df['p2l_f1'].mean():.3f}", "N/A"],
            ['Precision', f"{df['standard_precision'].mean():.3f}", f"{df['p2l_precision'].mean():.3f}", "N/A"],
            ['Recall', f"{df['standard_recall'].mean():.3f}", f"{df['p2l_recall'].mean():.3f}", "N/A"],
            ['Support Set', "N/A", f"{df['support_set_ratio'].mean():.1%}", "N/A"],
            ['Convergence', "N/A", f"{df['convergence_iterations'].mean():.1f}", "N/A"]
        ]
        
        table = ax9.table(cellText=summary_stats[1:], colLabels=summary_stats[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_stats)):
            for j in range(len(summary_stats[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#2196F3')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax9.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # Add overall title
        fig.suptitle('Experiment 3: Data Balance and Amount Studies\nP2L Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.results_dir, f"fixed_experiment_3_visualizations_{timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved: {viz_path}")
        return viz_path
    
    def run_all_experiments(self):
        """Run experiments for all configurations"""
        print(f"Starting Fixed Data Balance and Amount Experiment")
        print(f"Dimensions: {self.dimensions}")
        print(f"Balance ratios: {self.balance_ratios}")
        print(f"Sample sizes: {self.sample_sizes}")
        print(f"Total experiments: {len(self.dimensions) * len(self.balance_ratios) * len(self.sample_sizes)}")
        
        results = []
        for dim in self.dimensions:
            for balance in self.balance_ratios:
                for size in self.sample_sizes:
                    try:
                        result = self.run_single_experiment(dim, balance, size)
                        results.append(result)
                    except Exception as e:
                        print(f"❌ Error in {dim}D, balance={balance:.1%}, size={size} experiment: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        return results
    
    def save_results(self, results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"fixed_data_balance_amount_results_{timestamp}.json")
        
        # Convert numpy arrays and JAX arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = {}
            # Convert all arrays to lists
            for key, value in result.items():
                try:
                    if hasattr(value, 'tolist'):  # numpy or JAX array
                        json_result[key] = value.tolist()
                    elif hasattr(value, 'item'):  # numpy scalar
                        json_result[key] = value.item()
                    elif hasattr(value, '__iter__') and not isinstance(value, (str, list, dict)):
                        # Handle other iterable types
                        try:
                            json_result[key] = list(value)
                        except:
                            json_result[key] = str(value)
                    elif hasattr(value, 'dtype'):  # JAX array scalar
                        json_result[key] = float(value)
                    else:
                        # Handle regular Python types
                        json_result[key] = value
                except Exception as e:
                    # If all else fails, convert to string
                    json_result[key] = str(value)
            
            if 'metadata' in json_result:
                json_result['metadata'] = str(json_result['metadata'])
            json_results.append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
        return results_path
    
    def print_summary(self, results):
        """Print a summary of the results"""
        print("\n" + "="*80)
        print("📋 FIXED DATA BALANCE AND AMOUNT EXPERIMENT SUMMARY")
        print("="*80)
        
        df = pd.DataFrame(results)
        
        print(f"Dimensions Tested: {list(df['dimension'].unique())}")
        print(f"Balance Ratios: {list(df['balance_ratio'].unique())}")
        print(f"Sample Sizes: {list(df['sample_size'].unique())}")
        print(f"Total Experiments: {len(df)}")
        print()
        
        # Overall statistics
        print("Overall Statistics:")
        print("-" * 50)
        print(f"Average Standard Accuracy: {df['standard_accuracy'].mean():.3f}")
        print(f"Average Standard F1: {df['standard_f1'].mean():.3f}")
        print(f"Average P2L Accuracy: {df['p2l_accuracy'].mean():.3f}")
        print(f"Average P2L F1: {df['p2l_f1'].mean():.3f}")
        # Convert p2l_improvement to numeric to avoid dtype issues
        df['p2l_improvement_numeric'] = pd.to_numeric(df['p2l_improvement'], errors='coerce')
        print(f"Average Improvement: {df['p2l_improvement_numeric'].mean():.3f}")
        print(f"Best Improvement: {df['p2l_improvement_numeric'].max():.3f}")
        print(f"Average Support Set Ratio: {df['support_set_ratio'].mean():.1%}")
        
        print()
        print("Data Balance Verification:")
        print("-" * 50)
        print(f"Average Train Balance: {df['train_balance'].mean():.1%}")
        print(f"Average Val Balance: {df['val_balance'].mean():.1%}")
        print(f"Average Test Balance: {df['test_balance'].mean():.1%}")
        
        # Analysis by dimension
        print()
        print("Analysis by Dimension:")
        print("-" * 50)
        for dim in sorted(df['dimension'].unique()):
            dim_data = df[df['dimension'] == dim]
            print(f"{dim}D:")
            print(f"  Avg Standard Accuracy: {dim_data['standard_accuracy'].mean():.3f}")
            print(f"  Avg Standard F1: {dim_data['standard_f1'].mean():.3f}")
            print(f"  Avg P2L Accuracy: {dim_data['p2l_accuracy'].mean():.3f}")
            print(f"  Avg P2L F1: {dim_data['p2l_f1'].mean():.3f}")
            print(f"  Avg Improvement: {dim_data['p2l_improvement_numeric'].mean():.3f}")
        
        # Analysis by balance ratio
        print()
        print("Analysis by Balance Ratio:")
        print("-" * 50)
        for balance in sorted(df['balance_ratio'].unique()):
            balance_data = df[df['balance_ratio'] == balance]
            print(f"Balance {balance:.1%}:")
            print(f"  Avg Standard Accuracy: {balance_data['standard_accuracy'].mean():.3f}")
            print(f"  Avg Standard F1: {balance_data['standard_f1'].mean():.3f}")
            print(f"  Avg P2L Accuracy: {balance_data['p2l_accuracy'].mean():.3f}")
            print(f"  Avg P2L F1: {balance_data['p2l_f1'].mean():.3f}")
            print(f"  Avg Improvement: {balance_data['p2l_improvement_numeric'].mean():.3f}")
        
        # Analysis by sample size
        print()
        print("Analysis by Sample Size:")
        print("-" * 50)
        for size in sorted(df['sample_size'].unique()):
            size_data = df[df['sample_size'] == size]
            print(f"Sample Size {size}:")
            print(f"  Avg Standard Accuracy: {size_data['standard_accuracy'].mean():.3f}")
            print(f"  Avg Standard F1: {size_data['standard_f1'].mean():.3f}")
            print(f"  Avg P2L Accuracy: {size_data['p2l_accuracy'].mean():.3f}")
            print(f"  Avg P2L F1: {size_data['p2l_f1'].mean():.3f}")
            print(f"  Avg Improvement: {size_data['p2l_improvement_numeric'].mean():.3f}")
        
        # Top 10 best improvements
        print()
        print("Top 10 Best P2L Improvements:")
        print("-" * 50)
        top_improvements = df.nlargest(10, 'p2l_improvement_numeric')
        for _, row in top_improvements.iterrows():
            print(f"{row['dimension']:2d}D, balance={row['balance_ratio']:.1%}, size={row['sample_size']:4d}: "
                  f"Standard={row['standard_accuracy']:.3f}, P2L={row['p2l_accuracy']:.3f}, "
                  f"Improvement={row['p2l_improvement_numeric']:+.3f}")


def main():
    """Main function to run the experiment"""
    experiment = FixedDataBalanceAmountExperiment()
    
    # Run experiments
    results = experiment.run_all_experiments()
    
    # Save results
    experiment.save_results(results)
    
    # Create visualizations
    experiment.create_paper_style_visualizations(results)
    
    # Print summary
    experiment.print_summary(results)
    
    print("\n🎉 Fixed data balance and amount experiment completed! Check experiment_3_fixed_results for results.")


if __name__ == "__main__":
    main() 