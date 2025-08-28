#!/usr/bin/env python3
"""
Fixed Experiment 4: Hyperparameter Studies with Proper Data Balance and F1 Scores
Test different hyperparameters (learning rate, batch size, convergence criteria)
for P2L to understand their impact on performance
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
from sklearn.preprocessing import StandardScaler


class FixedHyperparameterExperiment:
    def __init__(self, results_dir="experiment_4_fixed_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment parameters
        self.dimensions = [5, 10]  # Focus on higher dimensions
        self.learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        self.batch_sizes = [16, 32, 64, 128]
        self.convergence_params = [0.75, 0.80, 0.85]  # Reduced to more reasonable values
        self.max_iterations = [50, 100]  # Reduced to avoid long runs
        
        # Fixed parameters
        self.balance_ratio = 0.3
        self.sample_size = 500  # Reduced for faster experiments
        self.train_epochs = 20  # Reduced for faster experiments
    
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
        
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        val_data = val_data[val_indices]
        val_labels = val_labels[val_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        # Calculate actual balances
        train_balance = np.mean(train_labels)
        val_balance = np.mean(val_labels)
        test_balance = np.mean(test_labels)
        
        return {
            'train': (train_data, train_labels),
            'val': (val_data, val_labels),
            'test': (test_data, test_labels),
            'train_balance': train_balance,
            'val_balance': val_balance,
            'test_balance': test_balance
        }
    
    def run_single_experiment(self, dimension, learning_rate, batch_size, convergence_param, max_iter, study_type):
        """Run a single experiment for one hyperparameter configuration"""
        print(f"\nTesting {dimension}D, lr={learning_rate}, batch={batch_size}, conv={convergence_param}, max_iter={max_iter}")
        print("=" * 70)
        
        # 1. Generate polytope
        print("1. Generating polytope...")
        generator = PolytopeGenerator(dimension=dimension)
        A, b, metadata = generator.create_hypercube()
        print(f"   Created {dimension}D hypercube")
        
        # 2. Generate dataset
        print("2. Generating dataset...")
        data_gen = PolytopeDataGenerator(
            dimension=dimension,
            random_seed=42
        )
        
        data, labels, _ = data_gen.generate_dataset(
            polytope_type='hypercube',
            n_samples=self.sample_size,
            sampling_strategy='direct_balanced',
            target_balance=self.balance_ratio,
            metadata={'polytope_metadata': metadata}
        )
        
        print(f"   Dataset: {data.shape}, balance: {np.mean(labels):.1%}")
        
        # 3. Split data with balanced ratios
        print("3. Splitting data with balanced ratios...")
        splits = self.split_data_balanced(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        
        train_balance = splits['train_balance']
        val_balance = splits['val_balance']
        test_balance = splits['test_balance']
        
        print(f"   Train balance: {train_balance:.1%}")
        print(f"   Val balance: {val_balance:.1%}")
        print(f"   Test balance: {test_balance:.1%}")
        
        # 4. Standard training
        print("4. Running standard training...")
        trainer = OptimalTrainer(
            learning_rate=learning_rate,
            epochs=self.train_epochs,
            batch_size=batch_size
        )
        
        standard_results = trainer.train_optimal(
            splits['train'][0], splits['train'][1],
            splits['val'][0], splits['val'][1]
        )
        
        # Evaluate standard model
        model = standard_results['model']
        scaler = standard_results['scaler']
        
        # Scale test data
        test_data_scaled = scaler.transform(splits['test'][0])
        
        # Get predictions
        test_logits = model(test_data_scaled)
        standard_predictions = (test_logits > 0).astype(int).flatten()
        
        standard_accuracy = np.mean(standard_predictions == splits['test'][1])
        standard_f1 = f1_score(splits['test'][1], standard_predictions, zero_division=0)
        standard_precision = precision_score(splits['test'][1], standard_predictions, zero_division=0)
        standard_recall = recall_score(splits['test'][1], standard_predictions, zero_division=0)
        
        print(f"   Standard test accuracy: {standard_accuracy:.3f}")
        print(f"   Standard F1 score: {standard_f1:.3f}")
        print(f"   Standard precision: {standard_precision:.3f}")
        print(f"   Standard recall: {standard_recall:.3f}")
        
        # 5. P2L training
        print("5. Running P2L training...")
        p2l_config = PolytopeP2LConfig(
            input_dim=dimension,
            train_epochs=self.train_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            convergence_param=convergence_param,
            max_iterations=max_iter
        )
        
        p2l_results = pick_to_learn(p2l_config, jax.random.key(42))
        
        # Evaluate P2L model
        p2l_model = p2l_results['final_model']
        
        # Scale test data - use the same scaler that was used during P2L training
        if 'scaler' in p2l_results:
            test_data_scaled = p2l_results['scaler'].transform(splits['test'][0])
        else:
            # If no scaler in results, create a new one and fit it to training data
            scaler = StandardScaler()
            test_data_scaled = scaler.fit_transform(splits['test'][0])
        
        # Get predictions
        test_logits = p2l_model(test_data_scaled)
        p2l_predictions = (test_logits > 0).astype(int).flatten()
        
        p2l_accuracy = np.mean(p2l_predictions == splits['test'][1])
        p2l_f1 = f1_score(splits['test'][1], p2l_predictions, zero_division=0)
        p2l_precision = precision_score(splits['test'][1], p2l_predictions, zero_division=0)
        p2l_recall = recall_score(splits['test'][1], p2l_predictions, zero_division=0)
        
        print(f"   P2L test accuracy: {p2l_accuracy:.3f}")
        print(f"   P2L F1 score: {p2l_f1:.3f}")
        print(f"   P2L precision: {p2l_precision:.3f}")
        print(f"   P2L recall: {p2l_recall:.3f}")
        print(f"   P2L improvement: {p2l_accuracy - standard_accuracy:+.3f}")
        print(f"   Support set: {len(p2l_results['support_indices'])} samples ({len(p2l_results['support_indices'])/len(splits['train'][0]):.1%})")
        
        # 6. Compile results
        result = {
            'dimension': dimension,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'convergence_param': convergence_param,
            'max_iterations': max_iter,
            'study_type': study_type,
            'standard_accuracy': standard_accuracy,
            'standard_f1': standard_f1,
            'standard_precision': standard_precision,
            'standard_recall': standard_recall,
            'p2l_accuracy': p2l_accuracy,
            'p2l_f1': p2l_f1,
            'p2l_precision': p2l_precision,
            'p2l_recall': p2l_recall,
            'p2l_improvement': p2l_accuracy - standard_accuracy,
            'support_set_size': len(p2l_results['support_indices']),
            'support_set_ratio': len(p2l_results['support_indices']) / len(splits['train'][0]),
            'convergence_iterations': len(p2l_results['support_indices']),
            'converged': p2l_results.get('converged', False),
            'train_balance': train_balance,
            'val_balance': val_balance,
            'test_balance': test_balance,
            'metadata': str(metadata)
        }
        
        return result
    
    def run_learning_rate_study(self):
        """Run learning rate study"""
        print("\n📚 LEARNING RATE STUDY")
        print("=" * 50)
        results = []
        
        for dim in self.dimensions:
            for lr in self.learning_rates:
                try:
                    result = self.run_single_experiment(
                        dimension=dim,
                        learning_rate=lr,
                        batch_size=32,  # Fixed
                        convergence_param=0.80,  # Fixed
                        max_iter=100,  # Fixed
                        study_type='learning_rate'
                    )
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error in learning rate study: {e}")
                    continue
        
        return results
    
    def run_batch_size_study(self):
        """Run batch size study"""
        print("\n📚 BATCH SIZE STUDY")
        print("=" * 50)
        results = []
        
        for dim in self.dimensions:
            for batch_size in self.batch_sizes:
                try:
                    result = self.run_single_experiment(
                        dimension=dim,
                        learning_rate=0.001,  # Fixed
                        batch_size=batch_size,
                        convergence_param=0.80,  # Fixed
                        max_iter=100,  # Fixed
                        study_type='batch_size'
                    )
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error in batch size study: {e}")
                    continue
        
        return results
    
    def run_convergence_study(self):
        """Run convergence parameter study"""
        print("\n📚 CONVERGENCE PARAMETER STUDY")
        print("=" * 50)
        results = []
        
        for dim in self.dimensions:
            for conv_param in self.convergence_params:
                try:
                    result = self.run_single_experiment(
                        dimension=dim,
                        learning_rate=0.001,  # Fixed
                        batch_size=32,  # Fixed
                        convergence_param=conv_param,
                        max_iter=100,  # Fixed
                        study_type='convergence_param'
                    )
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error in convergence study: {e}")
                    continue
        
        return results
    
    def run_max_iterations_study(self):
        """Run max iterations study"""
        print("\n📚 MAX ITERATIONS STUDY")
        print("=" * 50)
        results = []
        
        for dim in self.dimensions:
            for max_iter in self.max_iterations:
                try:
                    result = self.run_single_experiment(
                        dimension=dim,
                        learning_rate=0.001,  # Fixed
                        batch_size=32,  # Fixed
                        convergence_param=0.80,  # Fixed
                        max_iter=max_iter,
                        study_type='max_iterations'
                    )
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error in max iterations study: {e}")
                    continue
        
        return results
    
    def run_all_experiments(self):
        """Run all hyperparameter studies"""
        print(f"Starting Fixed Hyperparameter Experiment")
        print(f"Dimensions: {self.dimensions}")
        print(f"Learning rates: {self.learning_rates}")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Convergence params: {self.convergence_params}")
        print(f"Max iterations: {self.max_iterations}")
        
        all_results = []
        
        # Run all studies
        all_results.extend(self.run_learning_rate_study())
        all_results.extend(self.run_batch_size_study())
        all_results.extend(self.run_convergence_study())
        all_results.extend(self.run_max_iterations_study())
        
        return all_results
    
    def save_results(self, results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"fixed_hyperparameter_study_results_{timestamp}.json")
        
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
                        json_result[key] = list(value)
                    else:
                        json_result[key] = value
                except:
                    json_result[key] = str(value)
            json_results.append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
        return results_path
    
    def create_paper_style_visualizations(self, results):
        """Create comprehensive paper-style visualizations"""
        print("\nCreating paper-style visualizations...")
        
        df = pd.DataFrame(results)
        
        # Ensure p2l_improvement is numeric to avoid dtype issues
        df['p2l_improvement_numeric'] = pd.to_numeric(df['p2l_improvement'], errors='coerce')
        
        # Set up the figure with a 3x3 grid
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Learning Rate Impact
        ax1 = fig.add_subplot(gs[0, 0])
        lr_data = df[df['study_type'] == 'learning_rate'].groupby('learning_rate')['p2l_improvement_numeric'].mean()
        lr_data.plot(kind='bar', ax=ax1, width=0.6, color='skyblue')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('P2L Improvement')
        ax1.set_title('Learning Rate Impact on P2L Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 2: Batch Size Impact
        ax2 = fig.add_subplot(gs[0, 1])
        batch_data = df[df['study_type'] == 'batch_size'].groupby('batch_size')['p2l_improvement_numeric'].mean()
        batch_data.plot(kind='bar', ax=ax2, width=0.6, color='lightgreen')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('P2L Improvement')
        ax2.set_title('Batch Size Impact on P2L Performance')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 3: Convergence Parameter Impact
        ax3 = fig.add_subplot(gs[0, 2])
        conv_data = df[df['study_type'] == 'convergence_param'].groupby('convergence_param')['p2l_improvement_numeric'].mean()
        conv_data.plot(kind='bar', ax=ax3, width=0.6, color='orange')
        ax3.set_xlabel('Convergence Parameter')
        ax3.set_ylabel('P2L Improvement')
        ax3.set_title('Convergence Parameter Impact on P2L Performance')
        ax3.tick_params(axis='x', rotation=0)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 4: Max Iterations Impact
        ax4 = fig.add_subplot(gs[1, 0])
        iter_data = df[df['study_type'] == 'max_iterations'].groupby('max_iterations')['p2l_improvement_numeric'].mean()
        iter_data.plot(kind='bar', ax=ax4, width=0.6, color='red')
        ax4.set_xlabel('Max Iterations')
        ax4.set_ylabel('P2L Improvement')
        ax4.set_title('Max Iterations Impact on P2L Performance')
        ax4.tick_params(axis='x', rotation=0)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 5: Dimension Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        dim_data = df.groupby('dimension')['p2l_improvement_numeric'].mean()
        dim_data.plot(kind='bar', ax=ax5, width=0.6, color='purple')
        ax5.set_xlabel('Dimension')
        ax5.set_ylabel('P2L Improvement')
        ax5.set_title('P2L Performance by Dimension')
        ax5.tick_params(axis='x', rotation=0)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Panel 6: Support Set Efficiency
        ax6 = fig.add_subplot(gs[1, 2])
        support_data = df.groupby('study_type')['support_set_ratio'].mean()
        support_data.plot(kind='bar', ax=ax6, width=0.6, color='brown')
        ax6.set_xlabel('Study Type')
        ax6.set_ylabel('Support Set Ratio')
        ax6.set_title('Support Set Efficiency by Study Type')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Panel 7: Heatmap of Learning Rate vs Batch Size
        ax7 = fig.add_subplot(gs[2, 0])
        # Create pivot table for heatmap
        heatmap_data = df[df['study_type'].isin(['learning_rate', 'batch_size'])].pivot_table(
            values='p2l_improvement_numeric', 
            index='learning_rate', 
            columns='batch_size', 
            aggfunc='mean'
        )
        # Ensure heatmap data is numeric and fill NaN values
        heatmap_data = heatmap_data.astype(float).fillna(0)
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax7)
        ax7.set_xlabel('Batch Size')
        ax7.set_ylabel('Learning Rate')
        ax7.set_title('P2L Improvement Heatmap: LR vs Batch Size')
        
        # Panel 8: Convergence Analysis
        ax8 = fig.add_subplot(gs[2, 1])
        conv_iter_data = df.groupby('study_type')['convergence_iterations'].mean()
        conv_iter_data.plot(kind='bar', ax=ax8, width=0.6, color='teal')
        ax8.set_xlabel('Study Type')
        ax8.set_ylabel('Average Convergence Iterations')
        ax8.set_title('Convergence Analysis by Study Type')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
        
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
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax9.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # Add overall title
        fig.suptitle('Experiment 4: Hyperparameter Studies\nP2L Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.results_dir, f"fixed_experiment_4_visualizations_{timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved: {viz_path}")
        return viz_path
    
    def print_summary(self, results):
        """Print a comprehensive summary of the results"""
        print("\n" + "="*60)
        print("📋 FIXED HYPERPARAMETER STUDY SUMMARY")
        print("="*60)
        
        df = pd.DataFrame(results)
        
        # Ensure p2l_improvement is numeric to avoid dtype issues
        df['p2l_improvement_numeric'] = pd.to_numeric(df['p2l_improvement'], errors='coerce')
        
        print(f"Dimensions Tested: {list(df['dimension'].unique())}")
        print(f"Learning Rates Tested: {list(df['learning_rate'].unique())}")
        print(f"Batch Sizes Tested: {list(df['batch_size'].unique())}")
        print(f"Convergence Params Tested: {list(df['convergence_param'].unique())}")
        print(f"Max Iterations Tested: {list(df['max_iterations'].unique())}")
        print(f"Total Experiments: {len(df)}")
        print()
        
        # Overall statistics
        print("Overall Statistics:")
        print("-" * 50)
        print(f"Average Standard Accuracy: {df['standard_accuracy'].mean():.3f}")
        print(f"Average Standard F1: {df['standard_f1'].mean():.3f}")
        print(f"Average P2L Accuracy: {df['p2l_accuracy'].mean():.3f}")
        print(f"Average P2L F1: {df['p2l_f1'].mean():.3f}")
        print(f"Average Improvement: {df['p2l_improvement_numeric'].mean():.3f}")
        print(f"Best Improvement: {df['p2l_improvement_numeric'].max():.3f}")
        print(f"Average Support Set Ratio: {df['support_set_ratio'].mean():.1%}")
        
        print()
        print("Data Balance Verification:")
        print("-" * 50)
        print(f"Average Train Balance: {df['train_balance'].mean():.1%}")
        print(f"Average Val Balance: {df['val_balance'].mean():.1%}")
        print(f"Average Test Balance: {df['test_balance'].mean():.1%}")
        
        print()
        print("Best Performing Configurations by Study:")
        print("-" * 50)
        
        for study_type in df['study_type'].unique():
            study_data = df[df['study_type'] == study_type]
            best_idx = study_data['p2l_improvement_numeric'].idxmax()
            best_row = study_data.loc[best_idx]
            print(f"{study_type:15s}: {best_row['dimension']}D, "
                  f"lr={best_row['learning_rate']}, batch={best_row['batch_size']}, "
                  f"conv={best_row['convergence_param']}, max_iter={best_row['max_iterations']} "
                  f"({best_row['p2l_improvement_numeric']:+.3f})")
        
        print()
        print("Overall Best Configuration:")
        print("-" * 40)
        best_idx = df['p2l_improvement_numeric'].idxmax()
        best_row = df.loc[best_idx]
        print(f"Dimension: {best_row['dimension']}D")
        print(f"Learning Rate: {best_row['learning_rate']}")
        print(f"Batch Size: {best_row['batch_size']}")
        print(f"Convergence Param: {best_row['convergence_param']}")
        print(f"Max Iterations: {best_row['max_iterations']}")
        print(f"P2L Improvement: {best_row['p2l_improvement_numeric']:+.3f}")
        
        print()
        print("Key Insights:")
        print("-" * 40)
        # Find best learning rate overall
        best_lr = df.groupby('learning_rate')['p2l_improvement_numeric'].mean().idxmax()
        print(f"Best overall learning rate: {best_lr}")
        
        # Find best batch size overall
        best_batch = df.groupby('batch_size')['p2l_improvement_numeric'].mean().idxmax()
        print(f"Best overall batch size: {best_batch}")
        
        # Find best convergence param overall
        best_conv = df.groupby('convergence_param')['p2l_improvement_numeric'].mean().idxmax()
        print(f"Best overall convergence param: {best_conv}")
        
        # Find best max iterations overall
        best_iter = df.groupby('max_iterations')['p2l_improvement_numeric'].mean().idxmax()
        print(f"Best overall max iterations: {best_iter}")


def main():
    """Main function to run the experiment"""
    experiment = FixedHyperparameterExperiment()
    
    # Run experiments
    results = experiment.run_all_experiments()
    
    if results:
        # Save results
        experiment.save_results(results)
        
        # Create visualizations
        experiment.create_paper_style_visualizations(results)
        
        # Print summary
        experiment.print_summary(results)
        
        print("\n🎉 Fixed hyperparameter experiment completed! Check experiment_4_fixed_results for results.")
    else:
        print("❌ No experiments completed successfully.")


if __name__ == "__main__":
    main() 