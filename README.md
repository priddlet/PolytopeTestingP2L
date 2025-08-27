# Comprehensive P2L Research Framework

## Key Features

- **Multiple Polytope Types**: Hypercube, Simplex, Random Halfspaces, Ellipsoid Approximation
- **Direct Balanced Generation**: Uses ground truth knowledge to generate datasets with exact balance ratios
- **Optimal Neural Network Training**: Proven 90%+ accuracy on 3D+ polytopes
- **Comprehensive P2L Integration**: Full integration with the Pick-to-Learn framework
- **Multi-dimensional Support**: Test polytopes in 2D, 3D, 5D, 10D, and beyond
- **Systematic Research**: Automated experiments addressing key research questions
- **Advanced Analytics**: Boundary proximity, shape analysis, generalization bounds
- **Comprehensive Tracking**: All hyperparameters and results logged for analysis

## Core Files

### Essential Components
- **`polytope_generator.py`**: Generates various types of polytopes with random selection
- **`data_generator.py`**: Unified data generator with direct balanced sampling
- **`classifier.py`**: Optimal neural network classifier and trainer
- **`fixed_polytope_p2l.py`**: Fixed P2L configuration with improved convergence
- **`polytope_p2l.py`**: Original P2L configuration (legacy)

### Fixed Experiment Files
- **`fixed_experiment_1.py`**: Fixed polytope shape comparison with proper data balance and F1 scores
- **`fixed_experiment_2.py`**: Fixed polytope types comparison with improved evaluation
- **`fixed_experiment_3.py`**: Fixed data balance and amount effects study
- **`fixed_experiment_4.py`**: Fixed hyperparameter optimization study
- **`fixed_experiment_5.py`**: Fixed generalization bounds analysis
- **`fixed_experiment_6.py`**: Fixed shape elongation effects analysis

### Research Documentation
- **`RESEARCH_QUESTIONS.md`**: Comprehensive research questions and directions
- **`P2L_IMPROVEMENT_SUMMARY.md`**: Summary of P2L improvements and fixes
- **`EXPERIMENT_SUMMARY.md`**: Summary of experiment results and findings
- **`README.md`**: This documentation

### Configuration
- **`requirements.txt`**: Python dependencies
- **`__init__.py`**: Package initialization

### Results Directories
- **`experiment_1_fixed_results/`**: Results from fixed experiment 1
- **`experiment_2_fixed_results/`**: Results from fixed experiment 2
- **`experiment_3_fixed_results/`**: Results from fixed experiment 3
- **`experiment_4_fixed_results/`**: Results from fixed experiment 4
- **`experiment_5_fixed_results/`**: Results from fixed experiment 5 (generalization bounds)
- **`experiment_6_fixed_results/`**: Results from fixed experiment 6 (shape elongation)
- **`experiment_3_fixed_p2l_results/`**: Additional P2L-specific results

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Fixed Experiments
```python
# Run fixed experiment 1 (polytope shape comparison)
from fixed_experiment_1 import FixedPolytopeShapeExperiment
experiment = FixedPolytopeShapeExperiment()
results = experiment.run_comprehensive_experiment()

# Run fixed experiment 2 (polytope types)
from fixed_experiment_2 import FixedPolytopeTypesExperiment
experiment = FixedPolytopeTypesExperiment()
results = experiment.run_comprehensive_experiment()

# Run fixed experiment 3 (data balance effects)
from fixed_experiment_3 import FixedDataBalanceExperiment
experiment = FixedDataBalanceExperiment()
results = experiment.run_comprehensive_experiment()

# Run fixed experiment 4 (hyperparameter optimization)
from fixed_experiment_4 import FixedHyperparameterExperiment
experiment = FixedHyperparameterExperiment()
results = experiment.run_comprehensive_experiment()

# Run fixed experiment 5 (generalization bounds)
from fixed_experiment_5 import FixedGeneralizationBoundsExperiment
experiment = FixedGeneralizationBoundsExperiment()
results = experiment.run_comprehensive_experiment()

# Run fixed experiment 6 (shape elongation effects)
from fixed_experiment_6 import FixedShapeElongationExperiment
experiment = FixedShapeElongationExperiment()
results = experiment.run_comprehensive_experiment()

## Key Components

### Polytope Generator
- **Random Selection**: Automatically selects polytope type to save computation
- **Multiple Types**: Hypercube, Simplex, Random Halfspaces, Ellipsoid Approximation
- **Metadata Tracking**: Returns polytope information for analysis

### Data Generator
- **Direct Balanced Generation**: Uses ground truth to achieve exact balance ratios
- **Multiple Sampling Strategies**: Uniform, Adaptive, Biased, Boundary-Focused, Direct Balanced
- **Proper Test Splits**: 70% train, 15% validation, 15% test with same distribution

### Optimal Classifier
- **Proven Architecture**: 32-32-1 network with batch normalization
- **Optimal Training**: Learning rate scheduling, weight decay, early stopping
- **90%+ Accuracy**: Achieved on 3D+ polytopes with realistic data

### Fixed P2L Integration
- **Improved Convergence**: Better convergence parameters and sample selection
- **Proper Evaluation**: Held-out test set evaluation with F1 scores
- **Convergence Tracking**: Monitors P2L convergence and support set efficiency

## Research Framework

### Fixed Experiment Studies
- **Experiment 1**: Polytope shape comparison across dimensions (2D, 3D, 5D, 7D, 10D)
- **Experiment 2**: Polytope types comparison (hypercube, simplex, random halfspaces, ellipsoid)
- **Experiment 3**: Data balance and amount effects (balance ratios: 5%, 10%, 20%, 30%, 50%)
- **Experiment 4**: Hyperparameter optimization (learning rates, convergence thresholds)
- **Experiment 5**: Generalization bounds analysis (P2L configurations and theoretical guarantees)
- **Experiment 6**: Shape elongation effects (elongation factors: 1.0, 1.5, 2.0, 3.0, 5.0)

### Supported Parameters
- **Polytope Types**: `['hypercube', 'simplex', 'random_halfspaces', 'ellipsoid_approximation']`
- **Dimensions**: `[2, 3, 5, 7, 10]` (extensible)
- **Balance Ratios**: `[0.05, 0.10, 0.20, 0.30, 0.50]` (5% to 50% inside points)
- **Sample Sizes**: `[500, 1000, 2000, 5000]`
- **P2L Configs**: Conservative, Balanced, Aggressive

### Results Tracking
- **Comprehensive Metrics**: Accuracy, F1 score, precision, recall, P2L improvement
- **Advanced Analytics**: Boundary proximity, shape characteristics, convergence analysis
- **Automated Analysis**: Summary statistics and visualizations
- **JSON Output**: Detailed results for further analysis
- **Progress Tracking**: Intermediate saves and progress reporting

## Research Questions Addressed

### 1. Dimension Effects on Learnability
- **Systematic Testing**: 2D, 3D, 5D, 7D, 10D polytopes with multiple repetitions
- **Key Metrics**: Standard vs P2L accuracy, F1 scores, support set efficiency
- **Expected Insights**: How dimension affects learning difficulty and P2L efficiency

### 2. Polytope Types Comparison
- **Type Analysis**: Hypercube, simplex, random halfspaces, ellipsoid approximation
- **Complexity Metrics**: Geometric complexity vs. learnability
- **Expected Insights**: Which polytope types are easier/harder to learn

### 3. Data Balance Impact
- **Balance Ratios**: 5%, 10%, 20%, 30%, 50% inside points
- **Systematic Testing**: Each polytope type with each balance ratio
- **Expected Insights**: How data distribution affects P2L convergence

### 4. Hyperparameter Optimization
- **Parameter Sweep**: Learning rates, convergence thresholds, pretrain fractions
- **Optimization Analysis**: Best parameters for different scenarios
- **Expected Insights**: Optimal P2L configurations for different problems

### 5. Generalization Bounds Analysis
- **P2L Configurations**: Conservative, Balanced, Aggressive settings
- **Bound Computation**: Based on support set size and sample size
- **Expected Insights**: Optimal P2L parameters for theoretical guarantees

### 6. Shape Elongation Effects
- **Elongation Factors**: 1.0 (regular) to 5.0 (highly elongated)
- **Shape Metrics**: Aspect ratio, regularity, volume estimation
- **Expected Insights**: How polytope shape affects learnability

## Hyperparameter Summary

### Optimal Neural Network
```python
OptimalTrainer(
    learning_rate=0.001,
    batch_size=32,
    epochs=30,
    hidden_dim=32,
    weight_decay=1e-5
)
```

### Fixed P2L Configurations
```python
# Conservative
{
    'convergence_param': 0.95,
    'max_iterations': 100,
    'train_epochs': 20,
    'pretrain_fraction': 0.1
}

# Balanced (Recommended)
{
    'convergence_param': 0.90,
    'max_iterations': 75,
    'train_epochs': 15,
    'pretrain_fraction': 0.12
}

# Aggressive
{
    'convergence_param': 0.85,
    'max_iterations': 50,
    'train_epochs': 10,
    'pretrain_fraction': 0.15
}
```

### Data Generation
```python
# Direct balanced generation (recommended)
data_gen.generate_dataset(
    polytope_type='hypercube',
    n_samples=1000,
    sampling_strategy='direct_balanced',
    target_balance=0.3  # 30% inside points
)
```

## Experiment Log

### Latest Improvements (Fixed Versions)
- **Proper F1 Score Calculation**: Fixed evaluation metrics for imbalanced data
- **Improved Data Splitting**: Balanced splits preserving class ratios
- **Better P2L Convergence**: Adaptive convergence parameters
- **Comprehensive Evaluation**: Multiple metrics and statistical analysis
- **Robust Error Handling**: Better handling of edge cases

### Performance Achievements
- **90%+ Accuracy**: Consistently achieved on 3D+ polytopes
- **Exact Balance Control**: Direct generation ensures target ratios
- **Robust P2L Integration**: Proper convergence and evaluation
- **Comprehensive Testing**: Systematic evaluation across parameters

## Usage Examples

### Basic Usage
```python
# Generate a dataset
from data_generator import PolytopeDataGenerator
data_gen = PolytopeDataGenerator(dimension=10)
data, labels, metadata = data_gen.generate_dataset(
    polytope_type='hypercube',
    n_samples=1000,
    sampling_strategy='direct_balanced',
    target_balance=0.3
)

# Train a classifier
from classifier import OptimalTrainer
trainer = OptimalTrainer()
results = trainer.train_optimal(train_data, train_labels, val_data, val_labels)
```

### Fixed P2L Integration
```python
# Run fixed P2L experiment
from fixed_polytope_p2l import FixedPolytopeP2LConfig
from p2l import pick_to_learn

config = FixedPolytopeP2LConfig(
    input_dim=10,
    convergence_param=0.90,
    max_iterations=75
)

p2l_results = pick_to_learn(config, data, labels)
```

### Fixed Experiments
```python
# Run fixed experiment suite
from fixed_experiment_1 import FixedPolytopeShapeExperiment

experiment = FixedPolytopeShapeExperiment()
results = experiment.run_comprehensive_experiment()
```

## Notes

- **Random Seed**: Set to 42 for reproducibility
- **Test Set**: Always held out before any training
- **Balance Ratios**: 30% inside points often provides good learnability
- **Dimensions**: 3D+ polytopes achieve 90%+ accuracy consistently
- **P2L Config**: Balanced configuration recommended for most cases
- **Fixed Versions**: All experiments now use improved evaluation and convergence
