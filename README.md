# P2L Research Framework: Pick-to-Learn for Polytope Classification

A comprehensive research framework for evaluating Pick-to-Learn (P2L) algorithms in polytope classification tasks with geometric fidelity analysis.

## 🎯 Overview

This framework provides a complete pipeline for comparing P2L algorithms against standard training methods in polytope classification tasks. It includes:

- **Polytope Generation**: Various polytope types (hypercubes, simplices, random halfspaces)
- **Data Generation**: Balanced datasets with controlled class distributions
- **P2L Implementation**: Complete Pick-to-Learn algorithm with support set selection
- **Comprehensive Evaluation**: Accuracy, F1 scores, violation rates, and geometric fidelity metrics
- **Geometric Verification**: Monte Carlo IoU and volume-based analysis
- **Size-Matched Baselines**: Fair comparison with random subset baselines

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- JAX
- NumPy
- Matplotlib
- Scikit-learn
- Flax

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd polytope_testing
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### 1. Simple P2L vs Standard Comparison

```python
from data_generator import PolytopeDataGenerator
from polytope_generator import PolytopeGenerator
from fixed_polytope_p2l import FixedPolytopeP2LConfig, compare_fixed_p2l_vs_standard

# Generate a 2D hypercube polytope
poly_gen = PolytopeGenerator(dimension=2)
A, b, metadata = poly_gen.create_hypercube()

# Generate balanced dataset
data_gen = PolytopeDataGenerator(dimension=2)
data, targets = data_gen.generate_polytope_data(A, b, n_samples=1000, balance_ratio=0.3)

# Configure P2L
config = FixedPolytopeP2LConfig(
    learning_rate=0.001,
    train_epochs=100,
    batch_size=32,
    pretrain_fraction=0.1,
    max_iterations=50
)

# Run comparison
results = compare_fixed_p2l_vs_standard(data, targets, config)

print(f"Standard Accuracy: {results['standard_accuracy']:.3f}")
print(f"P2L Accuracy: {results['p2l_accuracy']:.3f}")
print(f"P2L Improvement: {results['p2l_accuracy_improvement']:+.3f}")
print(f"Support Set Size: {results['support_set_size']}")
```

#### 2. Geometric Fidelity Analysis

```python
from geometric_metrics import compute_geometric_metrics

# Compute geometric metrics
geo_metrics = compute_geometric_metrics(
    model, A, b, test_data, test_labels,
    auto_scale_samples=True,  # Automatically scale samples for high dimensions
    auto_bounds=True          # Automatically estimate bounding box
)

# Access geometric fidelity metrics
iou = geo_metrics['monte_carlo_metrics']['iou']
false_safe_volume = geo_metrics['monte_carlo_metrics']['false_safe_ratio']
print(f"Geometric IoU: {iou:.3f}")
print(f"False Safe Volume: {false_safe_volume:.3f}")
```

#### 3. Comprehensive Experiment

```python
from comprehensive_experiment import ComprehensivePolytopeExperiment

# Run comprehensive experiment across multiple dimensions
experiment = ComprehensivePolytopeExperiment(
    polytope_type="hypercube",
    dimensions=[2, 5, 10],
    balance_ratio=0.3,
    sample_size=1000
)

results = experiment.run_all_experiments()
experiment.create_comprehensive_visualizations(results)
```

## 📊 Key Features

### 1. **Rigorous Evaluation Metrics**
- **Classification**: Accuracy, F1 Score, Precision, Recall
- **Safety**: Violation Rate, False Safe Rate
- **Geometric**: IoU, Volume Ratios, False Safe/Unsafe Volumes

### 2. **Fair Comparison Baselines**
- **Standard Training**: Full dataset training
- **P2L Training**: Support set selection
- **Size-Matched Random**: Random subset of same size as P2L support set

### 3. **Geometric Fidelity Analysis**
- **Monte Carlo IoU**: Intersection over Union between predicted and true safe regions
- **Volume Analysis**: Predicted vs true safe region volumes
- **Automatic Scaling**: Handles curse of dimensionality automatically

### 4. **Multiple Polytope Types**
- **Hypercubes**: Regular n-dimensional cubes
- **Simplices**: Regular n-dimensional simplices
- **Random Halfspaces**: Randomly generated polytopes
- **Ellipsoid Approximations**: Ellipsoidal polytopes

## 🔧 Configuration

### P2L Configuration

```python
config = FixedPolytopeP2LConfig(
    input_dim=10,              # Input dimension
    learning_rate=0.001,       # Learning rate
    train_epochs=100,          # Training epochs per iteration
    batch_size=32,             # Batch size
    pretrain_fraction=0.1,     # Initial support set fraction
    max_iterations=50,         # Maximum P2L iterations
    convergence_param=0.85,    # Convergence threshold
    confidence_param=0.95      # Confidence parameter
)
```

### Data Generation

```python
data_gen = PolytopeDataGenerator(dimension=10)
data, targets = data_gen.generate_polytope_data(
    A, b,                      # Polytope constraints
    n_samples=1000,            # Number of samples
    balance_ratio=0.3,         # Target safe/unsafe ratio
    sampling_strategy='direct_balanced'  # Sampling strategy
)
```

## 📈 Understanding Results

### Performance Metrics

- **Accuracy**: Standard classification accuracy
- **F1 Score**: Harmonic mean of precision and recall (better for imbalanced data)
- **Violation Rate**: Fraction of unsafe points predicted as safe (critical for safety)
- **False Safe Rate**: Fraction of safe points predicted as unsafe

### Geometric Metrics

- **IoU**: Intersection over Union (1.0 = perfect match)
- **Volume Ratios**: Predicted vs true safe region sizes
- **False Safe Volume**: Volume of unsafe region predicted as safe
- **False Unsafe Volume**: Volume of safe region predicted as unsafe

### P2L Characteristics

- **Support Set Size**: Number of samples selected by P2L
- **Support Set Ratio**: Fraction of training data used
- **Convergence**: Whether P2L converged within max iterations
- **Iterations**: Number of P2L iterations performed

## 🧪 Running Experiments

### Quick Test

```bash
# Test basic functionality
python test_improvements.py

# Test geometric verification
python test_geometric_verification.py

# Test high-dimensional scaling
python test_high_dim_geometric.py
```

### Custom Experiments

1. **Choose polytope type and dimension**
2. **Generate balanced dataset**
3. **Configure P2L parameters**
4. **Run comparison with geometric analysis**
5. **Analyze results and visualizations**

## 📁 Project Structure

```
polytope_testing/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── .gitignore               # Git exclusions
├── 
├── # Core Framework
├── classifier.py            # Neural network classifier and trainer
├── data_generator.py        # Dataset generation with balance control
├── polytope_generator.py    # Polytope generation (hypercubes, simplices, etc.)
├── fixed_polytope_p2l.py    # P2L implementation with improved convergence
├── geometric_metrics.py     # Geometric fidelity evaluation
├── 
├── # Experiment Framework
├── comprehensive_experiment.py  # Complete experiment pipeline
├── 
└── # Documentation
    └── README_FORME.md      # Detailed research documentation
```

## 🔬 Research Applications

This framework is designed for:

- **P2L Algorithm Evaluation**: Compare P2L against standard training
- **Safety-Critical Systems**: Evaluate violation rates and geometric fidelity
- **High-Dimensional Analysis**: Handle curse of dimensionality automatically
- **Geometric Learning**: Understand how well models learn polytope structure
- **Sample Efficiency**: Analyze P2L's ability to learn with fewer samples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{p2l_polytope_framework,
  title={P2L Research Framework: Pick-to-Learn for Polytope Classification},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 🆘 Troubleshooting

### Common Issues

1. **JAX Installation**: Make sure you have the correct JAX version for your platform
2. **Memory Issues**: Reduce sample sizes or batch sizes for high-dimensional experiments
3. **Convergence**: Adjust `convergence_param` for different polytope types
4. **Geometric Metrics**: Use `auto_scale_samples=True` for high-dimensional analysis

### Getting Help

- Check the test files for usage examples
- Review the detailed documentation in `README_FORME.md`
- Open an issue for bugs or feature requests

---

**Happy Researching! 🚀**
