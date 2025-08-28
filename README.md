# P2L Research Framework: Pick-to-Learn for Polytope Classification

A comprehensive research framework for evaluating Pick-to-Learn (P2L) algorithms in polytope classification tasks with geometric fidelity analysis.

## Overview

This framework provides a complete pipeline for comparing P2L algorithms against standard training methods in polytope classification tasks. It includes:

- **Polytope Generation**: Various polytope types (hypercubes, simplices, random halfspaces)
- **Data Generation**: Balanced datasets with controlled class distributions
- **P2L Implementation**: Complete Pick-to-Learn algorithm with support set selection
- **Comprehensive Evaluation**: Accuracy, F1 scores, violation rates, and geometric fidelity metrics
- **Geometric Verification**: Monte Carlo IoU and volume-based analysis
- **Size-Matched Baselines**: Fair comparison with random subset baselines

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (required for JAX acceleration)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for high-dimensional experiments
- **Storage**: 2GB+ free space for experiments and results

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: Version 12.0 or higher (required for JAX GPU acceleration)
- **Operating System**: Linux (recommended), Windows with WSL, or macOS

## Installation

### 1. Install CUDA 12.x
First, install NVIDIA CUDA Toolkit 12.x from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
```bash
pip install -U "jax[cuda12]"
```

### 2. Clone the Repository
```bash
git clone <your-repo-url>
```

### 3. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify JAX GPU Installation
```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"GPU available: {jax.devices('gpu')}")
```


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

## Project Structure

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
└── 

```
