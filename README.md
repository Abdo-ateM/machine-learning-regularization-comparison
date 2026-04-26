# Comparative Analysis of Regularization Techniques in Deep Neural Networks

This repository contains a research-style deep learning project comparing common regularization techniques in neural networks:

- L1 Regularization
- L2 Regularization
- Dropout
- Batch Normalization
- Elastic Net (L1 + L2)

The experiments use the MNIST handwritten digits dataset with a fixed dense neural network architecture to compare training behavior, validation performance, and final test accuracy.

## Project Motivation

Deep neural networks can easily overfit because they contain many trainable parameters. Regularization techniques help improve generalization, training stability, and robustness. This project compares the impact of different regularization methods under a unified experimental setup.

## Repository Structure

```text
deep-learning-regularization-comparison/
├── assets/                     # Extracted training/validation plots
├── notebooks/                  # Main experiment notebook
├── paper/                      # Research paper PDF
├── results/                    # Summary CSV of reported results
├── src/                        # Reusable model/training scripts
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The project uses the MNIST dataset from TensorFlow/Keras.

- Training samples: 60,000
- Test samples: 10,000
- Input shape after preprocessing: 784 features
- Classes: 10 handwritten digit classes

## Methods Compared

| Method | Main Idea |
|---|---|
| L1 Regularization | Adds absolute-weight penalty to encourage sparse weights |
| L2 Regularization | Adds squared-weight penalty to discourage large weights |
| Dropout | Randomly disables neurons during training to reduce co-adaptation |
| Batch Normalization | Normalizes intermediate activations to stabilize training |
| Elastic Net | Combines L1 and L2 penalties |

## Reported Results

| Model | Test Accuracy |
|---|---:|
| L1 Regularization | 97.89% |
| L2 Regularization | 97.90% |
| Dropout | 98.17% |
| Batch Normalization | 98.27% |
| Elastic Net (L1 + L2) | 98.22% |

In this experiment, Batch Normalization achieved the highest test accuracy, followed closely by Elastic Net and Dropout.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Abdo-ateM/deep-learning-regularization-comparison.git
cd deep-learning-regularization-comparison
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/Regularization_Techniques_MNIST_Comparison.ipynb
```

### 5. Run a model from script

```bash
python src/train.py --model batch_norm --epochs 20
```

Available model names:

```text
l1
l2
dropout
batch_norm
elastic_net
```

## Key Takeaways

- L1 regularization is useful when sparsity is desired.
- L2 regularization provides stable convergence and controls large weights.
- Dropout reduces overfitting by forcing the network to learn more robust representations.
- Batch Normalization improves training stability and produced the strongest result in this experiment.
- Combining methods can be useful, but no single technique is universally best.

## Authors

- Abdelrahman Hatem

## Project Type

Research paper + implementation notebook + reusable training scripts.

## Keywords

Deep Learning, Neural Networks, Regularization, L1, L2, Dropout, Batch Normalization, Elastic Net, TensorFlow, MNIST
