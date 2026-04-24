"""
Train and evaluate a regularized neural network on MNIST.

Example:
    python src/train.py --model batch_norm --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import os

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from regularization_models import MODEL_BUILDERS


def load_mnist_data():
    """Load and preprocess MNIST for dense neural networks."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_BUILDERS.keys(), default="batch_norm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--output", default="results/run_result.csv")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_data()

    model = MODEL_BUILDERS[args.model]()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)

    with open(args.output, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["model", "epochs", "test_loss", "test_accuracy"])
        writer.writerow([args.model, args.epochs, test_loss, test_accuracy])

    print(f"Model: {args.model}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
