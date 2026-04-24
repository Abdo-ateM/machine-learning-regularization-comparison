"""
Model builders for the MNIST regularization comparison project.

The project compares common regularization techniques using a fixed
fully connected neural network architecture.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_l1_model(input_dim: int = 784, num_classes: int = 10, l1_value: float = 1e-4) -> tf.keras.Model:
    """Build a dense neural network with L1 regularization."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l1(l1_value)),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l1(l1_value)),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="l1_regularized_model",
    )


def build_l2_model(input_dim: int = 784, num_classes: int = 10, l2_value: float = 1e-4) -> tf.keras.Model:
    """Build a dense neural network with L2 regularization."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_value)),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_value)),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="l2_regularized_model",
    )


def build_dropout_model(input_dim: int = 784, num_classes: int = 10, dropout_rate: float = 0.3) -> tf.keras.Model:
    """Build a dense neural network with Dropout."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="dropout_model",
    )


def build_batch_norm_model(input_dim: int = 784, num_classes: int = 10) -> tf.keras.Model:
    """Build a dense neural network with Batch Normalization."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="batch_normalization_model",
    )


def build_elastic_net_model(
    input_dim: int = 784,
    num_classes: int = 10,
    l1_value: float = 1e-5,
    l2_value: float = 1e-4,
) -> tf.keras.Model:
    """Build a dense neural network with combined L1 and L2 regularization."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value)),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value)),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="elastic_net_model",
    )


MODEL_BUILDERS = {
    "l1": build_l1_model,
    "l2": build_l2_model,
    "dropout": build_dropout_model,
    "batch_norm": build_batch_norm_model,
    "elastic_net": build_elastic_net_model,
}
