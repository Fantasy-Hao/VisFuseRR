import numpy as np

IMAGE_WEIGHT = 0.6


def sigmoid_zscore(scores):
    values = 1 / (1 + np.exp(-np.asarray(scores, dtype=np.float64)))
    std = values.std()
    return (values - values.mean()) / std if std >= 1e-12 else np.zeros_like(values)


def fusion_inputs(text, image, method):
    if method == "sigmoid-zscore":
        return sigmoid_zscore(text), sigmoid_zscore(image)
    return text, image


def fuse(text, image, alpha=IMAGE_WEIGHT):
    return (1 - alpha) * text + alpha * image
