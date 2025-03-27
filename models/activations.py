import tensorflow as tf
import torch.nn.functional
# from models.generative.utils import power_iteration_method

def leakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLU(x):
    relu = torch.nn.ReLU()
    return relu(x)

def tanh(x):
    return tf.nn.tanh(x)

def sigmoid(x):
    return tf.sigmoid(x)