import torch
import logging

# Provides one-hot encoding based on Ookla data types
def to_one_hot(id, num_classes):
    return torch.nn.functional.one_hot(5)