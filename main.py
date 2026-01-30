import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

from DataOps import DataManager
from NeuralNet import MNISTModel
from multiprocessing import Process
import keyboard

def main():
    net = MNISTModel()
    net.load_state_dict(torch.load("./model/best.pth"))
    manager = DataManager()
    manager.load_mnist()
    # manager.show_image()
    train_results, test_results = manager.train(net, epochs=10, batch_size=100, lr = 0.0001)
    manager.plot_training_results(train_results, test_results)

if __name__ == "__main__":
    main()