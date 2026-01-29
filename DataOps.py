import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import MNISTModel
from torch import nn
from tqdm import tqdm


class DataManager:

    def load_mnist(self, root: str = "data") -> None:
        self.train_data = datasets.MNIST(
            root="data",
            download=True,
            train=True,
            transform=ToTensor(),
        )
        self.test_data = datasets.MNIST(
            root="data",
            download=True,
            train=False,
            transform=ToTensor(),
        )
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.select_criterion(nn.BCEWithLogitsLoss)
    
    def show_image(self) -> None:
        plt.imshow(self.test_data.data[0], cmap="gray")
        plt.title(f"Example Image of {self.test_data.targets[0].item()} in the MNIST Dataset")
        plt.show()

    def fetch_example(self) -> tuple[torch.FloatTensor, int]:
        img, label = self.test_data[0]
        return (img, label)
    
    def select_criterion(self, criterion):
        self.criterion = criterion()

    def train(self, model: MNISTModel, epochs = 10, lr = 0.01, batch_size = 32):
        best_test_loss = np.inf
        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size)
        optimiser = optim.SGD(model.parameters(), lr = lr)

        train_results = []
        test_results = []

        print("b")

        for epoch in (pbar_epoch := tqdm(range(epochs))):
            pbar_epoch.set_description(f"Processing epoch {epoch + 1}...")
            for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(train_loader, leave=False))
            ):
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = model(images)
                train_loss = self.criterion(outputs, one_hot_targets.float())
                batch_loss = train_loss.item()
                pbar.desc = f"    Processing Training Batch {batch_id} | Batch Loss = {batch_loss}"

                optimiser.zero_grad()
                train_loss.backward()
                optimiser.step()

                batch_number = batch_id + epoch * len(train_loader)
                train_results.append((batch_number, batch_loss))                

            test_loss, test_accuracy = self.test(model=model)
            test_results.append(((epoch + 1) * len(train_loader), test_loss, test_accuracy))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), "./model/best.pth")
            torch.save(model.state_dict(), "./model/latest.pth")

        return (train_results, test_results)

    def test(self, model: MNISTModel, batch_size = 32):
        model.eval()
        test_loader = DataLoader(dataset=self.test_data, batch_size=batch_size)
        total_tests = len(self.test_data)
        test_accuracy = 0
        test_loss = 0
        for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(test_loader, leave=False))
        ):
            with torch.no_grad():
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = model(images)
                predictions = outputs.argmax(1)
                batch_loss = self.criterion(outputs, one_hot_targets.float()).item()
                test_loss += batch_loss / len(test_loader)
                for i in range(targets.size(0)):
                    if targets[i] == predictions[i]:
                        test_accuracy += 1
            pbar.desc = f"Processing Test Batch {batch_id} | Batch Loss = {batch_loss}"
        test_accuracy /= total_tests
        return test_loss, test_accuracy
            
    @staticmethod
    def plot_training_results(train_results, test_results):
        
        fig, ax1 = plt.subplots()
        x_train, y_train = zip(*train_results)
        x_test, y_test, acc_test = zip(*test_results)
        ax1.set_xlabel("Batch Number")
        ax1.set_ylabel("Training Loss", color="red")
        ax1.plot(x_train, y_train, color="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Test Loss", color="blue")
        ax2.plot(x_test, y_test, color="blue")

        plt.title("Training and Test Loss for MNIST Model Training")
        fig.tight_layout()
        plt.show()

        plt.ylabel("Test Accuracy")
        plt.xlabel("Batch Number")
        plt.plot(x_test, acc_test, color="orange")
        plt.show()





            
