import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import sys
sys.path.append("/app")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
from shared.model import get_model  # Assuming get_model is defined in shared/model.py
# Add imports for metrics
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
print(sys.path)

# Load Chest X-ray dataset and split per client
def load_data(client_id):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    data_path = "/data/chest_xray/train"  # All clients read same dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    # Split dataset among 3 clients (non-overlapping, same size)
    total_len = len(dataset)
    client_len = total_len // 3
    start = (client_id - 1) * client_len
    end = start + client_len if client_id < 3 else total_len
    client_indices = list(range(start, end))
    client_dataset = Subset(dataset, client_indices)
    train_size = int(0.8 * len(client_dataset))
    test_size = len(client_dataset) - train_size
    trainset, testset = random_split(client_dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader

# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # You can increase epochs later
            for inputs, labels in self.trainloader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in self.testloader:
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = correct / len(self.testloader.dataset)
        precision = precision_score(all_labels, all_preds, pos_label=1)
        recall = recall_score(all_labels, all_preds, pos_label=1)
        f1 = f1_score(all_labels, all_preds, pos_label=1)
        return (
            float(loss),
            len(self.testloader.dataset),
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            },
        )

# Entry point
if __name__ == "__main__":
    import time
    print("Waiting for server to start...")
    time.sleep(5)  # Add a delay to ensure server is up
    
    client_id = int(os.environ.get("CLIENT_ID", "1"))
    print(f"Starting client {client_id}")
    
    model = get_model()
    trainloader, testloader = load_data(client_id)
    client = FlowerClient(model, trainloader, testloader)
    
    # Use the updated API method with correct server address
    print(f"Connecting to server at fl_server:8080")
    try:
        fl.client.start_client(
            server_address="fl_server:8080",
            client=client.to_client()
        )
    except Exception as e:
        print(f"Error connecting to server: {e}")