import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import numpy as np
from time import time

class SUN397Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.X = []
        self.y = []
        self.labels = []
        mean, std = calculate_mean_std()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transform

        for folder in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, folder)):
                for subfolder in os.listdir(os.path.join(self.data_dir, folder)):
                    label = f"{folder}/{subfolder}"
                    for img in os.listdir(os.path.join(self.data_dir, folder, subfolder)):
                        if img.lower().endswith(".jpg"):
                            full_path = os.path.join(self.data_dir, folder, subfolder, img)
                            self.X.append(full_path)
                            self.labels.append(label)
        classes, idx = np.unique(self.labels, return_inverse= True)
        self.y = torch.tensor(idx, dtype= torch.long)
        self.classes = classes
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        full_path = self.X[idx]
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (image, self.y[idx])
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
    
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2)) 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.classifier(x)
    
    
def calculate_mean_std(**kwargs):
    return [0.5127, 0.4529, 0.3974], [0.2496, 0.2534, 0.2615]


def train(model, train_loader, **kwargs):
    device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    val_loader = kwargs.get('val_loader', None)
    num_epochs = kwargs.get('num_epochs', 30) 
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"DEBUG: Current Working Directory is {os.getcwd()}")
    print(f"DEBUG: Directory contents: {os.listdir('.')}")

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        current_acc = test(model, val_loader if val_loader else train_loader, device=device)
        print(f"Epoch {epoch+1} Acc: {current_acc:.2f}%")

        if current_acc >= best_acc:
            best_acc = current_acc
            
            paths_to_try = [
                'model.pt',                    
                'submission/model.pt',         
                './submission/model.pt',       
                '../submission/model.pt'       
            ]

            for p in paths_to_try:
                try:
                    d = os.path.dirname(p)
                    if d and not os.path.exists(d):
                        os.makedirs(d, exist_ok=True)
                    
                    torch.save(model.state_dict(), p)
                    print(f"SUCCESS: Saved model to {os.path.abspath(p)}")
                except Exception as e:
                    print(f"FAILED: Could not save to {p}. Error: {e}")

    print("Training Complete.")

def test(model, test_loader, **kwargs):
    device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, 
                        default='welcome/to/CNN/homework',
                        help='Path to training data directory')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
 
    return parser.parse_args()
    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = SUN397Dataset(data_dir=args.train_dir)
    num_classes = len(np.unique(full_dataset.labels))
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = CNN(num_classes=num_classes).to(device)

    train(
        model, 
        train_loader, 
        val_loader=val_loader, 
        device=device, 
        num_epochs=30
    )

if __name__ == "__main__":
    main()
