# Import necessary packages
import os
import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from torchvision.datasets import ImageFolder


# Smoothing the labels
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        n_classes = predictions.size(1)
        log_probs = F.log_softmax(predictions, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(predictions)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# Architectures
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_googlenet(num_classes=10):
    model = torchvision.models.googlenet(weights=None, aux_logits=False)
    model.fc = nn.Linear(1024, num_classes)
    return model

# ARCH_MAP = {'simplecnn': SimpleCNN, 'resnet18': build_resnet18, 'googlenet': build_googlenet}


# Load CIFAR-10 and Tiny ImageNet
def get_cifar10_loaders(batch_size=128, advanced_augment=True):
    if advanced_augment:
        train_transform = transforms.Compose([
            transforms.RandAugment(),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return train_loader, test_loader


class TinyImageNetLoader:
    def __init__(self, root, batch_size, advanced_augment=False, num_workers=2):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        if advanced_augment:
            self.train_transform = transforms.Compose([
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
            ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2770, 0.2691, 0.2821))
        ])

    def _make_dataloader(self, folder, transform, shuffle=True):
        dataset = ImageFolder(folder, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )

    def get_data_loaders(self):
        train_folder = os.path.join(self.root, 'train')
        val_folder = os.path.join(self.root, 'val')
        train_loader = self._make_dataloader(train_folder, self.train_transform, shuffle=True)
        val_loader = self._make_dataloader(val_folder, self.val_transform, shuffle=False)
        return train_loader, val_loader

# Training and Evaluating Functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

#Plots to showcase performance 
def plot_metrics(epochs, train_losses, val_losses, val_accuracies):
    #Loss Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    
    #Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs. Epoch")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['simplecnn', 'resnet18', 'googlenet'],
                        help='Network architecture')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer: adam or sgd')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0 = off)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for SimpleCNN')
    parser.add_argument('--use_cosine_scheduler', action='store_true',
                        help='Use CosineAnnealingLR if set')
    parser.add_argument('--advanced_augment', action='store_true',
                        help='Use RandAugment if set')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 regularization)')
    args, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    # train_loader, test_loader = get_cifar10_loaders(
    #     batch_size=args.batch_size,
    #     advanced_augment=args.advanced_augment
    # )

    tiny_loader = TinyImageNetLoader(
        root='./tiny-imagenet-200',
        batch_size=args.batch_size,
        advanced_augment=args.advanced_augment
    )
    train_loader, test_loader = tiny_loader.get_data_loaders()
    # Models
    if args.arch == 'simplecnn':
        model = SimpleCNN(num_classes=10, dropout_rate=args.dropout_rate)
    elif args.arch == 'googlenet':
        model = build_googlenet(num_classes=10)
    else:
        model = build_resnet18(num_classes=10)
    model.to(device)

    # Loss Function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    
    # if args.optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


    # Cosine Annealing Learning Rate scheduling
    if args.use_cosine_scheduler:
        lrs = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        lrs = None

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    epoch_list = []

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        if lrs:
            lrs.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epoch_list.append(epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    plot_metrics(epoch_list, train_losses, val_losses, val_accuracies)

    results = {
        'epoch_list': epoch_list,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'model_state_dict': model.state_dict()  
    }
    with open("training_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Training results saved to training_results.pkl")


if __name__ == '__main__':
    main()







