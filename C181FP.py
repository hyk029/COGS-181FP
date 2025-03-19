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
from torchvision.datasets import ImageFolder

class CNN(nn.Module):

    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
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


ARCH_MAP = {'cnn': CNN, 'resnet18': build_resnet18}

class CIFAR10Loader:
    def __init__(self, batch_size, advanced_augment=False, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        if advanced_augment:
            self.train_transform = transforms.Compose([
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))
            ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ])

    def get_data_loaders(self):
        train_set = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.train_transform
        )
        val_set = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader


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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tinyimagenet'],
                        help='Which dataset to train on.')
    parser.add_argument('--data_root', type=str, default='./tiny-imagenet-200',
                        help='Root path for Tiny ImageNet. Ignored for CIFAR-10.')
    parser.add_argument('--arch', type=str, default='simplecnn',
                        choices=['simplecnn', 'resnet18'],
                        help='Model architecture.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Which optimizer to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate for SimpleCNN.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing parameter (0.0 means no smoothing).')

    parser.add_argument('--advanced_augment', action='store_true',
                        help='Use RandAugment or standard transforms.')
    parser.add_argument('--use_cosine_scheduler', action='store_true',
                        help='Use a Cosine Annealing LR scheduler.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.dataset == 'cifar10':
        loader_builder = CIFAR10Loader(
            batch_size=args.batch_size,
            advanced_augment=args.advanced_augment
        )
        train_loader, val_loader = loader_builder.get_data_loaders()
        num_classes = 10
    else:
        loader_builder = TinyImageNetLoader(
            root=args.data_root,
            batch_size=args.batch_size,
            advanced_augment=args.advanced_augment
        )
        train_loader, val_loader = loader_builder.get_data_loaders()
        num_classes = 200

    if args.arch == 'cnn':
        model = CNN(num_classes=num_classes, dropout_rate=args.dropout_rate)
    else:
        model = build_resnet18(num_classes=num_classes)
    model.to(device)

    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.2f}%  "
            f"(Best: {best_val_acc:.2f}%)")

    print("Training Complete")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()

