import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
import itertools

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
        self.fc1 = nn.Linear(8192, 256)
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

# Load CIFAR-10 
def get_cifar10_loaders(batch_size=128, advanced_augment=True, device='cpu'):
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
        root='./data', train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    # Adjust num_workers and pin_memory based on device
    num_workers = 4 if device.type == 'cuda' else 2
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader

# Training and Evaluating Functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        non_blocking = device.type == 'cuda'
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        
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
            non_blocking = device.type == 'cuda'
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Plots to showcase performance 
def plot_metrics(results, output_dir="results"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique experiment configurations for grouping
    configs = pd.DataFrame(results)
    
    # Group by architecture
    plt.figure(figsize=(15, 10))
    
    # Plot Best Validation Accuracy by Architecture
    plt.subplot(2, 2, 1)
    arch_data = configs.groupby('architecture')['best_val_acc'].max().reset_index()
    plt.bar(arch_data['architecture'], arch_data['best_val_acc'])
    plt.xlabel('Architecture')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Best Validation Accuracy by Architecture')
    
    # Plot Best Validation Accuracy by Optimizer
    plt.subplot(2, 2, 2)
    opt_data = configs.groupby('optimizer')['best_val_acc'].max().reset_index()
    plt.bar(opt_data['optimizer'], opt_data['best_val_acc'])
    plt.xlabel('Optimizer')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Best Validation Accuracy by Optimizer')
    
    # Plot Best Validation Accuracy by Batch Size
    plt.subplot(2, 2, 3)
    batch_data = configs.groupby('batch_size')['best_val_acc'].max().reset_index()
    plt.bar(batch_data['batch_size'].astype(str), batch_data['best_val_acc'])
    plt.xlabel('Batch Size')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Best Validation Accuracy by Batch Size')
    
    # Plot Best Validation Accuracy by Learning Rate
    plt.subplot(2, 2, 4)
    lr_data = configs.groupby('learning_rate')['best_val_acc'].max().reset_index()
    plt.bar(lr_data['learning_rate'].astype(str), lr_data['best_val_acc'])
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Best Validation Accuracy by Learning Rate')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_metrics.png")
    
    # Plot learning curves for top configurations
    top_configs = configs.sort_values('best_val_acc', ascending=False).head(6)
    plot_top_learning_curves(results, top_configs, output_dir)
    
    # Create detailed comparison plots
    plot_architecture_optimizer_comparison(results, output_dir)
    plot_lr_comparison(results, output_dir)
    plot_batch_size_comparison(results, output_dir)
    plot_augmentation_comparison(results, output_dir)
    plot_scheduler_comparison(results, output_dir)
    
    return

def plot_top_learning_curves(results, top_configs, output_dir):
    plt.figure(figsize=(20, 15))
    
    for i, (idx, config) in enumerate(top_configs.iterrows()):
        plt.subplot(3, 2, i+1)
        
        # Get the result data
        result = results[idx]
        epochs = list(range(1, len(result['train_losses']) + 1))
        
        # Plot training and validation loss
        plt.plot(epochs, result['train_losses'], label='Train Loss')
        plt.plot(epochs, result['val_losses'], label='Val Loss')
        plt.plot(epochs, [acc/100 for acc in result['val_accuracies']], label='Val Acc/100')
        
        # Set title with configuration details
        title = f"{config['architecture']}, {config['optimizer']}, lr={config['learning_rate']}\n"
        title += f"batch={config['batch_size']}, dropout={config['dropout_rate']}, acc={config['best_val_acc']:.2f}%"
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_learning_curves.png")

def plot_architecture_optimizer_comparison(results, output_dir):
    # Extract data for comparison
    data = []
    for i, result in enumerate(results):
        data.append({
            'architecture': result['architecture'],
            'optimizer': result['optimizer'],
            'best_val_acc': result['best_val_acc']
        })
    
    df = pd.DataFrame(data)
    
    # Create a pivot table for architecture vs optimizer
    pivot = df.pivot_table(
        index='architecture', 
        columns='optimizer', 
        values='best_val_acc', 
        aggfunc='max'
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar')
    plt.xlabel('Architecture')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Architecture vs Optimizer Comparison')
    plt.legend(title='Optimizer')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/architecture_optimizer_comparison.png")

def plot_lr_comparison(results, output_dir):
    # Filter data by architecture
    archs = set(result['architecture'] for result in results)
    
    fig, axes = plt.subplots(len(archs), 1, figsize=(12, 6*len(archs)))
    
    for i, arch in enumerate(sorted(archs)):
        # Extract data for this architecture
        arch_data = []
        for result in results:
            if result['architecture'] == arch:
                arch_data.append({
                    'lr': result['learning_rate'],
                    'optimizer': result['optimizer'],
                    'best_val_acc': result['best_val_acc']
                })
        
        df = pd.DataFrame(arch_data)
        
        # Create pivot table
        pivot = df.pivot_table(
            index='lr', 
            columns='optimizer', 
            values='best_val_acc', 
            aggfunc='max'
        )
        
        # Plot
        ax = axes[i] if len(archs) > 1 else axes
        pivot.plot(kind='bar', ax=ax)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Best Validation Accuracy (%)')
        ax.set_title(f'Learning Rate Comparison for {arch}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Optimizer')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_rate_comparison.png")

def plot_batch_size_comparison(results, output_dir):
    # Extract data
    data = []
    for result in results:
        data.append({
            'architecture': result['architecture'],
            'batch_size': result['batch_size'],
            'best_val_acc': result['best_val_acc']
        })
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='batch_size', 
        columns='architecture', 
        values='best_val_acc', 
        aggfunc='max'
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar')
    plt.xlabel('Batch Size')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Batch Size Comparison')
    plt.legend(title='Architecture')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/batch_size_comparison.png")

def plot_augmentation_comparison(results, output_dir):
    # Extract data
    data = []
    for result in results:
        if 'advanced_augment' in result:
            data.append({
                'architecture': result['architecture'],
                'augmentation': 'Advanced' if result['advanced_augment'] else 'Standard',
                'best_val_acc': result['best_val_acc']
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='architecture', 
        columns='augmentation', 
        values='best_val_acc', 
        aggfunc='max'
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar')
    plt.xlabel('Architecture')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Data Augmentation Comparison')
    plt.legend(title='Augmentation Type')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_comparison.png")

def plot_scheduler_comparison(results, output_dir):
    # Extract data
    data = []
    for result in results:
        if 'use_cosine_scheduler' in result:
            data.append({
                'architecture': result['architecture'],
                'scheduler': 'Cosine' if result['use_cosine_scheduler'] else 'None',
                'best_val_acc': result['best_val_acc']
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='architecture', 
        columns='scheduler', 
        values='best_val_acc', 
        aggfunc='max'
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar')
    plt.xlabel('Architecture')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Learning Rate Scheduler Comparison')
    plt.legend(title='Scheduler Type')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scheduler_comparison.png")

def create_configuration_grid():
    """Create a grid of hyperparameter configurations to test"""
    configurations = []
    
    # Define hyperparameter options
    architectures = ['simplecnn', 'resnet18']
    optimizers = ['adam', 'sgd']
    learning_rates = [0.001, 0.01]
    batch_sizes = [64, 128, 256]
    dropout_rates = [0.3, 0.5]
    label_smoothing = [0.0, 0.1]
    use_scheduler = [True, False]
    augmentation = [True, False]
    
    # Create configurations
    base_configs = list(itertools.product(
        architectures,
        optimizers,
        learning_rates,
    ))
    
    # Generate configurations
    for arch, opt, lr in base_configs:
        # Base configuration
        config = {
            'architecture': arch,
            'optimizer': opt,
            'learning_rate': lr,
            'batch_size': 128,
            'dropout_rate': 0.3 if arch == 'simplecnn' else 0.0,
            'label_smoothing': 0.0,
            'use_cosine_scheduler': False,
            'advanced_augment': False,
            'weight_decay': 1e-4,
            'epochs': 15,
        }
        configurations.append(config)
        
        # Add batch size variations
        for bs in batch_sizes:
            if bs != 128:
                config_bs = config.copy()
                config_bs['batch_size'] = bs
                configurations.append(config_bs)
        
        # Add dropout rate variation for SimpleCNN
        if arch == 'simplecnn':
            for dr in dropout_rates:
                if dr != 0.3:
                    config_dr = config.copy()
                    config_dr['dropout_rate'] = dr
                    configurations.append(config_dr)
        
        # Add label smoothing option
        config_ls = config.copy()
        config_ls['label_smoothing'] = 0.1
        configurations.append(config_ls)
        
        # Add scheduler option
        config_sch = config.copy()
        config_sch['use_cosine_scheduler'] = True
        configurations.append(config_sch)
        
        # Add augmentation option
        config_aug = config.copy()
        config_aug['advanced_augment'] = True
        configurations.append(config_aug)
    
    return configurations

def run_experiment(config, device):
    """Run a single experiment with the given configuration"""
    print(f"\n{'='*80}")
    print(f"Running experiment with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")
    
    # Setup data loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'],
        advanced_augment=config.get('advanced_augment', False),
        device=device
    )
    
    # Setup model
    if config['architecture'] == 'simplecnn':
        model = SimpleCNN(num_classes=10, dropout_rate=config['dropout_rate'])
    else:
        model = build_resnet18(num_classes=10)
    model.to(device)
    
    # Setup loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing']) if config['label_smoothing'] > 0 else nn.CrossEntropyLoss()
    
    # Setup optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    
    # Setup scheduler
    lrs = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']) if config.get('use_cosine_scheduler', False) else None
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        if lrs:
            lrs.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")
    
    # Save training metrics in the config
    result = config.copy()
    result.update({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
    })

    return result

def main():
    output_dir = f"experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse command line arguments for global settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_batch_multiplier', type=int, default=2,
                      help='Multiplier for batch size when using GPU')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # GPU optimizations
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Create configuration grid
    all_configurations = create_configuration_grid()
    
    # Print summary of experiments to run
    print(f"Generated {len(all_configurations)} configurations to test")
    
    # Run all experiments
    results = []
    for i, config in enumerate(all_configurations):
        print(f"\nExperiment {i+1}/{len(all_configurations)}")
        
        # Adjust batch size for GPU if needed
        if device.type == 'cuda':
            config['batch_size'] *= args.gpu_batch_multiplier
            print(f"Adjusted batch size for GPU: {config['batch_size']}")
        
        # Run experiment
        result = run_experiment(config, device)
        results.append(result)
        
        # Save incremental results
        with open(f"{output_dir}/results_incremental.pkl", "wb") as f:
            pickle.dump(results, f)
    
    # Generate summary plots
    plot_metrics(results, output_dir)
    
    # Generate CSV summary
    summary_data = []
    for result in results:
        summary = {
            'architecture': result['architecture'],
            'optimizer': result['optimizer'],
            'learning_rate': result['learning_rate'],
            'batch_size': result['batch_size'],
            'dropout_rate': result.get('dropout_rate', 'N/A'),
            'label_smoothing': result['label_smoothing'],
            'use_scheduler': result.get('use_cosine_scheduler', False),
            'advanced_augment': result.get('advanced_augment', False),
            'final_train_loss': result['train_losses'][-1],
            'final_val_loss': result['val_losses'][-1],
            'final_val_acc': result['val_accuracies'][-1],
            'best_val_acc': result['best_val_acc'],
        }
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/experiment_summary.csv", index=False)
    
    # Find best configuration
    best_idx = np.argmax([r['best_val_acc'] for r in results])
    best_config = results[best_idx]
    
    print("\n" + "="*80)
    print("Best configuration:")
    for k, v in best_config.items():
        if k not in ['train_losses', 'val_losses', 'val_accuracies']:
            print(f"  {k}: {v}")
    print(f"Best validation accuracy: {best_config['best_val_acc']:.2f}%")
    print("="*80 + "\n")
    
    # Save final results
    with open(f"{output_dir}/all_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"All results saved to {output_dir}/")
    print("Training complete!")

if __name__ == '__main__':
    main()