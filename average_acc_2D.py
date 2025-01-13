import argparse
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from torchvision.transforms import (
    InterpolationMode,
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    Normalize,
    Resize,
    CenterCrop,
)

from replace_conv_layers import replace_conv_layers  # Ensure this module is available

# -------------------------------
# Configuration
# -------------------------------
class Config:
    SEED = 42
    FLOAT32_MATMUL_PRECISION = 'high'
    WEIGHTS_PATH = {
        'radimagenet': "./_weights/R18.pth",
        'crossdconv': "./_weights/CDConvR18.pth"
    }
    DEFAULTS = {
        'data_flag': 'breastmnist_224',
        'num_epochs': 60,
        'batch_size': 8,
        'conv': 'crossdconv',
    }
    FINETUNING_STAGES = [
        {'epoch': 10, 'unfreeze_layers': ['layer4', 'fc']},
        {'epoch': 20, 'unfreeze_layers': ['layer3']},
        {'epoch': 30, 'unfreeze_layers': ['layer2']},
        {'epoch': 40, 'unfreeze_layers': ['layer1']},
        {'epoch': 50, 'unfreeze_layers': ['conv1', 'bn1']},
        # Add more stages as needed
    ]

# -------------------------------
# 1. SEEDING
# -------------------------------
def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)
torch.set_float32_matmul_precision(Config.FLOAT32_MATMUL_PRECISION)

# -------------------------------
# 3. MODEL CREATION
# -------------------------------
def create_model(n_classes, conv_opt=False, weights_path=None):
    """
    Creates a ResNet-18 model with optional convolution layer replacements.
    
    Args:
        n_classes (int): Number of output classes.
        conv_opt (str or bool): Option for convolution layer modification.
        weights_path (str): Path to the weights file.
        
    Returns:
        model (nn.Module): The modified ResNet-18 model.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)

    if conv_opt in Config.WEIGHTS_PATH:
        if conv_opt == 'crossdconv':
            replace_conv_layers(model)
        try:
            ckpt = torch.load(weights_path or Config.WEIGHTS_PATH[conv_opt], map_location='cpu')
            state_dict = ckpt.get("model", ckpt)
            weights = {
                key.replace("_orig_mod.", ""): value 
                for key, value in state_dict.items()
            }
            keys_to_remove=['fc.weight', 'fc.bias']
            for key in keys_to_remove:
                weights.pop(key, None)
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            print(f"Loaded weights with missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        except FileNotFoundError:
            print(f"Weight file for '{conv_opt}' not found at {weights_path or Config.WEIGHTS_PATH[conv_opt]}.")
    elif conv_opt:
        print(f"Unknown conv_opt '{conv_opt}'. Proceeding with standard ResNet-18.")

    return model

# -------------------------------
# 4. DATASET CLASS
# -------------------------------
class ArrayDataset(Dataset):
    """
    A custom Dataset class for numpy array images and labels.
    """
    def __init__(self, images, labels, transform=None):
        """
        Initializes the dataset.
        
        Args:
            images (np.ndarray): Array of images.
            labels (np.ndarray): Array of labels.
            transform (callable, optional): Transformations to apply.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        if self.images.ndim == 3:
            # Expand to (N, 1, H, W) and repeat channels to make it (N, 3, H, W)
            self.images = np.expand_dims(self.images, 1)
            self.images = np.repeat(self.images, 3, axis=1)
        elif self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError("Images should have shape (N, C, H, W) with C=3 or (N, H, W).")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.from_numpy(image).float()
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()
        return image, label

# -------------------------------
# 5. TRANSFORMS
# -------------------------------
class BiomedicalPresetTrain:
    """
    Preset transformations for training.
    """
    def __init__(self, crop_size, mean, std, interpolation=InterpolationMode.BILINEAR, hflip_prob=0.5, color_jitter_params=None):
        transforms = [
            RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True),
            RandomHorizontalFlip(p=hflip_prob)
        ]
        if color_jitter_params:
            transforms.append(ColorJitter(**color_jitter_params))
        transforms.append(Normalize(mean=mean, std=std))
        self.transforms = Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)

class BiomedicalPresetEval:
    """
    Preset transformations for evaluation.
    """
    def __init__(self, crop_size, resize_size, mean, std, interpolation=InterpolationMode.BILINEAR):
        self.transforms = Compose([
            Resize(resize_size, interpolation=interpolation, antialias=True),
            CenterCrop(crop_size),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, img):
        return self.transforms(img)

# -------------------------------
# 5a. FINETUNING SCHEDULER
# -------------------------------
class FinetuningScheduler:
    """
    Scheduler to manage progressive finetuning by unfreezing layers at specified epochs.
    """
    def __init__(self, model, stages):
        """
        Initializes the finetuning scheduler.
        
        Args:
            model (nn.Module): The model to finetune.
            stages (list of dict): Each dict contains 'epoch' and 'unfreeze_layers' keys.
        """
        self.model = model
        self.stages = sorted(stages, key=lambda x: x['epoch'])
        self.current_stage = 0

    def step(self, epoch, optimizer):
        """
        Checks if a new stage should be activated at the current epoch and updates the optimizer.
        
        Args:
            epoch (int): Current epoch number.
            optimizer (torch.optim.Optimizer): Current optimizer.
            
        Returns:
            optimizer (torch.optim.Optimizer): Updated optimizer if layers are unfrozen.
        """
        if self.current_stage < len(self.stages) and epoch == self.stages[self.current_stage]['epoch']:
            layers_to_unfreeze = self.stages[self.current_stage]['unfreeze_layers']
            print(f"\n--- Finetuning Stage {self.current_stage + 1} ---")
            print(f"Unfreezing layers: {layers_to_unfreeze}")

            for layer_name in layers_to_unfreeze:
                layer = dict([*self.model.named_modules()])[layer_name]
                for param in layer.parameters():
                    param.requires_grad = True

            # Update optimizer to include newly unfrozen parameters
            optimizer = self.update_optimizer(optimizer)
            self.current_stage += 1

        return optimizer

    def update_optimizer(self, optimizer):
        """
        Updates the optimizer to include parameters that are now trainable.
        
        Args:
            optimizer (torch.optim.Optimizer): Current optimizer.
        
        Returns:
            optimizer (torch.optim.Optimizer): Updated optimizer.
        """
        # Collect all parameters that require gradients
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        # Create a new optimizer with trainable parameters
        optimizer = torch.optim.Adam(trainable_params, lr=optimizer.param_groups[0]['lr'])
        print("Optimizer updated to include newly trainable parameters.")
        return optimizer

# -------------------------------
# 6. TRAINING AND EVALUATION
# -------------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, data_loader, device, criterion):
    """
    Evaluates the model on the given data loader.
    
    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for evaluation.
        device (torch.device): Device to perform computations on.
        criterion (nn.Module): Loss function.
    
    Returns:
        avg_loss (float): Average loss over the dataset.
        accuracy (float): Accuracy score.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy = accuracy_score(all_targets.numpy(), all_preds.numpy())

    return avg_loss, accuracy

def train_model(conv_opt, n_classes, train_loader, val_loader, num_epochs, device, weights_path=None):
    """
    Creates and trains the model with progressive finetuning.
    
    Args:
        conv_opt (str or bool): Convolution option for model creation.
        n_classes (int): Number of output classes.
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        weights_path (str, optional): Path to custom weights file.
    
    Returns:
        model (nn.Module): Trained model with best validation accuracy.
    """
    model = create_model(n_classes, conv_opt=conv_opt, weights_path=weights_path).to(device)

    # Initially, freeze all layers except the final fully connected layer
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * num_epochs), int(0.75 * num_epochs)],
        gamma=0.1
    )

    # Initialize finetuning scheduler
    finetuning_scheduler = FinetuningScheduler(model, Config.FINETUNING_STAGES)

    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Check for finetuning stage updates
        optimizer = finetuning_scheduler.step(epoch, optimizer)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Step the scheduler
        scheduler.step()

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        # Checkpoint if best
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = deepcopy(model.state_dict())

        # Log epoch metrics
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_wts)
    return model

# -------------------------------
# 7. MAIN LOOP
# -------------------------------
def main(args):
    """
    The main function to execute the training and evaluation pipeline.
    """
    # Configuration and device setup
    data_flag = args.data_flag
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    conv = args.conv
    weights_path = args.weights_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load .npz data
    path_to_npz = os.path.join("./Enhanced-MedMNIST", f"{data_flag}.npz")
    if not os.path.isfile(path_to_npz):
        raise FileNotFoundError(f"Data file not found at {path_to_npz}")

    data = np.load(path_to_npz)
    images = data['images'] / 255.0
    labels_all = data['labels'].squeeze()

    # Ensure correct image shape
    if images.ndim == 3:
        pass  # Already handled in ArrayDataset
    elif images.ndim == 4 and images.shape[1] != 3:
        images = np.transpose(images, [0, 3, 1, 2])
    elif images.ndim != 4:
        raise ValueError("Unsupported image shape. Expected 3D or 4D array.")

    n_classes = len(np.unique(labels_all))
    print(f"Number of classes: {n_classes}")

    # Compute mean and std
    #if images.ndim == 4:
    #    mean = images.mean(axis=(0, 2, 3))
    #    std = images.std(axis=(0, 2, 3))
    #else:
    #    mean = images.mean()
    #    std = images.std()
    
    if args.conv == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.3162, 0.3162, 0.3162]
        std = [0.3213, 0.3213, 0.3213]
    print(f"Computed mean: {mean}")
    print(f"Computed std: {std}")

    # Define image sizes
    crop_size = 224
    resize_size = 256

    # Initialize transformation presets
    train_transform = BiomedicalPresetTrain(
        crop_size=crop_size,
        mean=mean,
        std=std,
        hflip_prob=0.5,
        color_jitter_params={'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}
    )

    eval_transform = BiomedicalPresetEval(
        crop_size=crop_size,
        resize_size=resize_size,
        mean=mean,
        std=std
    )

    # Metrics storage
    best_val_metrics = []
    best_test_metrics = []

    # Perform Trials
    for trial in range(1, 4):
        print(f"\n=== Trial {trial} ===")
        # Set a different seed for each trial
        set_seed(Config.SEED + trial)

        # 5-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED + trial)
        for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):
            print(f"\n--- Fold {fold} ---")

            train_images, test_images = images[train_idx], images[test_idx]
            train_labels_, test_labels_ = labels_all[train_idx], labels_all[test_idx]

            # 80/20 split for training and validation
            val_split = int(0.2 * len(train_images))
            val_images, val_labels_ = train_images[:val_split], train_labels_[:val_split]
            train_images_split, train_labels_split = train_images[val_split:], train_labels_[val_split:]

            print(f"Train size: {train_images_split.shape[0]}, Val size: {val_images.shape[0]}, Test size: {test_images.shape[0]}")

            # Create datasets
            train_dataset = ArrayDataset(
                images=train_images_split,
                labels=train_labels_split,
                transform=train_transform
            )
            val_dataset = ArrayDataset(
                images=val_images,
                labels=val_labels_,
                transform=eval_transform
            )
            test_dataset = ArrayDataset(
                images=test_images,
                labels=test_labels_,
                transform=eval_transform
            )

            # Create DataLoaders with pin_memory for faster GPU transfers
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # Inspect the first batch
            inspect_batch(train_loader)

            # Train the model
            model = train_model(conv_opt=conv, 
                                n_classes=n_classes, 
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                num_epochs=num_epochs, 
                                device=device,
                                weights_path=weights_path)

            # Evaluate on validation and test sets
            criterion = nn.CrossEntropyLoss()
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)

            print(f"Fold {fold} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Fold {fold} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            best_val_metrics.append(val_acc)
            best_test_metrics.append(test_acc)

    # Summarize results
    mean_test_acc = np.mean(best_test_metrics)
    std_test_acc = np.std(best_test_metrics)

    print("\n#==================== Final Results ====================")
    print(f"# Test for {data_flag}")
    print(f"# Mean Test Accuracy: {mean_test_acc:.4f}")
    print(f"# Std Test Accuracy:  {std_test_acc:.4f}\n")

def inspect_batch(data_loader):
    """
    Inspects and prints information about the first batch in the DataLoader.
    """
    try:
        data_iter = iter(data_loader)
        features, labels = next(data_iter)
    except StopIteration:
        print("DataLoader is empty.")
        return

    print(f'Features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    if features.ndim == 4:
        print(f'Feature batch size: {features.size(0)}')
        print(f'Number of channels: {features.size(1)}')
        print(f'Image height: {features.size(2)}')
        print(f'Image width: {features.size(3)}')
        print(f'Feature pixel value range: {features.min().item()} to {features.max().item()}')

    unique_labels, counts = torch.unique(labels, return_counts=True)
    print('Labels distribution in the batch:')
    for label, count in zip(unique_labels, counts):
        print(f'Label {label.item()}: {count.item()} samples')

# -------------------------------
# 8. ENTRY POINT
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossDConv/ResNet-18 with Progressive Finetuning and 3 Trials x 5-Fold CV on npz data')
    parser.add_argument('--data_flag',  default=Config.DEFAULTS['data_flag'], type=str, 
                        help='Which .npz file to load (prefix or dataset name).')
    parser.add_argument('--num_epochs', default=Config.DEFAULTS['num_epochs'], type=int, 
                        help='Number of epochs for training.')
    parser.add_argument('--batch_size', default=Config.DEFAULTS['batch_size'], type=int,
                        help='Batch size.')
    parser.add_argument('--conv',       default=Config.DEFAULTS['conv'], type=str,
                        help='Choose convolution option (e.g., "regular", "crossdconv").')
    parser.add_argument('--weights_path', default=None, type=str,
                        help='Custom path to weights file if different from default.')
    args = parser.parse_args()

    main(args)
