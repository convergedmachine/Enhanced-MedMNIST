import argparse
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from torchvision.models import resnet18, ResNet18_Weights
from acsconv.converters import ACSConverter  # For converting ResNet to ResNet3D

from monai.transforms import (
    Compose,
    ScaleIntensity,
    NormalizeIntensity,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise,
    RandShiftIntensity,
    EnsureType
)

from replace_conv_layers import convert2threed  # Ensure this module is available

# -------------------------------
# Configuration
# -------------------------------
class Config:
    SEED = 42
    FLOAT32_MATMUL_PRECISION = 'high'
    WEIGHTS_PATH = {
        'crossdconv': "../_weights/CDConvR18.pth",
        'acsconv': "../_weights/R18.pth"
    }
    DEFAULTS = {
        'data_flag': 'organmnist3d_64',
        'num_epochs': 100,
        'batch_size': 8,
        'num_workers': 8,      # Default number of workers
        'conv': 'crossdconv',  # Default convolution option
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
    # For Python random module
    import random
    random.seed(seed)

set_seed(Config.SEED)
torch.set_float32_matmul_precision(Config.FLOAT32_MATMUL_PRECISION)

# -------------------------------
# 3. MODEL CREATION
# -------------------------------
def create_model(n_classes, conv_opt='imagenet', weights_path=None):
    """
    Creates a ResNet-18 3D model with optional convolution layer replacements.

    Args:
        n_classes (int): Number of output classes.
        conv_opt (str): Convolution option. One of 'crossdconv', 'acsconv', 'imagenet'.
        weights_path (str, optional): Path to the weights file.

    Returns:
        model (nn.Module): The modified ResNet-18 3D model.
    """
    # Initialize ResNet-18 with ImageNet weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)  # Modify the final layer

    # Handle convolution options
    if conv_opt == 'crossdconv':
        # Convert to 3D using CrossDConv
        convert2threed(model)
        weights_to_load = weights_path or Config.WEIGHTS_PATH['crossdconv']

        try:
            ckpt = torch.load(weights_to_load, map_location='cpu')
            state_dict = ckpt.get("model", ckpt)

            # Modify state_dict keys if necessary
            weights = {
                key.replace("weights_3d", "weight"): value 
                for key, value in state_dict.items()
            }

            # Remove weights related to the fully connected layer
            keys_to_remove = ['fc.weight', 'fc.bias']
            for key in keys_to_remove:
                weights.pop(key, None)

            # Exclude rotation parameters and batch norm layers if present
            weights = {
                key: value for key, value in weights.items()
                if 'rotation_params' not in key and 'bn' not in key
            }

            # Load the modified state_dict
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            print(f"Loaded CrossDConv weights with missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        except FileNotFoundError:
            print(f"CrossDConv weights not found at {weights_to_load}. Proceeding without loading pre-trained weights.")
        except Exception as e:
            print(f"Error loading CrossDConv weights: {e}")

    elif conv_opt == 'acsconv':
        # Apply ACSConv conversion
        weights_to_load = weights_path or Config.WEIGHTS_PATH['acsconv']

        try:
            ckpt = torch.load(weights_to_load, map_location='cpu')
            state_dict = ckpt.get("model", ckpt)

            # Modify state_dict keys if necessary
            weights = {
                key.replace("_orig_mod.", ""): value 
                for key, value in state_dict.items()
            }

            # Remove weights related to the fully connected layer
            keys_to_remove = ['fc.weight', 'fc.bias']
            for key in keys_to_remove:
                weights.pop(key, None)

            # Load the modified state_dict
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            model = ACSConverter(model)
            print(f"Loaded ACSConv weights with missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        except FileNotFoundError:
            print(f"ACSConv weights not found at {weights_to_load}. Proceeding without loading pre-trained weights.")
        except Exception as e:
            print(f"Error loading ACSConv weights: {e}")

    elif conv_opt == 'imagenet':
        model = ACSConverter(model)
        print("Using 3D ResNet-18 with ImageNet weights.")
    else:
        raise ValueError(f"Unsupported conv_opt '{conv_opt}'. Choose from 'crossdconv', 'acsconv', 'imagenet'.")

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
            images (np.ndarray): Array of images with shape (N, D, H, W).
            labels (np.ndarray): Array of labels.
            transform (callable, optional): Transformations to apply.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        if self.images.ndim == 4:
            # Expand to (N, 1, D, H, W) and repeat channels to make it (N, 3, D, H, W)
            self.images = np.expand_dims(self.images, 1)
            self.images = np.repeat(self.images, 3, axis=1)
        elif self.images.ndim == 5 and self.images.shape[1] == 3:
            # Already in (N, C, D, H, W)
            pass
        else:
            raise ValueError("Images should have shape (N, C, D, H, W) with C=3 or (N, D, H, W).")

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
    Preset transformations for training using MONAI.
    """
    def __init__(self, mean, std, flip_prob=0.5, rotate_prob=0.5, zoom_prob=0.5, noise_prob=0.5, shift_intensity_prob=0.5):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
            RandFlip(spatial_axis=0, prob=flip_prob),
            RandRotate(range_x=15, prob=rotate_prob),  # Random rotation within ±15 degrees
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=zoom_prob),  # Random zoom between 90% and 110%
            RandGaussianNoise(prob=noise_prob),  # Add random Gaussian noise
            RandShiftIntensity(offsets=0.1, prob=shift_intensity_prob),  # Random intensity shift
            EnsureType()
        ])

    def __call__(self, img):
        return self.transforms(img)

class BiomedicalPresetEval:
    """
    Preset transformations for evaluation using MONAI.
    """
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
            EnsureType()
        ])

    def __call__(self, img):
        return self.transforms(img)

# -------------------------------
# 5a. FINETUNING SCHEDULER
# -------------------------------
import torch.nn as nn

class FinetuningScheduler:
    """
    Scheduler to manage progressive finetuning by unfreezing layers at specified epochs.
    Ensures that all Batch Normalization layers remain trainable throughout training.
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

        # Ensure BatchNorm layers are always trainable
        self._keep_batch_norm_trainable()

    def _keep_batch_norm_trainable(self):
        """
        Sets requires_grad=True for all BatchNorm layers to keep them trainable.
        """
        # Define BatchNorm layer types
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

        for module in self.model.modules():
            if isinstance(module, bn_types):
                for param in module.parameters():
                    param.requires_grad = True

    def step(self, epoch, optimizer):
        """
        Checks if a new finetuning stage should be activated at the current epoch and updates the optimizer.

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
                try:
                    layer = dict([*self.model.named_modules()])[layer_name]
                except KeyError:
                    print(f"Layer '{layer_name}' not found in the model. Skipping.")
                    continue

                for param in layer.parameters():
                    param.requires_grad = True

            # Reaffirm that BatchNorm layers remain trainable
            self._keep_batch_norm_trainable()

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
        
        # Create a new optimizer with the updated trainable parameters
        optimizer = torch.optim.Adam(trainable_params, lr=optimizer.param_groups[0]['lr'])
        print("Optimizer updated to include newly trainable parameters.")
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
        conv_opt (str): Convolution option ('crossdconv', 'acsconv', 'imagenet').
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

    if conv_opt != 'imagenet':
        # For conv_opt other than 'imagenet', freeze all layers except the final fully connected layer
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    else:
        # For 'imagenet', you might choose to fine-tune the entire model or freeze specific layers
        print("Training with standard ImageNet weights. Consider freezing layers if necessary.")

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
    num_workers = Config.DEFAULTS['num_workers']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load .npz data
    path_to_npz = os.path.join("./Enhanced-MedMNIST", f"{data_flag}.npz")
    if not os.path.isfile(path_to_npz):
        raise FileNotFoundError(f"Data file not found at {path_to_npz}")

    data = np.load(path_to_npz)
    images = data['images'] / 255.0  # Assuming images are in [0, 255]
    images = np.stack([images, images, images], axis=1)
    labels_all = data['labels'].squeeze()

    # Ensure correct image shape
    if images.ndim == 4:
        pass  # Already handled in ArrayDataset
    elif images.ndim == 5 and images.shape[1] == 3:
        pass  # Already handled in ArrayDataset
    elif images.ndim == 5 and images.shape[1] != 3:
        images = np.transpose(images, [0, 4, 1, 2, 3])  # From (N, D, H, W, C) to (N, C, D, H, W)
    else:
        raise ValueError("Unsupported image shape. Expected 4D or 5D array.")

    n_classes = len(np.unique(labels_all))
    print(f"Number of classes: {n_classes}")

    # Define mean and std based on conv option
    if conv == "imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0.3162, 0.3162, 0.3162])
        std = np.array([0.3213, 0.3213, 0.3213])
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    # Initialize transformation presets
    train_transform = BiomedicalPresetTrain(
        mean=mean,  # Lists are passed directly
        std=std,
        flip_prob=0.5
    )

    eval_transform = BiomedicalPresetEval(
        mean=mean,  # Lists are passed directly
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
                num_workers=num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
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

    print("\nData statistics:")
    print(f"Type: {type(features)} {features.dtype}")
    print(f"Shape: {features.shape}")
    print(f"Value range: ({features.min().item()}, {features.max().item()})")

    if features.ndim == 5:
        print(f'Feature batch size: {features.size(0)}')
        print(f'Number of channels: {features.size(1)}')
        print(f'Depth: {features.size(2)}')
        print(f'Height: {features.size(3)}')
        print(f'Width: {features.size(4)}')
        print(f'Feature pixel value range: {features.min().item()} to {features.max().item()}')

    unique_labels, counts = torch.unique(labels, return_counts=True)
    print('Labels distribution in the batch:')
    for label, count in zip(unique_labels, counts):
        print(f'Label {label.item()}: {count.item()} samples')

# -------------------------------
# 8. ENTRY POINT
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossDConv/ACSConv/ResNet-18 3D with Progressive Finetuning and 3 Trials x 5-Fold CV on npz data')
    parser.add_argument('--data_flag',  default=Config.DEFAULTS['data_flag'], type=str, 
                        help='Which .npz file to load (prefix or dataset name).')
    parser.add_argument('--num_epochs', default=Config.DEFAULTS['num_epochs'], type=int, 
                        help='Number of epochs for training.')
    parser.add_argument('--batch_size', default=Config.DEFAULTS['batch_size'], type=int,
                        help='Batch size.')
    parser.add_argument('--conv',       default=Config.DEFAULTS['conv'], type=str, 
                        choices=['crossdconv', 'acsconv', 'imagenet'],
                        help='Choose convolution option: "crossdconv", "acsconv", or "imagenet".')
    parser.add_argument('--weights_path', default=None, type=str,
                        help='Custom path to weights file if different from default.')
    parser.add_argument('--num_workers', default=Config.DEFAULTS['num_workers'], type=int,
                        help='Number of worker processes for DataLoader.')    
    args = parser.parse_args()

    main(args)
