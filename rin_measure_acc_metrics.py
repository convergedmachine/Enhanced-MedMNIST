import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.models import resnet18
from replace_conv_layers import replace_conv_layers
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
import timm
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import OrderedDict
import presets
from tqdm import tqdm
torch.set_float32_matmul_precision('high')

preprocessing = presets.ClassificationPresetEval(
    crop_size=224,
    resize_size=256,
    interpolation=InterpolationMode("bilinear"),
    backend="tensor",
    use_v2=False
)

_path = "/scratch/myavuz/rRadImagenet1L/test/"

dataset_test = torchvision.datasets.ImageFolder(_path, preprocessing)

data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=64, num_workers=12, shuffle=False, pin_memory=True
    )

my_model = resnet18(num_classes=165)
replace_conv_layers(my_model)
my_model.to(device="cuda:0")

model_checkpoint = torch.load("../_weights/CDConvR18.pth", map_location=torch.device('cuda:0'))['model']
new_state_dict = OrderedDict()
for k, v in model_checkpoint.items():
    new_state_dict[k[10:]] = v        
        
my_model.load_state_dict(model_checkpoint)   

from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

def check_metrics(loader, model, device="cuda:0", num_classes=165):
    # Prepare model for evaluation
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x).softmax(-1)
            predictions = torch.argmax(scores, dim=1)
            
            # Collect all predictions and targets
            all_predictions.append(predictions)
            all_targets.append(y)
            all_scores.append(scores)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # Compute the final metrics using torcheval's functional metrics
    test_precision_score = precision_score(all_targets, all_predictions, average='macro')
    test_recall_score = recall_score(all_targets, all_predictions, average='macro')
    test_f1_score = f1_score(all_targets, all_predictions, average='macro')
    test_roc_auc_score = roc_auc_score(all_targets, all_scores, average='macro', multi_class='ovo')
    test_balanced_accuracy_score = balanced_accuracy_score(all_targets, all_predictions)
    test_accuracy_score = accuracy_score(all_targets, all_predictions)
    
    return test_precision_score, test_recall_score, test_f1_score, test_roc_auc_score, test_balanced_accuracy_score, test_accuracy_score

# Example usage:
test_precision_score, test_recall_score, test_f1_score, test_roc_auc_score, test_balanced_accuracy_score, test_accuracy_score = check_metrics(data_loader, my_model)
print(f"Test Precision: {test_precision_score:.4f}")
print(f"Test Recall: {test_recall_score:.4f}")
print(f"Test F1-Score: {test_f1_score:.4f}")
print(f"Test AUROC: {test_roc_auc_score:.4f}")
print(f"Test Accuracy (Balanced): {test_balanced_accuracy_score:.4f}")
print(f"Test Accuracy: {test_accuracy_score:.4f}")

