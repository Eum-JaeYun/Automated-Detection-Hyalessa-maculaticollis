#%%
"""
Evaluating models trained with K-fold
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score, confusion_matrix, roc_curve, auc, 
    precision_score, recall_score, average_precision_score, precision_recall_curve
)
#%%
class EnsembleModel(nn.Module):
    def __init__(self, model_paths, cfg):
        super().__init__()
        self.models = nn.ModuleList()
        self.cfg = cfg
        
        print(f"Loading ensemble... {len(model_paths)} models in total")
        for path in model_paths:
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(cfg['dropout_rate']),
                nn.Linear(model.fc.in_features, cfg['num_classes'])
            )
            state_dict = torch.load(path, map_location=cfg['device'])
            model.load_state_dict(state_dict)
            model.to(cfg['device'])
            model.eval()
            self.models.append(model)
            print(f"- Loaded: {os.path.basename(path)}")

    def forward(self, x):
        total_prob = 0.0
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
                total_prob += prob
        return total_prob / len(self.models)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)

class GPUAugmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, x): return self.normalize(x)

def evaluate_ensemble(ensemble_model, test_dl, cfg):
    print(f"\n📊 Final Ensemble Evaluation (Threshold: {cfg['threshold']})")
    
    all_labels = []
    all_preds = []
    all_probs = [] 
    full_results = []
    
    gpu_proc = GPUAugmentation(cfg).to(cfg['device'])
    
    for inputs, labels, paths in tqdm(test_dl, desc="Ensemble Testing"):
        inputs = inputs.to(cfg['device'])
        labels = labels.to(cfg['device'])
        inputs = gpu_proc(inputs)
        
        avg_probs = ensemble_model(inputs)

        cicada_probs = avg_probs[:, 1]
        preds = (cicada_probs >= cfg['threshold']).long()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(cicada_probs.cpu().numpy())
        
        for i in range(len(labels)):
            file_name = os.path.basename(paths[i])
            true_idx = labels[i].item()
            pred_idx = preds[i].item()
            prob_val = cicada_probs[i].item()
            
            true_name = test_dl.dataset.classes[true_idx]
            pred_name = test_dl.dataset.classes[pred_idx]
            
            full_results.append({
                'File Name': file_name,
                'True Label': true_name,
                'Predicted Label': pred_name,
                'Cicada Probability': prob_val,
                'Correct': 'O' if true_idx == pred_idx else 'X'
            })

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    roc_auc = 0.0
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except: pass

    # AP (Average Precision)
    ap_score = average_precision_score(all_labels, all_probs)

    # --- Result Output ---
    print("\n" + "="*40)
    print(f"Detailed Performance Report (Threshold={cfg['threshold']})")
    print("="*40)
    print(f" [Confusion Matrix]")
    print(f"  - TP: {tp:3d} | FN: {fn:3d}")
    print(f"  - FP: {fp:3d} | TN: {tn:3d}")
    print("-" * 40)
    print(f" [Key Indicators]")
    print(f"  - Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("-" * 40)
    print(f" [Overall Performance Summary]")
    print(f"  - AUROC: {roc_auc:.4f}")
    print(f"  - AP(Average Precision): {ap_score:.4f}")
    print("="*40 + "\n")
    
    # Save the results file as csv
    save_path = os.path.join(CHECKPOINT_DIR, f'ensemble_test_results_thr{cfg["threshold"]}.csv')
    df = pd.DataFrame(full_results)
    df = df.sort_values(by='Cicada Probability', ascending=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Result file saved: {save_path}")
    
    # --- visualization: Confusion Matrix + ROC + PR Curve ---
    plt.figure(figsize=(18, 5))
        
    # 1. Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dl.dataset.classes, yticklabels=test_dl.dataset.classes)
    plt.title(f'Confusion Matrix (Thr={cfg["threshold"]})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    
    
    # 2. ROC Curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(); plt.title('ROC Curve')
        
    # 3. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    plt.subplot(1, 3, 3)
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'AP = {ap_score:.4f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(); plt.title('Precision-Recall Curve')
        
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR,'output.png'), dpi=300)
    plt.show()

# Misclassification analysis
def analyze_ensemble_errors(ensemble_model, test_loader, cfg):
    save_dir = os.path.join(CHECKPOINT_DIR, f'ensemble_error_analysis_thr{cfg["threshold"]}')
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    print(f"\nStart misclassification analysis (Threshold: {cfg['threshold']})")
    
    gpu_proc = GPUAugmentation(cfg).to(cfg['device'])
    error_log = []
    cnt = 0
    visual_samples = []
    
    for inputs, labels, paths in tqdm(test_loader, desc="Errors Analyzing"):
        inputs = inputs.to(cfg['device'])
        labels = labels.to(cfg['device'])
        inputs_norm = gpu_proc(inputs)
        
        avg_probs = ensemble_model(inputs_norm)
        cicada_probs = avg_probs[:, 1]
        preds = (cicada_probs >= cfg['threshold']).long()
        
        wrong_indices = (preds != labels).nonzero().flatten()
        
        for idx in wrong_indices:
            true_label = labels[idx].item()
            pred_label = preds[idx].item()
            file_name = os.path.basename(paths[idx])
            prob_val = cicada_probs[idx].item()
            
            true_name = test_loader.dataset.classes[true_label]
            pred_name = test_loader.dataset.classes[pred_label]
            
            error_log.append({
                'File Name': file_name,
                'True Label': true_name,
                'Predicted': pred_name,
                'Cicada Prob': f"{prob_val:.4f}", 
                'Full Path': paths[idx]
            })
            
            mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
            img_np = inputs_norm[idx].cpu().permute(1, 2, 0).numpy()
            img_np = std * img_np + mean; img_np = np.clip(img_np, 0, 1)
            
            plt.imsave(os.path.join(save_dir, f"Wrong_{true_name}_pred_{pred_name}_{file_name}.png"), img_np, dpi=300)
            cnt += 1
            
            if len(visual_samples) < 6:
                visual_samples.append((img_np, true_name, pred_name))
            
    if error_log:
        pd.DataFrame(error_log).to_csv(os.path.join(save_dir, 'ensemble_error_list.csv'), index=False, encoding='utf-8-sig')
        print(f"A total of {cnt} misclassified data were found.")
        
        if visual_samples:
            print("\n Misclassification examples (Top 6)")
            num_view = min(6, len(visual_samples))
            fig, axes = plt.subplots(1, num_view, figsize=(15, 5))
            if num_view == 1: axes = [axes]
            for i in range(num_view):
                img, t, p = visual_samples[i]
                axes[i].imshow(img)
                axes[i].set_title(f"True: {t}\nPred: {p}", color='red')
                axes[i].axis('off')
            plt.show()
    else:
        print(f"Perfect! No errors at threshold {cfg['threshold']}.")
#%%
if __name__ == '__main__':
    CHECKPOINT_DIR = input(r"Enter the path to the folder containing the trained model: ")
    DATA_DIR = input(r"Enter the path to the folder containing the test data set: ")
    th = float(input("Please enter a threshold value between 0.0 and 1.0: "))

    # --- Configuration ---
    # If you want to adjust parameters, modify the dictionary below
    CFG = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'img_size': (224, 224),
        'batch_size': 128,
        'num_classes': 2,
        'dropout_rate': 0.4, # Set the same as the learning model
        'threshold': th  # [Important] Setting the threshold
        }
    
    model_paths = glob.glob(os.path.join(CHECKPOINT_DIR, "fold_*_best.pt"))
    
    if not model_paths:
        print(f"Error: No model files in folder '{CHECKPOINT_DIR}'")
    else:
        ensemble_model = EnsembleModel(model_paths, CFG)
        test_ds = ImageFolderWithPaths(os.path.join(DATA_DIR, 'Test'), transform=transforms.ToTensor())
        test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=0)
        
        evaluate_ensemble(ensemble_model, test_loader, CFG)
        analyze_ensemble_errors(ensemble_model, test_loader, CFG)
