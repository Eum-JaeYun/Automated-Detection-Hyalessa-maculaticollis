#%%
"""
Learning using ResNet (Cicada Detection)
Refined based on discussion:
1. Correct Metric Calculation
2. Two-phase Training (Warmup -> Fine-tuning)
3. Domain Adaptive BN Updates
"""
import os
# System settings: prevent computer overload
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import json
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
#%%
# [Configuration]
CFG = {
    'project_name': 'Cicada_ResNet18_Final', 
    'data_dir': r"Enter the segmented training image path: ",
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
    
    'img_size': (224, 224),
    'batch_size': 128,
    'num_workers': 0,
    'num_classes': 2,
    'k_folds': 5,
    
    'aug_prob_noise': 0.05,
    'noise_std': 0.05,     
    'sp_ratio': 0.02,      
    
    'use_specaug': True, 
    'aug_prob_spec': 0.6,
    'spec_freq_mask': 30,
    'spec_time_mask': 40,
    
    'num_epochs': 100, 
    'warmup_epochs': 10,
    'dropout_rate': 0.4,
    
    'lr_fc': 1e-4,                  
    'lr_base': 1e-5,                
    'lr_head': 5e-5,                
    
    'patience': 25, 
    'weight_decay': 0.01
}

# Setup Directories
Runing_Date = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
CHECKPOINT_DIR = os.path.join(CFG['data_dir'], f'{Runing_Date}')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Save Hyperparameters
cfg_to_save = CFG.copy()
cfg_to_save['device'] = str(cfg_to_save['device'])
save_path = os.path.join(CHECKPOINT_DIR, 'hyperparameters.json')
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(cfg_to_save, f, indent=4, ensure_ascii=False)
print(f"Parameters saved to: {save_path}")

# --- Classes & Functions ---

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class GPUAugmentation(nn.Module):
    def __init__(self, cfg, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.cfg = cfg
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        if self.is_training:
            # Time Shift
            if torch.rand(1, device=x.device) < 0.5:
                x = self.apply_time_shift(x)

            # Random Spectrogram Scaling
            if torch.rand(1, device=x.device) < 0.5:
                scale = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
                x = x * scale

            # SpecAugment
            if self.cfg['use_specaug'] and torch.rand(1, device=x.device) < self.cfg['aug_prob_spec']:
                x = self.apply_spec_augment(x)
            
            # Noise Injection
            if torch.rand(1, device=x.device) < self.cfg['aug_prob_noise']:
                if torch.rand(1, device=x.device) < 0.5: 
                    x = self.add_salt_pepper_noise(x)
                else: 
                    x = self.add_gaussian_noise(x)
                    
        return self.normalize(x)

    def apply_time_shift(self, x):
        B, C, H, W = x.shape
        shift_max = int(W * 0.2) 
        shift_amt = torch.randint(-shift_max, shift_max, (1,)).item()
        return torch.roll(x, shifts=shift_amt, dims=3)

    def apply_spec_augment(self, x):
        B, C, H, W = x.shape; mask_val = 0.0
        if self.cfg['spec_freq_mask'] > 0:
            f_width = torch.randint(0, self.cfg['spec_freq_mask'], (1,)).item()
            if f_width > 0:
                f_start = torch.randint(0, H - f_width, (1,)).item()
                x[:, :, f_start:f_start+f_width, :] = mask_val
        if self.cfg['spec_time_mask'] > 0:
            t_width = torch.randint(0, self.cfg['spec_time_mask'], (1,)).item()
            if t_width > 0:
                t_start = torch.randint(0, W - t_width, (1,)).item()
                x[:, :, :, t_start:t_start+t_width] = mask_val
        return x

    def add_gaussian_noise(self, x): 
        return torch.clamp(x + torch.randn_like(x) * self.cfg['noise_std'], 0.0, 1.0)
    
    def add_salt_pepper_noise(self, x):
        noisy = x.clone(); prob = torch.rand_like(x)
        noisy[prob < self.cfg['sp_ratio']] = 1.0; noisy[prob > (1 - self.cfg['sp_ratio'])] = 0.0
        return noisy

def get_model(cfg):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Start with all frozen
    for param in model.parameters(): param.requires_grad = False
    # Replace FC (Head)
    model.fc = nn.Sequential(nn.Dropout(cfg['dropout_rate']), nn.Linear(model.fc.in_features, cfg['num_classes']))
    return model
    
def plot_history(loss_hist, acc_hist, fold_idx):
    epochs = range(1, len(loss_hist['train']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist['train'], 'b-', label='Train')
    plt.plot(epochs, loss_hist['val'], 'r-', label='Val')
    plt.title(f'Fold {fold_idx + 1} Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_hist['train'], 'b-', label='Train')
    plt.plot(epochs, acc_hist['val'], 'r-', label='Val')
    plt.title(f'Fold {fold_idx + 1} Accuracy Curve')
    plt.legend()
    
    save_path = os.path.join(CHECKPOINT_DIR, f'fold_{fold_idx}_history.png')
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved: {save_path}")
    # plt.show() # Optional based on environment

def train_one_fold(fold_idx, model, train_loader, val_loader, cfg):
    gpu_aug = {'train': GPUAugmentation(cfg, True).to(cfg['device']), 'val': GPUAugmentation(cfg, False).to(cfg['device'])}
    
    # Loss functions
    train_loss_func = nn.CrossEntropyLoss(label_smoothing=0.1) 
    logging_loss_func = nn.CrossEntropyLoss()
    val_loss_func = nn.CrossEntropyLoss()
    
    # 1. Initial Optimizer for Warmup (Only FC)
    optimizer = optim.AdamW(model.fc.parameters(), lr=cfg['lr_fc'], weight_decay=cfg['weight_decay'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    
    best_loss = float('inf'); best_acc = 0.0
    early_stopping = EarlyStopping(patience=cfg['patience'])

    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}
    
    print(f"\n[Fold {fold_idx+1}/{cfg['k_folds']}] Training Start")
    
    for epoch in range(cfg['num_epochs']):
        # --- Strategy: 2-Phase Learning ---
        if epoch == cfg['warmup_epochs']:
            print(f"\n[Info] Epoch {epoch}: Warmup Finished. Unfreezing Layer4 & Resetting Optimizer.")
            
            # Unfreeze Layer4 & FC
            for param in model.layer4.parameters(): param.requires_grad = True
            for param in model.fc.parameters(): param.requires_grad = True
            
            # Reset Optimizer (Momentum reset, Weights kept)
            optimizer = optim.AdamW([
                {'params': model.layer4.parameters(), 'lr': cfg['lr_base']}, 
                {'params': model.fc.parameters(), 'lr': cfg['lr_head']}
            ], lr=cfg['lr_head'], weight_decay=cfg['weight_decay'])
            
            # Reset Scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=7
            )
        
        # --- Train ---
        # NOTE: model.train() is kept even when layers are frozen.
        # This allows BN layers to update stats (mean/var) for the spectrogram domain.
        model.train()
        
        train_loss = 0.0
        train_corrects = 0
        total_train_samples = 0 # [Modified] Counter for accurate calculation
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg['device']), labels.to(cfg['device'])
            with torch.no_grad(): inputs = gpu_aug['train'](inputs)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = train_loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Logging
            batch_size = inputs.size(0) # Actual batch size
            with torch.no_grad():
                pure_loss = logging_loss_func(outputs, labels)
                train_loss += pure_loss.item() * batch_size
            
            train_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
            total_train_samples += batch_size # Accumulate samples
            
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val_samples = 0 # [Modified] Counter for accurate calculation
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(cfg['device']), labels.to(cfg['device'])
                inputs = gpu_aug['val'](inputs)
                outputs = model(inputs)
                
                loss = val_loss_func(outputs, labels)
                
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
                total_val_samples += batch_size
        
        # [Modified] Accurate Average Calculation
        epoch_train_loss = train_loss / total_train_samples
        epoch_train_acc = (train_corrects.double() / total_train_samples).item() 
        
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = (val_corrects.double() / total_val_samples).item()
        
        # History & Scheduler
        loss_history['train'].append(epoch_train_loss); loss_history['val'].append(epoch_val_loss)
        acc_history['train'].append(epoch_train_acc); acc_history['val'].append(epoch_val_acc)
        
        if scheduler: 
            scheduler.step(epoch_val_loss)
        
        # Checkpoint
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss; best_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'fold_{fold_idx}_best.pt'))
        
        # Print Log
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:03d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {lr_current:.2e}")

        # Early Stopping
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch+1}")
            break

    print(f"Fold {fold_idx+1} Done. Best Loss: {best_loss:.4f} / Best Acc: {best_acc:.4f}")
    plot_history(loss_history, acc_history, fold_idx)
    return best_acc
#%%
if __name__ == '__main__':
    # Prepare Data
    to_tensor = transforms.ToTensor()
    # Check if paths exist before running
    if not os.path.exists(os.path.join(CFG['data_dir'], 'Train')):
         print("Error: Train directory not found. Please check 'data_dir' in CFG.")
    else:
        combined_ds = datasets.ImageFolder(os.path.join(CFG['data_dir'], 'Train'), transform=to_tensor)
        
        kfold = KFold(n_splits=CFG['k_folds'], shuffle=True, random_state=CFG['seed'])
        fold_results = []
        
        print(f"Running Device: {CFG['device']}")
        print(f"Saving Checkpoints to: {CHECKPOINT_DIR}")
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(combined_ds)):
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(combined_ds, batch_size=CFG['batch_size'], sampler=train_subsampler, num_workers=CFG['num_workers'])
            val_loader = DataLoader(combined_ds, batch_size=CFG['batch_size'], sampler=val_subsampler, num_workers=CFG['num_workers'])
            
            model = get_model(CFG).to(CFG['device'])
            best_acc = train_one_fold(fold, model, train_loader, val_loader, CFG)
            fold_results.append(best_acc)
            
        print("\n" + "="*40)
        print(f"{CFG['k_folds']}-Fold Validation Results")
        print(f"  Mean Acc  : {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
        print("="*40)
        print(f"\n[Guide] Learning Complete. Path to be used in 'evaluate.py':")
        print(f"{CHECKPOINT_DIR}")