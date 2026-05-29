import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from tqdm import tqdm

# --- 1. Dataset Class ---
class PURE_H5_Dataset(Dataset):
    def __init__(self, h5_path, window_size=128, train=True):
        self.h5_path = h5_path
        self.window_size = window_size
        self.data = None 
        
        with h5py.File(self.h5_path, 'r') as f:
            all_subjects = sorted(list(f.keys()))
            split = int(len(all_subjects) * 0.8)
            self.subjects = all_subjects[:split] if train else all_subjects[split:]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(self.h5_path, 'r')
        
        sub = self.subjects[idx]
        frames = self.data[sub]['frames']
        bvp = self.data[sub]['bvp']

        # Random window slice
        start = np.random.randint(0, len(frames) - self.window_size)
        x = frames[start : start + self.window_size]
        y = bvp[start : start + self.window_size]

        # Transform: (T, H, W, C) -> (C, T, H, W) and scale to [0, 1]
        x = torch.from_numpy(x).float().permute(3, 0, 1, 2) / 255.0
        y = torch.from_numpy(y).float()
        
        # Z-score normalize BVP signal
        y = (y - y.mean()) / (y.std() + 1e-6)
        
        return x, y

# --- 2. Loss Function (Negative Pearson Correlation) ---
class PearsonLoss(nn.Module):
    def forward(self, pred, target):
        x = pred - torch.mean(pred)
        y = target - torch.mean(target)
        polar = torch.sum(x * y)
        rect = torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2))
        return 1 - (polar / (rect + 1e-6))

# --- 3. Training Loop ---
def train_model():
    # Hyperparameters
    DEVICE = torch.device("cpu")
    EPOCHS = 50
    BATCH_SIZE = 4
    LR = 1e-4
    H5_PATH = "/home/jib/pure_processed_192.h5"

    # Data Loaders
    train_ds = PURE_H5_Dataset(H5_PATH, train=True)
    val_ds = PURE_H5_Dataset(H5_PATH, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model (Assuming nasir6/LinkNet is imported or defined)
    # Replace 'YourNasirModel()' with the actual class name from your repo
    from models import Nasir6  # Adjust based on your actual file name
    model = Nasir6().to(DEVICE)

    criterion = PearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Expected output shape: (Batch, Window)
            
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()

        print(f"Avg Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Avg Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"nasir6_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_model()