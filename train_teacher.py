import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm

# Import existing modules
from config import get_paths, set_seed, get_rng_state, set_rng_state, save_checkpoint
from data.dataset import get_datasets
from models.teacher import get_teacher_model
from utils.loss_metrics import WeightedDiceBCELoss, calculate_dice

# ============================================================
# 1. Configuration
# ============================================================
parser = argparse.ArgumentParser(description="Teacher Training")
parser.add_argument("--dataset", type=str, default="brats2023", choices=["brats2023", "msd"])
parser.add_argument("--data_dir", type=str, default=None, help="Override dataset directory")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--fold", type=int, default=4)
args = parser.parse_args()

DATASET_TYPE = args.dataset  # Choose "brats2023" or "msd"
CURRENT_FOLD = args.fold
NUM_FOLDS = 5
SEED = 42

# Hyperparameters
TOTAL_EPOCHS = args.epochs
BATCH_SIZE = 2
LR = 1e-4
ROI_SIZE = (128, 128, 128)
VAL_INTERVAL = 1

# Get paths dynamically
paths = get_paths(DATASET_TYPE, CURRENT_FOLD, args.data_dir)
DATA_DIR = paths["data_dir"]
SAVE_DIR = paths["teacher_save_dir"]
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, "best_teacher.pth")
LAST_PATH = os.path.join(SAVE_DIR, "last_teacher.pth")
IS_KAGGLE = paths["is_kaggle"]

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Dataset: {DATASET_TYPE} | Fold: {CURRENT_FOLD} | Data Dir: {DATA_DIR}")

# ============================================================
# 2. Setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)


def main():
    # --- 1. Data Loading ---
    # sdf_dir is set to None because the Teacher model does not require Boundary Loss/SDFs
    train_ds, val_ds = get_datasets(
        data_dir=DATA_DIR,
        fold=CURRENT_FOLD,
        num_folds=NUM_FOLDS,
        seed=SEED,
        sdf_dir=None,
        roi_size=ROI_SIZE
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- 2. Model Initialization ---
    # Initializes the standard MONAI U-Net defined in teacher.py
    model = get_teacher_model(device)

    # Calculate and display parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Teacher Initialized. Total Params: {params / 1e6:.2f} M")

    # --- 3. Optimizer & Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # Standard cosine annealing scheduler for segmentation tasks
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    # Teacher uses standard Supervised Loss (Dice + BCE)
    loss_function = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])

    # --- 4. Training Loop ---
    best_dice = 0.0
    start_epoch = 1

    # Resume training logic
    if os.path.exists(LAST_PATH):
        print(f"Resuming training from checkpoint: {LAST_PATH}")
        ckpt = torch.load(LAST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            scheduler.step(ckpt["epoch"])
            print("Warning: legacy checkpoint has no scheduler state, position reconstructed from epoch.")
        if not set_rng_state(ckpt.get("rng_state")):
            set_seed(SEED + ckpt['epoch'])
            print("Warning: legacy checkpoint has no RNG state, using a deterministic new RNG sequence.")
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)

    # Session timeout setup
    safety_timeout = False
    epoch_durations = []
    max_session_seconds = max(0.0, float(os.environ.get("KAGGLE_MAX_QUOTA_HOURS", 11.5)) * 3600 - 300)
    session_start_time = float(os.environ.get("KAGGLE_SESSION_START", time.time()))

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        epoch_start_time = time.time()

        model.train()
        epoch_loss = 0.0
        train_dices = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")

        for batch in pbar:
            # Dataset returns: [image, label, sdf, case_name]
            # Note: Teacher model ignores SDF and case_name
            images, labels, _, _ = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate Loss (Supervised only)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate Training Dice for real-time monitoring
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_dices.append(calculate_dice(preds, labels))

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Dice': f"{np.mean(train_dices[-1]):.3f}"})

        # Update learning rate
        scheduler.step()
        epoch_loss /= len(train_loader)

        # --- Validation Loop ---
        if epoch % VAL_INTERVAL == 0:
            model.eval()
            val_dices = []

            with torch.no_grad():
                for batch in val_loader:
                    images, labels, _, _ = batch
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    val_dices.append(calculate_dice(preds, labels))

            mean_val_dice = np.mean(val_dices)
            val_dices_arr = np.array(val_dices)

            # Performance Logging by Channel
            wt_dice = val_dices_arr[:, 0].mean()
            tc_dice = val_dices_arr[:, 1].mean()
            et_dice = val_dices_arr[:, 2].mean()

            print(f"\nEvaluation for Epoch {epoch}:")
            print(f"  Average Train Loss: {epoch_loss:.4f}")
            print(f"  Average Val Dice : {mean_val_dice:.4f} (WT={wt_dice:.3f}, TC={tc_dice:.3f}, ET={et_dice:.3f})")

            # Update Best Model
            if mean_val_dice > best_dice:
                best_dice = mean_val_dice

                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_dice": float(best_dice),
                }, BEST_PATH)

                print(f"New Best Teacher Saved! Score: ({best_dice:.4f})")

        # Save Last Checkpoint (for potential resumption)
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": float(best_dice),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": get_rng_state(),
        }, LAST_PATH)

        # Safety timeout check
        if IS_KAGGLE:
            epoch_durations.append(time.time() - epoch_start_time)
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            est_next_epoch = max(avg_epoch_time, max(epoch_durations)) * 1.15
            elapsed_session = time.time() - session_start_time
            remaining_session = max_session_seconds - elapsed_session

            if epoch < TOTAL_EPOCHS and remaining_session < est_next_epoch:
                safety_timeout = True
                print(
                    f"\n[Safety timeout] Stopping after Epoch {epoch}/{TOTAL_EPOCHS}. "
                    f"Elapsed: {elapsed_session / 3600:.2f}h | "
                    f"Remaining buffer: {remaining_session / 60:.1f}m | "
                    f"Est. next epoch: {est_next_epoch / 60:.1f}m"
                )
                break

    if not safety_timeout:
        print(f"\nTeacher Training Completed! Highest Val Dice achieved: {best_dice:.4f}")


if __name__ == "__main__":
    main()
