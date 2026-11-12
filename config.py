import os
import random

import numpy as np
import torch


def get_paths(dataset_type, fold=4, user_data_dir=None, user_teacher_weights=None):
    # Data directories
    if user_data_dir:
        data_dir = user_data_dir
    else:
        # Local fallback
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if dataset_type == "brats2023":
            data_dir = os.path.join(base_dir, "data", "brats2023", "Training")
        else:
            data_dir = os.path.join(base_dir, "data", "msd", "Task01_BrainTumour")

    # Output directories
    teacher_save_dir = f"./teacher_results_{dataset_type}_fold_{fold}"
    student_save_dir = f"./kd_results_{dataset_type}_fold_{fold}"

    # Teacher weights path
    if user_teacher_weights:
        teacher_weights = user_teacher_weights
    else:
        teacher_weights = os.path.join(teacher_save_dir, "best_teacher.pth")

    return {
        "data_dir": data_dir,
        "teacher_save_dir": teacher_save_dir,
        "student_save_dir": student_save_dir,
        "teacher_weights": teacher_weights,
        "is_kaggle": os.path.exists("/kaggle/input")
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }


def set_rng_state(state):
    if state is None:
        return False

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state, but CUDA is unavailable.")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from checkpoint.")
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])

    return True


def save_checkpoint(checkpoint, path):
    """Write to a temporary file, then atomically replace the previous checkpoint."""
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
