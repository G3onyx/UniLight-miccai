import os


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
