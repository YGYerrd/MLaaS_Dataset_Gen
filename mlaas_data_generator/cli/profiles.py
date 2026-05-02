# cli/profiles.py

DATASET_CHOICES = [
    "fashion_mnist",
    "mnist",
    "cifar10",
    "digits",
    "iris",
    "wine",
    "california_housing",
    "diabetes",
    "hf",          
]

def infer_dataset_profile(dataset: str) -> dict:
    ds = dataset.lower()


    if ds in {"mnist", "fashion_mnist"}:
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "is_image": True,
            "default_model_pretty": "CNN",
            "ask_split": False,
            "split_key": None,
            "ask_scaler": False,
            "ask_target_scaler": False,
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
        }

    if ds == "cifar10":
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "is_image": True,
            "default_model_pretty": "MobileNetV2",
            "ask_split": False,
            "split_key": None,
            "ask_scaler": False,
            "ask_target_scaler": False,
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
        }


    if ds in {"iris", "wine", "digits"}:
        return {
            "default_task": "classification",
            "tasks_supported": ["classification", "clustering"],
            "is_image": False,
            "default_model_pretty": "MLP",
            "ask_split": True,
            "split_key": "test_size",
            "ask_scaler": True,
            "ask_target_scaler": False,
            "allow_label_splits": True,
            "allow_quantity_skew": True,
            "allow_custom": True,
        }


    if ds in {"california_housing", "diabetes"}:
        return {
            "default_task": "regression",
            "tasks_supported": ["regression", "clustering"],
            "is_image": False,
            "default_model_pretty": "Random Forest",
            "ask_split": True,
            "split_key": "test_size",
            "ask_scaler": True,
            "ask_target_scaler": True,
            "allow_label_splits": False,
            "allow_quantity_skew": True,
            "allow_custom": False,
        }


    return {
        "default_task": "classification",
        "tasks_supported": ["classification", "clustering"],
        "is_image": False,
        "default_model_pretty": "MLP",
        "ask_split": False,
        "split_key": None,
        "ask_scaler": False,
        "ask_target_scaler": False,
        "allow_label_splits": True,
        "allow_quantity_skew": True,
        "allow_custom": True,
    }
