# app/config.py
import os
import yaml
from dotenv import load_dotenv
from types import SimpleNamespace

def load_config(config_path: str = "config.yaml") -> SimpleNamespace:
    """
    Load config.yaml + .env into a single cfg object.
    """

    # Load environment variables
    load_dotenv()

    # Load YAML config
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Inject env vars (HF_TOKEN etc.)
    cfg_dict["env"] = {
        "HF_TOKEN": os.getenv("HF_TOKEN", None)
    }

    def dict_to_ns(d):
        if isinstance(d, dict):
            # Recursively convert all dict values
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            # If the object is a list, apply conversion to each element
            return [dict_to_ns(x) for x in d]
        else:
            # Base case: if not a dict or list, return value as-is
            return d
        
    # Convert the loaded config dict into a nested namespace object
    return dict_to_ns(cfg_dict)


if __name__ == "__main__":
    cfg = load_config()

    print("Config loaded:")
    print(f"Project: {cfg.project.name}")
    print(f"Seed: {cfg.project.seed}")
    print(f"Train CSV: {cfg.data.train_csv}")
    print(f"Val CSV:   {cfg.data.val_csv}")
    print(f"Test CSV:  {cfg.data.test_csv}")
    print(f"Backbone:  {cfg.model.backbone}")
    print(f"Device:    {cfg.compute.device}")
    print(f"HF_Token:  {'Found' if cfg.env.HF_TOKEN else 'Missing'}")

# Remember to export the HF_TOKEN first, otherwise, it will crash!
# Run python3 app/config.py
# It worked and was quick
# Next, is eda.ipynb
