import yaml
from pathlib import Path

def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist.")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg
