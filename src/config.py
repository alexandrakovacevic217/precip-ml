import yaml


def load_config(path: str) -> dict:
    """Load an experiment config from a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
