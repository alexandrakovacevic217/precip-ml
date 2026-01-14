import argparse

from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loaded config:", args.config)
    print("Target type:", cfg["target"]["type"])
    print("Variables:", cfg["inputs"]["variables"])
    print("N input PCs:", cfg["inputs"]["n_pcs"])


if __name__ == "__main__":
    main()


