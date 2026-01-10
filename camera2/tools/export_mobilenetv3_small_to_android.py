import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export MobileNetV3-Small binary classifier weights (.pth state_dict) "
            "to a PyTorch Lite module (.ptl) for Android."
        )
    )
    parser.add_argument(
        "--weights",
        default="forest_fire_classifier_mobilenetv3_small.pth",
        help="Path to the .pth state_dict weights file.",
    )
    parser.add_argument(
        "--output",
        default="app/src/main/assets/forest_fire_classifier_mobilenetv3_small.ptl",
        help="Path to write the exported .ptl file.",
    )
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.mobile_optimizer import optimize_for_mobile
        from torchvision import models
    except Exception as e:
        raise SystemExit(
            "Missing Python deps. Create a venv and install torch+torchvision.\n"
            "Example:\n"
            "  python3.11 -m venv .venv-ml\n"
            "  ./.venv-ml/bin/pip install 'numpy<2' torch==2.1.0 torchvision==0.16.0\n"
        ) from e

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.zeros(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)
    optimized = optimize_for_mobile(traced)

    optimized._save_for_lite_interpreter(str(output_path))
    print(str(output_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
