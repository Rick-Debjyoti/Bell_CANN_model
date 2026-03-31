"""Google Colab setup for Bell CANN project.

Run this cell first in Colab to install dependencies and mount Drive.

Colab provides:
  - Free tier: T4 GPU (15GB VRAM) — ~3-4x faster than RTX 3050 Laptop
  - Pro: A100 (40GB) or V100 (16GB) — ~10-20x faster
  - TF 2.15+ pre-installed, but we need specific TFP version

Usage in Colab:
    !git clone https://github.com/[YOUR_REPO]/Bell_CANN_model.git
    %cd Bell_CANN_model
    !python src2/python/colab_setup.py
    %cd src2/python
    !python run_optuna_improved.py
"""

import subprocess
import sys


def setup():
    packages = [
        "tensorflow-probability>=0.23", 
        "optuna>=4.0",
        "seaborn>=0.13",
    ]

    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Verify
    import tensorflow as tf
    import tensorflow_probability as tfp
    import optuna

    print(f"TensorFlow:  {tf.__version__}")
    print(f"TF-Prob:     {tfp.__version__}")
    print(f"Optuna:      {optuna.__version__}")
    print(f"GPU:         {tf.config.list_physical_devices('GPU')}")
    print(f"\nSetup complete! cd src2/python and run scripts.")


if __name__ == "__main__":
    setup()
