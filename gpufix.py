import torch
import subprocess
import platform
import os

def check_torch_cuda():
    print("=== 🧠 PyTorch GPU Diagnostic ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (torch): {torch.version.cuda}")
    print(f"cuDNN version (torch): {torch.backends.cudnn.version()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
    else:
        print("❌ GPU not detected or PyTorch is not built with CUDA.")

def check_tensorflow_gpu():
    try:
        import tensorflow as tf
        print("\n=== 🔍 TensorFlow GPU Diagnostic ===")
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print("GPUs Detected:", gpus if gpus else "❌ None")
    except ImportError:
        print("TensorFlow is not installed.")

def run_nvidia_smi():
    print("\n=== 📊 NVIDIA-SMI Output ===")
    try:
        result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        print(result)
    except Exception as e:
        print("❌ Could not run nvidia-smi. Make sure NVIDIA drivers are installed and system PATH is set.")
        print(f"Error: {e}")

def check_env_vars():
    print("\n=== ⚙️ Environment Variables Check ===")
    cuda_path = os.environ.get("CUDA_PATH", "❌ Not Set")
    print(f"CUDA_PATH: {cuda_path}")
    
    path = os.environ.get("PATH", "")
    cu_found = any("cudnn" in p.lower() for p in path.split(";"))
    print(f"cuDNN in PATH: {'✅ Yes' if cu_found else '❌ No'}")

def test_model_device():
    print("\n=== 🚦 Model Device Sanity Test ===")
    class DummyModel(torch.nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x): return x

    model = DummyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Tensor is on device: {dummy_input.device}")

if __name__ == "__main__":
    check_torch_cuda()
    check_tensorflow_gpu()
    run_nvidia_smi()
    check_env_vars()
    test_model_device()
