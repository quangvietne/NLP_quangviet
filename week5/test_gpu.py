import torch

print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test tensor trên GPU
    x = torch.rand(5, 3).cuda()
    print(f"Tensor on GPU:\n{x}")
else:
    print("CUDA không khả dụng!")
print("="*50)