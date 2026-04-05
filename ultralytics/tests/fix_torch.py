import os
import ctypes

# Manually point to the problematic DLL
# Change the path below to match your actual .venv location from the screenshot
dll_path = r"C:\Users\ADMIN\Desktop\CV surveillance detection\ultralytics\.venv\Lib\site-packages\torch\lib\c10.dll"

try:
    ctypes.CDLL(dll_path)
    import torch
    print("✅ Torch loaded successfully!")
except Exception as e:
    print(f"❌ Still failing: {e}")