import importlib, traceback
import b12x
print("b12x", b12x.__file__)
try:
    m = importlib.import_module("vllm.model_executor.layers.quantization.exl3")
    print("exl3 import OK", [n for n in dir(m) if "Config" in n or "Method" in n][:8])
except Exception:
    traceback.print_exc()
