import psutil
import importlib.util
import os

spec = importlib.util.spec_from_file_location('backend', os.path.join(os.getcwd(), 'backend.py'))
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
proc = psutil.Process()
print(proc.memory_info().rss)
print(proc.memory_info().rss / (1024*1024))
