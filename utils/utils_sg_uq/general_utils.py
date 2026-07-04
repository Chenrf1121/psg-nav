import torch
import time
import numpy as np
from datetime import datetime
import uuid
    
def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)
    
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds")
        return result  # Return the result of the function call
    return wrapper

def generate_run_id(suffix: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    uid = str(uuid.uuid4())[:4]
    return f"{timestamp}_{uid}_{suffix}"

def read_run_id(id_path: str):
    with open(id_path, "r") as f:
        run_id = f.read().strip()
    return run_id

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)