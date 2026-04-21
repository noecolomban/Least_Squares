import json
from datetime import datetime
import os
import numpy as np


def save_optimization_results(results, filename=None):
    """Save optimization results to a file in JSON format."""
    if filename is None:
        filename = f"optimize_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    path = os.path.join("saved_files", "base_lr", filename)
    with open(path, 'w') as file:
        json.dump(clean_keys_json(results), file, indent=4)
    print(f"Optimization results saved to {path}")



def restore_integer_keys(obj):
    """Traverse the dictionary and convert numeric string keys to int."""
    if isinstance(obj, dict):
        nouveau_dict = {}
        for k, v in obj.items():
            # Try to convert the key to an integer
            try:
                nouvelle_cle = int(k)
            except ValueError:
                # If it fails (e.g., the key is a word like "results"), keep as text
                nouvelle_cle = k
            
            # Recursively apply the function for nested dictionaries
            nouveau_dict[nouvelle_cle] = restore_integer_keys(v)
        return nouveau_dict
    elif isinstance(obj, list):
        return [restore_integer_keys(element) for element in obj]
    else:
        return obj


def read_optimization_results(filename):
    """Read optimization results from a file and return a structured dictionary."""
    path = os.path.join("saved_files", "base_lr", filename)
    
    with open(path, 'r') as file:
        # 1. Load the JSON (numeric keys are strings "1", "2"...)
        read_file = json.load(file)
        
        # 2. Restore keys as integers (1, 2...) where appropriate
        result = restore_integer_keys(read_file)
        print(f"Optimization results loaded from {path}")
        
    return result


def clean_keys_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_keys_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_keys_json(element) for element in obj]
    else:
        return obj


def save_risk_results(results: np.ndarray, filename=None):
    """Save risk computation results to a file in JSON format."""
    if filename is None:
        filename = f"risk_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    
    # Ensure the directory exists
    os.makedirs(os.path.join("saved_files", "risks"), exist_ok=True)
    path = os.path.join("saved_files", "risks", filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(path, 'w') as file:
        json.dump(convert_numpy(results), file, indent=4)
    print(f"Risk results saved to {path}")
    return path


def read_risk_results(filename) -> np.ndarray:
    """Read risk results from a file and return a structured dictionary of numpy arrays."""
    path = os.path.join("saved_files", "risks", filename)
    
    with open(path, 'r') as file:
        data = json.load(file)
        
    # Convert lists back to numpy arrays
    def convert_to_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj)
        if isinstance(obj, dict):
            return {k: convert_to_numpy(v) for k, v in obj.items()}
        return obj

    result = convert_to_numpy(data)
    print(f"Risk results loaded from {path}")
    return result




if __name__ == "__main__":
    read_file = read_optimization_results("optimize_results_13-04-2023_15-30-45.json")  
    print(read_file)