#%%
from cProfile import label
import json
import pathlib
import ast
from scipy.special import zeta

#%%
def save_optimization_results(*args, **kwargs):
    pass

def read_optimization_results(*args, **kwargs):
    pass


def convert_key(key):
    # Return immediately if the key is already a number or a tuple
    if isinstance(key, (int, float, tuple)):
        return key
    
    if isinstance(key, str):
        # Attempt to convert to an integer
        try:
            return int(key)
        except ValueError:
            pass
        
        # Attempt to convert to a float
        try:
            return float(key)
        except ValueError:
            pass
        
        # Attempt to evaluate the string as a tuple
        try:
            parsed_val = ast.literal_eval(key)
            # Ensure the evaluated value is a tuple and contains only numbers
            if isinstance(parsed_val, tuple) and all(isinstance(x, (int, float)) for x in parsed_val):
                return parsed_val
        except (ValueError, SyntaxError):
            # Catches strings that cannot be evaluated, like normal words
            pass

    return key

def save_dict_to_json(d: dict, folder: str, filename: str):
    path = pathlib.Path("saved_files") / folder
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    with open(file_path, 'w') as f:
        json.dump(d, f,indent=4, ensure_ascii=False)
        print(f"Saved dictionary to {file_path}")

def read_dict_from_json(folder: str, filename: str) -> dict:
    file_path = pathlib.Path("saved_files") / folder / filename
    with open(file_path, 'r') as f:
        d = json.load(f)
        dict_new_keys = {convert_key(key): value for key, value in d.items()}
        print(f"Read dictionary from {file_path}")
    return dict_new_keys




#
def constant_zeta_correction(alpha):
    assert alpha > 1, "Alpha should be greater than 1."
    if alpha <= 2:
        return 1
    else:
        return zeta(alpha/2)*(alpha/2 - 1)
    
#%%
if __name__ == "__main__":
    # import numpy as np
    # import matplotlib.pyplot as plt
    # alphas = np.linspace(2, 5, 100)
    # plt.plot(alphas, [constant_zeta_correction(alpha) for alpha in alphas], label="Zeta Correction")
    # plt.plot(alphas, alphas/2, label="Alpha/2")
    # plt.xlabel("Alpha")
    # plt.ylabel("Zeta Correction")
    # plt.title("Zeta Correction as a function of Alpha")
    # plt.legend()
    # plt.grid()
    # plt.show()

    print("2:", constant_zeta_correction(2))
    print("2.5:", constant_zeta_correction(2.5))
    print("3.5:", constant_zeta_correction(3.5))
    print("4.5:", constant_zeta_correction(4.5))
# %%


