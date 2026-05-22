#%%
from cProfile import label
import json
import pathlib
from scipy.special import zeta

#%%
def save_optimization_results(*args, **kwargs):
    pass

def read_optimization_results(*args, **kwargs):
    pass

def save_dict_to_json(d: dict, folder: str, filename: str):
    path = pathlib.Path("saved_files") / folder
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    with open(file_path, 'w') as f:
        json.dump(d, f)
        print(f"Saved dictionary to {file_path}")

def read_dict_from_json(folder: str, filename: str) -> dict:
    file_path = pathlib.Path("saved_files") / folder / filename
    with open(file_path, 'r') as f:
        d = json.load(f)
        print(f"Read dictionary from {file_path}")
    return d

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

