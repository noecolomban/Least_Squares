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
    """Parcourt le dictionnaire et convertit les clés numériques (str) en int."""
    if isinstance(obj, dict):
        nouveau_dict = {}
        for k, v in obj.items():
            # Tente de convertir la clé en entier
            try:
                nouvelle_cle = int(k)
            except ValueError:
                # Si ça plante (ex: la clé est un mot comme "resultats"), on garde le texte
                nouvelle_cle = k
            
            # On applique la fonction récursivement pour les dictionnaires imbriqués
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
        # 1. On charge le JSON (les clés numériques sont des chaînes de caractères "1", "2"...)
        read_file = json.load(file)
        
        # 2. On restaure les clés en entiers (1, 2...) là où c'est approprié
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





if __name__ == "__main__":
    read_file = read_optimization_results("optimize_results_13-04-2023_15-30-45.json")  
    print(read_file)