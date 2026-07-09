#%%
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from enum import Enum
from scheduled.schedules.wsd import WSDSchedule
from enum import Enum
from src.utils import read_dict_from_json

folder = pathlib.Path(__file__).parent.resolve() / "plots"
folder.mkdir(exist_ok=True)

DIMENSIONS = (4, 3)

plt.rcParams.update({
    "text.usetex": True,                   # Use LaTeX to write all text
    "font.family": "serif",                # Use serif fonts
    "font.serif": ["Computer Modern"],     # LaTeX's default font
    "axes.labelsize": 11,                  # Match your LaTeX document font size (e.g., 11pt)
    "font.size": 11,                       # Base font size
    "legend.fontsize": 9,                  # Slightly smaller for legends
    "xtick.labelsize": 9,                  # Tick labels
    "ytick.labelsize": 9,
    "figure.figsize": DIMENSIONS,          # Figure size in inches (match LaTeX \textwidth)
    "pgf.texsystem": "pdflatex",           # Use pdflatex for processing
    "pgf.rcfonts": False,                  # Don't setup fonts from rc parameters
})

class ScheduleCmap(Enum):
    # Associate each schedule to a built-in Matplotlib colormap
    CONSTANT = "Blues"
    LINEAR   = "Oranges"
    WSD      = "Greens"

    def get_shade(self, intensity: float):
        """
        Get a specific shade from the colormap.
        Intensity must be a float between 0.0 (lightest) and 1.0 (darkest).
        """
        assert 0.0 <= intensity <= 1.0, "Intensity must be between 0.0 and 1.0"
        # Fetch the colormap object from matplotlib
        cmap = plt.get_cmap(self.value)
        # Return the RGBA color code for the requested intensity
        return cmap(intensity)
    
    def __call__(self, intensity: float):
        return self.get_shade(intensity)

def plot(X, Y, xlabel, ylabel, filename, legend=False, label="", save=False, show=False, close=True, schedule: ScheduleCmap | None = None, intensity=0.8, xscale='linear', yscale='linear', **kwargs):
    if schedule:
        plt.plot(X, Y,  color=schedule.get_shade(intensity=intensity), label=label, **kwargs)
    else:
        plt.plot(X, Y,  color='blue', label=label, **kwargs)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    if legend:
        plt.legend()
    if save:
        plt.savefig(folder / filename, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close()

def plots(X, Y_dict, xlabel, ylabel, filename, save=False, show=False, close=True, schedule: ScheduleCmap | None = None):
    for label, Y in Y_dict.items():
        color = schedule.get_shade(intensity=0.2 + 0.6 * (list(Y_dict.keys()).index(label) / max(1, len(Y_dict)-1))) if schedule else 'blue'
        plt.plot(X, Y, marker='.', linestyle='-', label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    if save:
        plt.savefig(folder / filename, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close()



# %%
if __name__ == "__main__":
    def eta_of_cooldown():
        import matplotlib.ticker as ticker
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        results_eta_ratio = read_dict_from_json(folder="figures", filename="eta_ratio_vs_cooldown.json")
        list_c, results_to_print = results_eta_ratio[400].keys(), {T: [results_eta_ratio[T][c] for c in results_eta_ratio[T]] for T in results_eta_ratio}
        plots(
            X=list_c,
            Y_dict=results_to_print,
            xlabel="Cooldown Length (c)",
            ylabel="log(eta_star(1) / eta_star(c))",
            filename="eta_ratio_vs_cooldown.pdf",
            save=True,
            show=True
        )
        plt.show()

    def wsd(c=0.2):
        from scheduled import WSDSchedule
        wsd = WSDSchedule(steps=1000, cooldown_len=c, base_lr=1)
        plot(
            X=np.arange(1000),
            Y=wsd.schedule,
            xlabel="Step",
            ylabel="Learning Schedule",
            filename=f"wsd_schedule_c={c}.pdf",
            schedule=ScheduleCmap.WSD,
            label=r"$\eta_t / \eta$",
            intensity=0.8,
            show=False,
            close=False
        )
        #plt.axhline(xmin=1-c, xmax=1, y=0.5, color='r', linestyle=':', label=r'$c \times T$')
        plt.legend()

    def sgd_vs_formula_constant():
        results = read_dict_from_json(folder="slock_experiment_dim=100", filename="losses_and_risks_alpha=1.5_beta=2_L=0.1_Delta=1_sigma=0.1.json")
        print("Results loaded for SGD vs Computed Risk comparison.")
        sgd_values = {int(T): results["sgd"][T] for T in results["sgd"].keys()}
        true_values = {int(T): results["true"][T] for T in results["true"].keys()}
        print(f"SGD values: {list(sgd_values.items())[:5]} ...")
        print(f"True values: {list(true_values.items())[:5]} ...")
               
        plot(
            X=list(sgd_values.keys()),
            Y=list(sgd_values.values()),
            xlabel="Step",
            ylabel="Loss",
            filename=f"sgd_vs_formula_constant.pdf",
            label="SGD Loss",
            save=False,
            show=False,
            close=False,
            schedule=ScheduleCmap.CONSTANT,
            intensity=0.5,
            linewidth=2
        )
        plot(
            X=list(true_values.keys()),
            Y=list(true_values.values()),
            xlabel="Step",
            ylabel="Loss / Risk",
            filename=f"sgd_vs_formula_constant.pdf",
            label="Computed Risk",
            xscale='log',
            yscale='log',
            save=True,
            show=True,
            close=True,
            legend=True,
            schedule=ScheduleCmap.CONSTANT,
            intensity=1.0,
            linewidth=1,
            linestyle='--',
            marker='.',
        )
    
    def sgd_vs_formula_linear():
        results = read_dict_from_json(folder="slock_experiment_dim=100", filename="LINEAR_losses_and_risks_alpha=1.5_beta=2_L=0.1_Delta=1_sigma=0.1.json")
        print("Results loaded for SGD vs Computed Risk comparison.")
        sgd_values = {}
        for T in results["sgd"].keys():
            sgd_values[int(T)] = {int(t): results["sgd"][T][t] for t in results["sgd"][T].keys()}
        true_values = {int(T): results["true"][T] for T in results["true"].keys()}
        print(f"SGD values: {list(sgd_values.items())[:5]} ...")
        print(f"True values: {list(true_values.items())[:5]} ...")
        
        intensities = np.linspace(0.4, 0.8, len(sgd_values))**2

        for i,T in enumerate(sgd_values.keys()):
            plot(
                X=list(sgd_values[T].keys()),
                Y=list(sgd_values[T].values()),
                xlabel="Step",
                ylabel="Loss",
                filename=f"sgd_vs_formula_linear_T={T}.pdf",
                label=f"SGD Loss" if T == max(sgd_values.keys()) else None,
                save=False,
                show=False,
                close=False,
                schedule=ScheduleCmap.LINEAR,
                intensity=intensities[i],  # Scale intensity based on T
                linewidth=2,
            )
        plot(
            X=list(true_values.keys()),
            Y=list(true_values.values()),
            xlabel="Step",
            ylabel="Loss / Risk",
            filename=f"sgd_vs_formula_linear.pdf",
            label="Computed Risk",
            xscale='log',
            yscale='log',
            save=True,
            show=True,
            close=True,
            legend=True,
            schedule=ScheduleCmap.LINEAR,
            intensity=1.0,
            linewidth=1,
            linestyle='--',
            marker='.',
        )


    sgd_vs_formula_linear()

# %%
