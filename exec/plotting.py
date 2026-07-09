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

DIMENSIONS = (1.95, 1.56)

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


def plot(X, Y, xlabel, ylabel, filename, label="", show=False, close=True, schedule: ScheduleCmap | None = None):
    if schedule:
        plt.plot(X, Y, linestyle='-', color=schedule.get_shade(intensity=0.8), label=label)
    else:
        plt.plot(X, Y, linestyle='-', color='blue', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(folder / filename, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close()

def plots(X, Y_dict, xlabel, ylabel, filename, show=False, close=True):
    for label, Y in Y_dict.items():
        plt.plot(X, Y, marker='.', linestyle='-', label=label, color = ScheduleCmap.WSD.get_shade(intensity=0.2 + 0.6 * (list(Y_dict.keys()).index(label) / max(1, len(Y_dict)-1))))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
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
            show=False,
            close=False
        )
        #plt.axhline(xmin=1-c, xmax=1, y=0.5, color='r', linestyle=':', label=r'$c \times T$')
        plt.legend()
    wsd(c=0.4)
# %%
