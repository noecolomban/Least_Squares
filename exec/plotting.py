#%%
from matplotlib import lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pathlib
from enum import Enum
from scheduled.schedules.wsd import WSDSchedule
from matplotlib.legend_handler import HandlerTuple, HandlerBase
from enum import Enum
from src.utils import read_dict_from_json
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D


folder = pathlib.Path(__file__).parent.resolve() / "plots"
folder.mkdir(exist_ok=True)

#DIMENSIONS = (2, 1.5)
#DIMENSIONS = (3.5, 2.5)  # Width and height in inches for LaTeX document
DIMENSIONS = (4,3)

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



# Custom handler to stack lines vertically in the legend
class VerticalLineHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        artists = []
        num_lines = len(orig_handle)
        step = height / max(num_lines, 1)
        
        for i, handle in enumerate(orig_handle):
            # Calculate the vertical position of each line (from top to bottom)
            y = height - (i + 0.5) * step - ydescent
            line = Line2D([0, width], [y, y], 
                          color=handle.get_color(),
                          linestyle=handle.get_linestyle(),
                          linewidth=handle.get_linewidth())
            artists.append(line)
        return artists


# %%

def eta_of_cooldown():
    #Dimensions = (4,3)
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    results_eta_ratio = read_dict_from_json(folder="figures", filename="eta_ratio_vs_cooldown.json")
    list_c, results_to_print = results_eta_ratio[400].keys(), {T: [results_eta_ratio[T][c] for c in results_eta_ratio[T]] for T in results_eta_ratio}
    X = [float(c) for c in list_c] 
    for T in results_to_print:
        print(f"T={T}: {results_to_print[T]}")
        plot(
            label=fr"$T={T}$",
            X=X,
            Y=results_to_print[T],
            xlabel="Cooldown Length (c)",
            ylabel=r"$\log(\widetilde\gamma^*(T;1)) - \log(\widetilde\gamma^*(T;c))$",
            filename="eta_ratio_vs_cooldown.pdf",
            schedule=ScheduleCmap.WSD,
            intensity=0.5 + 0.5 * (list(results_to_print.keys()).index(T) / max(1, len(results_to_print)-1)),
            save=False,
            close=False,
            show=False,
        )
    ax = plt.gca()
    grouped_label_1 = r"$T \in \{" + ", ".join([f"{T}" for T in list(results_to_print.keys())]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(folder / "eta_ratio_vs_cooldown.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()

def wsd(c=0.2):
    #DIMENSIONS = (2,1.5)
    from scheduled import WSDSchedule
    wsd = WSDSchedule(steps=1000, cooldown_len=c, base_lr=1)
    plot(
        X=np.arange(1000),
        Y=wsd.schedule,
        xlabel="Step",
        ylabel=r"$\eta_t$",
        filename=f"wsd_schedule_c={c}.pdf",
        schedule=ScheduleCmap.WSD,
        intensity=0.8,
        show=False,
        close=False,
        save=True,
    )
    #plt.axhline(xmin=1-c, xmax=1, y=0.5, color='r', linestyle=':', label=r'$c \times T$')

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
        label=r"Risk $\mathcal R_T$",
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
        label=r"Risk $\mathcal R_T$",
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


def asymptotics_vs_true_constant(dim=100):
    results_var_true = read_dict_from_json(folder=f"slock_constant_dim={dim}", filename="true_variance_trajectories.json")
    results_var_approx = read_dict_from_json(folder=f"slock_constant_dim={dim}", filename="variance_trajectories.json")
    results_bias_true = read_dict_from_json(folder=f"slock_constant_dim={dim}", filename="true_bias_trajectories.json")
    results_bias_approx = read_dict_from_json(folder=f"slock_constant_dim={dim}", filename="bias_trajectories.json")
    print("Results loaded for Asymptotics vs True comparison.")
    list_alphas = sorted(set(alpha for (alpha, T) in results_var_true.keys()))
    T_values = sorted(set(T for (alpha, T) in results_var_true.keys()))

    ratios_variance = {key: results_var_approx[key] / results_var_true[key] for key in results_var_true.keys()}
    ratios_bias = {key: results_bias_approx[key] / results_bias_true[key] for key in results_bias_true.keys()}

    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[ratios_variance[(alpha, T)] for T in T_values],
            xlabel=r"$T$ (log scale)",
            ylabel=r"$\widetilde V_T / V_T$",
            filename=f"variance_ratio_constant_dim={dim}.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=True,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.CONSTANT,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    ax = plt.gca()
        
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(folder / f"variance_ratio_constant_dim={dim}.pdf")
    plt.show()


    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[ratios_bias[(alpha, T)] for T in T_values],
            xlabel=r"$T$ (log scale)",
            ylabel=r"$\widetilde B_T / B_T$",
            filename=f"bias_ratio_constant_dim={dim}.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=False,
            schedule=ScheduleCmap.CONSTANT,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    ax = plt.gca()
        
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(folder / f"bias_ratio_constant_dim={dim}.pdf")
    plt.show()
                    
def asymptotics_vs_true_linear(dim=100):
    results_var_true = read_dict_from_json(folder=f"slock_linear_dim={dim}", filename="true_variance_trajectories.json")
    results_var_approx = read_dict_from_json(folder=f"slock_linear_dim={dim}", filename="variance_trajectories.json")
    results_bias_true = read_dict_from_json(folder=f"slock_linear_dim={dim}", filename="true_bias_trajectories.json")
    results_bias_approx = read_dict_from_json(folder=f"slock_linear_dim={dim}", filename="bias_trajectories.json")
    print("Results loaded for Asymptotics vs True comparison.")
    list_alphas = sorted(set(alpha for (alpha, T) in results_var_true.keys()))
    T_values = sorted(set(T for (alpha, T) in results_var_true.keys()))

    ratios_variance = {key: results_var_approx[key] / results_var_true[key] for key in results_var_true.keys()}
    ratios_bias = {key: results_bias_approx[key] / results_bias_true[key] for key in results_bias_true.keys()}

    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[ratios_variance[(alpha, T)] for T in T_values],
            xlabel=r"$T$ (log scale)",
            ylabel=r"$\widetilde V_T / V_T$",
            filename=f"variance_ratio_linear.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=False,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    ax = plt.gca()  
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(folder / f"variance_ratio_linear.pdf")
    plt.show()
    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[ratios_bias[(alpha, T)] for T in T_values],
            xlabel=r"$T$ (log scale)",
            ylabel=r"$\widetilde B_T / B_T$",
            filename=f"bias_ratio_linear.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=False,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    ax = plt.gca()  
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(folder / f"bias_ratio_linear.pdf")
    plt.show()

def asymptotics_vs_true_wsd(dim=100):
    results_var_true = read_dict_from_json(folder=f"slock_wsd_dim={dim}", filename="true_variance_trajectories.json")
    results_var_approx = read_dict_from_json(folder=f"slock_wsd_dim={dim}", filename="variance_trajectories.json")
    results_bias_true = read_dict_from_json(folder=f"slock_wsd_dim={dim}", filename="true_bias_trajectories.json")
    results_bias_approx = read_dict_from_json(folder=f"slock_wsd_dim={dim}", filename="bias_trajectories.json")
    print("Results loaded for Asymptotics vs True comparison.")
    list_alphas = sorted(set(alpha for (alpha, T) in results_var_true.keys()))
    T_values = sorted(set(T for (alpha, T) in results_var_true.keys()))

    ratios_variance = {key: results_var_approx[key] / results_var_true[key] for key in results_var_true.keys()}
    ratios_bias = {key: results_bias_approx[key] / results_bias_true[key] for key in results_bias_true.keys()}

    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[ratios_variance[(alpha, T)] for T in T_values],
            xlabel=r"$T$ (log scale)",
            ylabel=r"$\widetilde V_T / V_T$",
            filename=f"variance_ratio_wsd.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=False,
            schedule=ScheduleCmap.WSD,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    ax = plt.gca()
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(folder / f"variance_ratio_wsd.pdf")
    plt.show()
    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[ratios_bias[(alpha, T)] for T in T_values],
            xlabel=r"$T$ (log scale)",
            ylabel=r"$\widetilde B_T / B_T$",
            filename=f"bias_ratio_wsd.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=True,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.WSD,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    ax = plt.gca()    
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(folder / f"bias_ratio_wsd.pdf")
    plt.show()

def eta_optimization_constant():
    results_risks = read_dict_from_json(folder="slock_constant_dim=100", filename="risks_for_different_etas.json")
    results_eta_opt = read_dict_from_json(folder="slock_constant_dim=100", filename="optimal_etas.json")
    list_alphas = list(results_risks.keys())
    eta_values = sorted(float(eta) for eta in results_risks[list_alphas[0]].keys())
    print(results_risks)
    for alpha in list_alphas:
        plot(
            X=eta_values,
            Y=[results_risks[alpha][str(eta)] for eta in eta_values],
            xlabel=r"$\gamma$",
            ylabel="Risk",
            filename=f"eta_opt_constant.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.CONSTANT,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    for alpha in list_alphas:
        intensity = 0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1))
        plt.axvline(x=results_eta_opt[alpha], color=ScheduleCmap.CONSTANT.get_shade(intensity), linestyle='--', label=rf"$\tilde\eta^*$ for $\alpha$ = {alpha}")
    plt.legend()
    
    plt.xlim(1e-3, 1e-2)
    plt.ylim(2e-5, 6*1e-5)

    ax = plt.gca()
    ax.set_xticks([1e-3, 1e-2])
    #ax.set_yticks([1e-3, 1e-2])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    # plt.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)
    # ax.set_position([0.18, 0.15, 0.75, 0.75])
    grouped_label_1 = r"$\mathcal R_T, \alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    grouped_label_2 = r"$\widetilde \gamma^*(T), \alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    dashed_lines = [line for line in ax.lines if line.get_linestyle() in ['--', 'dashed']]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines), tuple(dashed_lines)], 
        [grouped_label_1, grouped_label_2],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(folder / "eta_opt_constant.pdf")
    plt.show()


def eta_optimization_linear():
    results_risks = read_dict_from_json(folder="slock_linear_dim=100", filename="risks_for_different_etas.json")
    results_eta_opt = read_dict_from_json(folder="slock_linear_dim=100", filename="optimal_etas.json")
    list_alphas = list(results_risks.keys())
    eta_values = sorted(float(eta) for eta in results_risks[list_alphas[0]].keys())
    print(results_risks)
    for alpha in list_alphas:
        plot(
            X=eta_values,
            Y=[results_risks[alpha][str(eta)] for eta in eta_values],
            xlabel=r"$\gamma$",
            ylabel="Risk",
            filename=f"eta_opt_linear.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
    for alpha in list_alphas:
        intensity = 0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1))
        plt.axvline(x=results_eta_opt[alpha], color=ScheduleCmap.LINEAR.get_shade(intensity), linestyle='--', label=rf"$\tilde\eta^*$ for $\alpha$ = {alpha}")
    plt.legend()
    
    plt.xlim(2e-3, 1e-1)
    plt.ylim(4e-6, 1e-4)

    ax = plt.gca()
    # ax.set_xticks([1e-3, 2e-3, 5e-3, 1e-2, 1e-1])
    # ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    # plt.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)
    # ax.set_position([0.18, 0.15, 0.75, 0.75])
    grouped_label_1 = r"$\mathcal R_T, \alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    grouped_label_2 = r"$\widetilde \gamma^*(T), \alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    dashed_lines = [line for line in ax.lines if line.get_linestyle() in ['--', 'dashed']]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines), tuple(dashed_lines)], 
        [grouped_label_1, grouped_label_2],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right"
    )
    
    plt.tight_layout()
    plt.savefig(folder / "eta_opt_linear.pdf")
    plt.show()


def slock_vs_normal_comparison_linear():
    results_variance_ratio = read_dict_from_json(folder="slock_linear_dim=100", filename="true_slock_vs_normal_variance_ratios.json")
    results_bias_ratio = read_dict_from_json(folder="slock_linear_dim=100", filename="true_slock_vs_normal_bias_ratios.json")
    list_alphas = sorted(set(float(alpha) for alpha in results_variance_ratio.keys()))
    T_values = sorted(set(int(T) for T in results_variance_ratio[list_alphas[0]].keys()))
    variance_ratio = {float(alpha): {int(T): results_variance_ratio[alpha][str(T)] for T in results_variance_ratio[alpha]} for alpha in results_variance_ratio}
    bias_ratio = {float(alpha): {int(T): results_bias_ratio[alpha][str(T)] for T in results_bias_ratio[alpha]} for alpha in results_bias_ratio}

    for alpha in list_alphas:
        plot(
            X=T_values,
            Y=[variance_ratio[alpha][T] for T in T_values],
            xlabel=r"$T$",
            ylabel=r"$V_T / V_T^{\mathrm{gauss}}$",
            filename=f"variance_ratio_slock_vs_normal.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='linear',
            marker='.',
        )
    plt.legend()
    plt.ylim(0.96, 1.005)
    ax = plt.gca()
    
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]

    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=3, # Increase the height of the legend box to fit all lines comfortably
        loc="lower right"
    )
    plt.tight_layout()

    plt.savefig(folder / "variance_ratio_slock_vs_normal.pdf",)
    plt.show()

    for alpha in list_alphas:  
        plot(
            X=T_values,
            Y=[bias_ratio[alpha][T] for T in T_values],
            xlabel=r"$T$",
            ylabel=r"$B_T / B_T^{\mathrm{gauss}}$",
            filename=f"bias_ratio_slock_vs_normal.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='linear',
            marker='.',
        )
    plt.legend()
    plt.ylim(0.9, 1.01)

    ax = plt.gca()
        # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=3, # Increase the height of the legend box to fit all lines comfortably
        loc="lower right"
    )
    plt.tight_layout()

    plt.savefig(folder / "bias_ratio_slock_vs_normal.pdf",
                )
    plt.show()

def cooldown_length_comparing_at_eta_star(d=100):
    T = 100000
    results_risks = read_dict_from_json(folder=f"slock_wsd_dim={d}", filename="true_slock_risks_cooldown.json")
    list_cooldown = [float(c) for c in results_risks.keys()]
    print(results_risks)
    list_alphas = [float(alpha) for alpha in results_risks[list_cooldown[0]].keys()]
    print(f"Loaded risks for cooldown lengths at T={T} for alphas: {list_alphas}")
    risks = {float(c): {float(alpha): results_risks[c][alpha] for alpha in results_risks[c]} for c in results_risks}
    print(f"Loaded risks for cooldown lengths for alphas: {list_alphas}")
    for alpha in list_alphas:
        plot(
            X=list_cooldown,
            Y=[risks[c][alpha] for c in list_cooldown],
            xlabel=r"Cooldown Length ($c$)",
            ylabel="Risk",
            filename=f"risk_vs_cooldown_length.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.WSD,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='linear',
            yscale='linear',
            marker='.',
        )
    ax = plt.gca()
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(folder / "risk_vs_cooldown_length.pdf",)
    plt.show()

def cooldown_length_comparing_fixed_eta(d=100):
    T = 100000
    results_risks = read_dict_from_json(folder=f"slock_wsd_dim={d}", filename="risks_slock_cooldown.json")
    list_cooldown = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(results_risks)
    list_alphas = [float(alpha) for alpha in results_risks.keys()]
    print(f"Loaded risks for cooldown lengths at T={T} for alphas: {list_alphas}")
    risks = results_risks
    print(f"Loaded risks for cooldown lengths for alphas: {list_alphas}")
    for alpha in list_alphas:
        plot(
            X=list_cooldown,
            Y=risks[alpha],
            xlabel=r"Cooldown Length ($c$)",
            ylabel="Risk",
            filename=f"risk_vs_cooldown_length_fixed_eta.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.WSD,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='linear',
            yscale='linear',
            marker='.',
        )
    ax = plt.gca()
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(folder / "risk_vs_cooldown_length_fixed_eta.pdf",)
    plt.show()

def batch_bT_constant():
    bT = 200000
    results_biases = read_dict_from_json(folder="slock_linear_dim=100", filename=f"BATCH_comparison_bias_bT={bT}_eta_star_for_each_batch.json")
    results_variances = read_dict_from_json(folder="slock_linear_dim=100", filename=f"BATCH_comparison_variance_bT={bT}_eta_star_for_each_batch.json")
    biases = {int(batch): {float(alpha): results_biases[batch][alpha] for alpha in results_biases[batch]} for batch in results_biases}
    variances = {int(batch): {float(alpha): results_variances[batch][alpha] for alpha in results_variances[batch]} for batch in results_variances}
    print("Results loaded for Batch Size vs Risk comparison.")
    batches = sorted(set(int(batch) for batch in biases.keys()))
    list_alphas = sorted(set(float(alpha) for alpha in biases[batches[0]].keys()))
    for alpha in list_alphas:
        Y0 = {alpha: biases[1][alpha] + variances[1][alpha] for alpha in list_alphas}
        plot(
            X=batches,
            Y=[(biases[batch][alpha] + variances[batch][alpha]) / Y0[alpha] for batch in batches],
            xlabel=r"Batch Size $b$",
            ylabel=r"$\mathcal R_{\lfloor N/b \rfloor}^{(b)} / \mathcal R_{N}^{(1)}$",
            filename=f"batch_risk_comparison.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=True,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (list_alphas.index(alpha) / max(1, len(list_alphas)-1)),
            xscale='log',
            yscale='linear',
            marker='.',
        )
    plt.xlim(1, 10000)
    plt.ylim(0.995, 1.1)
    ax = plt.gca()
    
    grouped_label_1 = r"$\alpha \in \{" + ", ".join([f"{alpha}" for alpha in list_alphas]) + r"\}$"
    
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(list_alphas)]
    dashed_lines = [line for line in ax.lines if line.get_linestyle() in ['--', 'dashed']]

    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines)], 
        [grouped_label_1],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(folder / "batch_risk_comparison.pdf",
                )
    plt.show()

def steps_to_fixed_risk():
    # Risk = 1e-3
    results_steps = read_dict_from_json(folder="slock_linear_dim=100", filename=f"steps_to_risk.json")
    results_b_crit = read_dict_from_json(folder="slock_linear_dim=100", filename=f"critical_batches.json")

    # alphas is a list of floats
    alphas = sorted(list(float(alpha) for alpha in results_steps.keys()))

    # Sort items numerically and use float(alpha) as the dictionary key to match the loop below
    steps_X = {float(alpha): [int(k) for k, v in sorted(results_steps[alpha].items(), key=lambda item: int(item[0]))] for alpha in results_steps}
    steps_Y = {float(alpha): [v for k, v in sorted(results_steps[alpha].items(), key=lambda item: int(item[0]))] for alpha in results_steps}
    
    # Fix X and Y mapping: X should be the batch size (keys), Y should be T (values)
    b_crit_X = {float(alpha): [float(k) for k, v in sorted(results_b_crit[alpha].items(), key=lambda item: float(item[0]))] for alpha in results_b_crit}
    b_crit_Y = {float(alpha): [v for k, v in sorted(results_b_crit[alpha].items(), key=lambda item: float(item[0]))] for alpha in results_b_crit}

    for alpha in alphas:
        plot(
            X=steps_X[alpha],
            Y=steps_Y[alpha],
            xlabel=r"Batch Size $b$",
            ylabel=r"Steps to reach $\mathcal R_T \leq 10^{-3}$",
            filename=f"steps_to_fixed_risk.pdf",
            label=rf"$\alpha$ = {alpha}",
            save=False,
            show=False,
            close=False,
            legend=False,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (alphas.index(alpha) / max(1, len(alphas)-1)),
            xscale='log',
            yscale='log',
            marker='.',
        )
        plot(
            X=b_crit_X[alpha],
            Y=b_crit_Y[alpha],
            xlabel=r"Batch Size $b$",
            ylabel=r"$T$",
            filename=f"critical_batch_size.pdf",
            label="",
            save=False,
            show=False,
            close=False,
            legend=False,
            schedule=ScheduleCmap.LINEAR,
            intensity=0.5 + 0.5 * (alphas.index(alpha) / max(1, len(alphas)-1)),
            xscale='log',
            yscale='log',
            marker='',
            linestyle='--',
            linewidth=1.
        )
    plt.xlim(500, 100000)
    plt.ylim(100, 10000)
    ax = plt.gca()

    grouped_label_1 = r"$T$ to reach $\mathcal R_T^{(b)} = r$, $\alpha \in \{" + ", ".join([f"{alpha}" for alpha in alphas]) + r"\}$"
    grouped_label_2 = r"$b_{\mathrm{max}}$, $\alpha \in \{" + ", ".join([f"{alpha}" for alpha in alphas]) + r"\}$"
    
    solid_lines = [line for line in ax.lines if line.get_linestyle() in ['-', 'solid']]
    solid_lines = solid_lines[:len(alphas)]
    dashed_lines = [line for line in ax.lines if line.get_linestyle() in ['--', 'dashed']]

    # Create the custom legend using HandlerTuple to combine the lines horizontally
    ax.legend(
        [tuple(solid_lines), tuple(dashed_lines)], 
        [grouped_label_1, grouped_label_2],
        handler_map={tuple: VerticalLineHandler()},
        handleheight=2.5, # Increase the height of the legend box to fit all lines comfortably
        loc="upper right"
    )
    plt.tight_layout()
    plt.savefig(folder / "steps_to_fixed_risk.pdf")
    plt.show()

if __name__ == "__main__":
            
    #wsd(c=0.4)
    #asymptotics_vs_true_constant()
    #sgd_vs_formula_constant()
    #sgd_vs_formula_linear()
    #slock_vs_normal_comparison_linear()
    #asymptotics_vs_true_constant()
    #asymptotics_vs_true_linear()
    #asymptotics_vs_true_wsd()
    #eta_optimization_constant()
    #eta_optimization_linear()
    #cooldown_length_comparing_at_eta_star()
    eta_of_cooldown()
    #slock_vs_normal_comparison_linear()
    #cooldown_length_comparing_at_eta_star()
    #cooldown_length_comparing_fixed_eta()
    #batch_bT_constant()
    #steps_to_fixed_risk()

# %%
