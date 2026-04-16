from .risk_computations import RiskComputations
from .SGD import SGD
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import os
from datetime import datetime

class Visualization:
    def __init__(self, schedules, schedules_name=None):
        if schedules_name is None:
            schedules_name = [schedule.name for schedule in schedules]
        self.schedules = {name: schedule for name, schedule in zip(schedules_name, schedules)}
        self.schedules_names = schedules_name
        self.colors = self._generate_schedule_colors()

    def _generate_schedule_colors(self):
        """Generate distinct colors for each schedule automatically."""
        
        # Use a colormap to generate distinct colors
        colormap = matplotlib.colormaps.get_cmap('tab10')  # 10 distinct colors
        colors = {}
        
        for name in self.schedules_names:
            # Use schedule index to get consistent color from colormap
            color_rgba = colormap(list(self.schedules.keys()).index(name) % 10)
            # Convert to hex format
            color_hex = mcolors.to_hex(color_rgba)
            colors[name] = color_hex
        return colors
    
    def _make_filename(self, text):
        """Generate a filename for saving plots based on the given text."""
        return os.path.join("images", f"{text}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")

    def plot_for_every_schedule(self, values: dict, title="Risk Trajectories for Different Schedules", legend=True, savefig=False, logscale=False, filename=None):
        """Plot the given values for every schedule with distinct colors."""
        plt.figure(figsize=(10, 6))
        for name in self.schedules_names:
            plt.plot(values[name], label=name, color=self.colors[name])
        plt.xlabel("Time Step")
        plt.ylabel("Risk")
        plt.title(title)
        if legend: 
            plt.legend()
        plt.grid(True)
        if logscale:
            plt.yscale('log')
        if savefig:
            if filename is None:
                filename = "Risk_Trajectories"
            plt.savefig(self._make_filename(filename))
        plt.show()

    def plot_comparison(self, theoretical_values, empirical_values, title="Theoretical vs Empirical Risk", legend=True, savefig=False, logscale=False, filename=None):
        """Plot theoretical and empirical values for each schedule."""
        plt.figure(figsize=(10, 6))
        for name in self.schedules_names:
            plt.plot(theoretical_values[name], label=f"{name} - Theoretical", color=self.colors[name], linestyle='--')
            plt.plot(empirical_values[name], label=f"{name} - Empirical", color=self.colors[name])
        plt.xlabel("Time Step")
        plt.ylabel("Risk")
        plt.title(title)
        if legend: 
            plt.legend()
        if logscale:
            plt.yscale('log')
        plt.grid(True)
        if savefig:
            if filename is None:
                filename = "Theoretical_vs_Empirical_Risk"
            plt.savefig(self._make_filename(filename))
        plt.show()


    import matplotlib.pyplot as plt


    def plot_optimization_at_several_ts(self, results, legend=True, plot_etas=False, savefig=False, logscale=False, filename=None):
        """Plot optimization results at several time steps."""
        # Création des figures : 2 colonnes si plot_etas est True, sinon 1 seule
        if plot_etas:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_risk = axes[0]
            ax_eta = axes[1]
        else:
            fig, ax_risk = plt.subplots(1, 1, figsize=(10, 6))
            ax_eta = None

        for name in self.schedules_names:
            t_values = list(results.keys())
            risks = [results[t][name]["min_risk"] for t in t_values]
            etas = [results[t][name]["best_eta"] for t in t_values]
            
            # Tracé du risque sur le premier graphique
            ax_risk.plot(t_values, risks, label=name, color=self.colors[name], marker='o')
            
            # Tracé des etas sur le deuxième graphique si demandé
            if plot_etas:
                ax_eta.plot(t_values, etas, label=f"{name}", color=self.colors[name], linestyle='--', marker='x')

        ax_risk.set_xlabel("Time Step")
        ax_risk.set_ylabel("Optimized Risk")
        ax_risk.set_title("Optimized Risk at Several Time Steps")
        ax_risk.grid(True)
        if logscale:
            ax_risk.set_yscale('log')
        if legend:
            ax_risk.legend()

        if plot_etas:
            ax_eta.set_xlabel("Time Step")
            ax_eta.set_ylabel("Best Eta")
            ax_eta.set_title("Best Eta at Several Time Steps")
            ax_eta.grid(True)
            if logscale:
                ax_eta.set_yscale('log') # Applique aussi le logscale si pertinent
            if legend:
                ax_eta.legend()

        plt.tight_layout()

        if savefig:
            if filename is None:
                filename = "Optimized_Risk_at_Several_Time_Steps"
            plt.savefig(self._make_filename(filename))
            
        plt.show()


    def plot_sgd_classes_comparison(self, risks_class1, risks_class2, label_class1="True SGD", label_class2="Noisy GD", title="Comparison: SGD vs Noisy GD", legend=True, savefig=False, logscale=False, filename=None):
        """
        risks_class1 et risks_class2 doivent être des dictionnaires : {nom_schedule: tableau_des_risques}
        """
        plt.figure(figsize=(10, 6))
        
        for name in self.schedules_names:
            if name in risks_class1 and name in risks_class2:
                # Ligne continue pour la première classe (ex: SGD classique)
                plt.plot(risks_class1[name], label=f"{name} - {label_class1}", color=self.colors[name], linestyle='-')
                
                # Ligne pointillée pour la seconde classe (ex: NoisyGD)
                plt.plot(risks_class2[name], label=f"{name} - {label_class2}", color=self.colors[name], linestyle='--')
                
        plt.xlabel("Time Step")
        plt.ylabel("Risk")
        plt.title(title)
        
        if legend: 
            plt.legend()
        if logscale:
            plt.yscale('log')
        plt.grid(True)
        
        if savefig:
            if filename is None:
                filename = "Comparison_SGD_Classes"
            # Utilise la fonction _make_filename existante pour sauvegarder
            plt.savefig(self._make_filename(filename))
            
        plt.show()