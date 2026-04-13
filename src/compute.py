from .SGD import SGD, NoisyGD
import numpy as np 
from scheduled.schedules.base import ScheduleBase
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib
from datetime import datetime
import os

class Computations:
    """
    Class for computing and plotting risks for different SGD schedules.
    
    Automatically generates distinct colors for each schedule using matplotlib's tab10 colormap.
    Colors are assigned consistently based on schedule order and remain the same across runs.
    """
    def __init__(self, model, schedules: list[ScheduleBase], schedules_names: list[str] | None = None, sgd_class=SGD):
        self.model = model
        self.schedules = schedules
        self.schedules_names = schedules_names if schedules_names is not None else [schedule.name for schedule in schedules]
        self.sgd_class = sgd_class
        self.class_name = sgd_class.name
        
        # Ensure images directory exists
        os.makedirs('images', exist_ok=True)
        
        # Generate consistent colors for each schedule automatically
        self.schedule_colors = self._generate_schedule_colors()
    
    def _generate_schedule_colors(self):
        """Generate distinct colors for each schedule automatically."""
        
        # Use a colormap to generate distinct colors
        colormap = matplotlib.colormaps.get_cmap('tab10')  # 10 distinct colors
        colors = {}
        
        for i, schedule in enumerate(self.schedules):
            # Use schedule index to get consistent color from colormap
            color_rgba = colormap(i % 10)
            # Convert to hex format
            color_hex = mcolors.to_hex(color_rgba)
            colors[self.schedules_names[i]] = color_hex
            
        return colors
    
    def _get_schedule_color(self, schedule):
        """Get consistent color for a schedule."""
        return self.schedule_colors.get(schedule, '#7f7f7f')  # default gray

    def compute_risk(self, x0, i_schedule=0):
        """Compute theoretical risk trajectory for a single schedule."""
        sgd = self.sgd_class(self.model, x0, self.schedules[i_schedule])
        return sgd.compute_all_theoretical_risks()

    def compute_mean_empirical_risk(self, x0, n_runs=5, i_schedule=0):
        """Compute the mean empirical risk trajectory for a single schedule."""
        sgd = self.sgd_class(self.model, x0, self.schedules[i_schedule])
        risks = []  
        for i in range(n_runs):
            risk = sgd.train(show=False)
            risks.append(risk)
        return np.mean(np.array(risks), axis=0)
    
    def compute_all_empirical_risks(self, x0, n_runs=5, plot=False, log_scale=False):
        """Compute mean empirical risk trajectories for all schedules."""
        all_risks = {}
        for i, schedule in enumerate(self.schedules):
            sgd = self.sgd_class(self.model, x0, schedule)
            risks = []  
            for _ in range(n_runs):
                risk = sgd.train(show=False)
                risks.append(risk)
            all_risks[self.schedules_names[i]] = np.mean(np.array(risks), axis=0)

        if plot:
            for schedule_name, risk in all_risks.items():
                color = self._get_schedule_color(schedule_name)
                plt.plot(risk, label=schedule_name, color=color)
            plt.xlabel("Epoch")
            plt.ylabel("Empirical Risk")
            plt.title(f"Mean Empirical Risk over Epochs ({self.class_name})")
            if log_scale:
                plt.yscale('log')
            plt.legend()
            plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_empirical_risks.pdf")
            plt.show()
        
        return all_risks
    
    def compute_all_theoretical_risks(self, x0, plot=False, log_scale=False):
        """Compute theoretical risk trajectories for all schedules."""
        all_risks = {}
        for i, schedule in enumerate(self.schedules):
            sgd = self.sgd_class(self.model, x0, schedule)
            risk = sgd.compute_all_theoretical_risks()
            all_risks[self.schedules_names[i]] = risk

        if plot:
            for schedule_name, risk in all_risks.items():
                color = self._get_schedule_color(schedule_name)
                plt.plot(risk, label=schedule_name, color=color)
            plt.xlabel("Epoch")
            plt.ylabel("Theoretical Risk")
            plt.title(f"Theoretical Risk over Epochs ({self.class_name})")
            if log_scale:
                plt.yscale('log')
            plt.legend()
            plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_theoretical_risks.pdf")
            plt.show()
        
        return all_risks

    def compute_all_risks(self, x0, n_runs=5, plot=False, log_scale=False):
        """Compute both empirical and theoretical risks for all schedules."""
        empirical_risks = self.compute_all_empirical_risks(x0, n_runs=n_runs, plot=False)
        theoretical_risks = self.compute_all_theoretical_risks(x0, plot=False)
        
        if plot:
            for schedule_name in empirical_risks.keys():
                color = self._get_schedule_color(schedule_name)
                plt.plot(empirical_risks[schedule_name], label=f"{schedule_name} Empirical", 
                        color=color, linestyle='solid')
                plt.plot(theoretical_risks[schedule_name], label=f"{schedule_name} Theoretical", 
                        color=color, linestyle='dashed')
            
            plt.xlabel("Epoch")
            plt.ylabel("Risk")
            plt.title(f"Empirical vs Theoretical Risk over Epochs ({self.class_name})")
            if log_scale:
                plt.yscale('log')
            plt.legend()
            plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_risks.pdf")
            plt.show()
        return empirical_risks, theoretical_risks
    
    