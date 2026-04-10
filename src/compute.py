from .SGD import SGD
import numpy as np 
from scheduled.schedules.base import ScheduleBase

class Simulations:
    def __init__(self, model, schedules: list[ScheduleBase]):
        self.model = model
        self.schedules = schedules

    def compute_risk(self, x0, i_schedule=0):
        """Compute theoretical risk trajectory for a single schedule."""
        sgd = SGD(self.model, x0, self.schedules[i_schedule])
        return sgd.compute_all_theoretical_risks()

    def compute_mean_empirical_risk(self, x0, n_runs=5, i_schedule=0):
        """Compute the mean empirical risk trajectory for a single schedule."""
        sgd = SGD(self.model, x0, self.schedules[i_schedule])
        risks = []  
        for i in range(n_runs):
            risk = sgd.train(show=False)
            risks.append(risk)
        return np.mean(np.array(risks), axis=0)
    
    def compute_all_empirical_risks(self, x0, n_runs=5, plot=False):
        """Compute mean empirical risk trajectories for all schedules."""
        all_risks = {}
        for schedule in self.schedules:
            sgd = SGD(self.model, x0, schedule)
            risks = []  
            for _ in range(n_runs):
                risk = sgd.train(show=False)
                risks.append(risk)
            all_risks[schedule.name] = np.mean(np.array(risks), axis=0)

        if plot:
            import matplotlib.pyplot as plt
            for schedule_name, risk in all_risks.items():
                plt.plot(risk, label=schedule_name)
            plt.xlabel("Epoch")
            plt.ylabel("Empirical Risk")
            plt.title("Mean Empirical Risk over Epochs")
            plt.yscale('log')
            plt.legend()
            plt.show()
        
        return all_risks
    
    def compute_all_theoretical_risks(self, x0, plot=False):
        """Compute theoretical risk trajectories for all schedules."""
        all_risks = {}
        for schedule in self.schedules:
            sgd = SGD(self.model, x0, schedule)
            risk = sgd.compute_all_theoretical_risks()
            all_risks[schedule.name] = risk

        if plot:
            import matplotlib.pyplot as plt
            for schedule_name, risk in all_risks.items():
                plt.plot(risk, label=schedule_name)
            plt.xlabel("Epoch")
            plt.ylabel("Theoretical Risk")
            plt.title("Theoretical Risk over Epochs")
            plt.yscale('log')
            plt.legend()
            plt.show()
        
        return all_risks
    