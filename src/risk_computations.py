from src.SGD import BaseSGD, SGD
from src.least_squares import LinearRegression
import numpy as np 
from datetime import datetime
from scheduled.schedules.base import ScheduleBase
from src.utils import save_optimization_results, read_optimization_results

class Risk:
    def __init__(self, model: LinearRegression, x0: np.ndarray, schedules: list[ScheduleBase], schedules_names: list[str] | None = None, sgd_class=SGD):
        self.model = model
        self.x0 = x0
        if schedules_names is None:
            schedules_names = [schedule.name for schedule in schedules]
        self.schedules_names = schedules_names
        self.schedules = {name: schedule for name, schedule in zip(schedules_names, schedules)}
        self.sgd_class = sgd_class
        self.sgds = {name: self._make_sgd(schedule) for name, schedule in zip(schedules_names, schedules)}
        self.final_T = max(schedule._steps for schedule in schedules)

    def _make_sgd(self, schedule) -> BaseSGD:
        return self.sgd_class(self.model, self.x0, schedule)

    def compute_risk(self, name=None):
        """Compute theoretical risk trajectory for a single schedule."""
        if name is None:
            name = self.schedules_names[0]
        return self.sgds[name].compute_all_theoretical_risks()

    def compute_mean_empirical_risk(self, n_runs=5, name=None):
        """Compute the mean empirical risk trajectory for a single schedule."""
        if name is None:
            name = self.schedules_names[0]
        risks = [self.sgds[name].train(show=False) for _ in range(n_runs)]
        return np.mean(np.array(risks), axis=0)
    
    def compute_all_empirical_risks(self, n_runs=5):
        """Compute mean empirical risk trajectories for all schedules."""
        all_risks = {
            name: np.mean(
                np.array([self.sgds[name].train(show=False) for _ in range(n_runs)]),
                axis=0
            )
            for name in self.schedules_names}
        return all_risks
    
    def compute_all_theoretical_risks(self):
        """Compute theoretical risk trajectories for all schedules."""
        all_risks = {
            name: self.sgds[name].compute_all_theoretical_risks()
            for name in self.schedules_names
        }
        return all_risks
    
    def approx_all_theoretical_risks(self):
        """Compute approximate theoretical risk trajectories for all schedules."""
        all_risks = {
            name: self.sgds[name].approx_all_theoretical_risks()
            for name in self.schedules_names
        }
        return all_risks
    
    def compute_approx_vs_theoretical_risks(self):
        """Compute both theoretical and approximate theoretical risk trajectories for all schedules."""
        theoretical_risk = self.compute_all_theoretical_risks()
        approx_risk = self.approx_all_theoretical_risks()
        return {"theoretical": theoretical_risk, "approximate": approx_risk}
    
    def compute_all_risks(self, n_runs=5):
        """Compute both empirical and theoretical risks for all schedules."""
        theoretical_risk = self.compute_all_theoretical_risks()
        empirical_risk = self.compute_all_empirical_risks(n_runs=n_runs)
        return {"theoretical": theoretical_risk, "empirical": empirical_risk}
    
    def _evaluate_eta(self, name, eta, t_value):
        """Evaluate the learning rate at a specific time step for a given schedule."""
        schedule = self.schedules[name]
        schedule.set_base_lr(eta)
        return self.compute_risk(name=name)[t_value]
    
    def optimize_base_lr(self, t_value=None, name=None, eta_range=None, change_eta=True):
        if eta_range is None:
            eta_range = np.logspace(-4, 0.5, 50)
        eta_range = np.asarray(eta_range)

        if name is None:
            name = self.schedules_names[0]
        schedule = self.schedules[name]
        original_lr = schedule.get_base_lr()

        if t_value is None:
            t_value = schedule._steps - 1
        assert 0 <= t_value < schedule._steps, "t_value must be a valid schedule index."

        try:
            final_risks = [self._evaluate_eta(name, eta, t_value) for eta in eta_range]
        finally:
            schedule.set_base_lr(original_lr)

        best_idx = int(np.argmin(final_risks))
        best_eta = eta_range[best_idx]
        min_risk = final_risks[best_idx]

        if change_eta:
            schedule.set_base_lr(best_eta)

        return best_eta, min_risk
    
    def _get_file_name(self, text):
        return f"optimize_results_{self.sgd_class.name}_{text}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    def optimize_all_base_lrs(self, t_value=None, eta_range=None, change_eta=True, save_results=True):
        schedule_results = {}
        for name in self.schedules_names:
            best_eta, min_risk = self.optimize_base_lr(
                t_value=t_value,
                name=name,
                eta_range=eta_range,
                change_eta=change_eta
            )
            schedule_results[name] = {"best_eta": best_eta, "min_risk": min_risk}
        if save_results:
            if t_value is None:
                t_value = "final"
            self.last_optimization_file = self._get_file_name(f"t_value_{t_value}")
            save_optimization_results(schedule_results, filename=self.last_optimization_file)
        return schedule_results
    
    def optimize_at_several_ts(self, t_values, eta_range=None, change_eta=True, save_results=True):
        assert all(0 <= t < self.final_T for t in t_values), "All t values must be valid schedule indices."
        
        if eta_range is None:
            eta_range = np.logspace(-4, 0.5, 50)
        eta_range = np.asarray(eta_range)

        results = {t: {} for t in t_values}

        # Boucle principale par schedule
        for name in self.schedules_names:
            schedule = self.schedules[name]
            original_lr = schedule.get_base_lr()
            
            # Stockera les trajectoires complètes : shape = (len(eta_range), steps)
            all_trajectories = []

            # 1. On calcule TOUTES les trajectoires une seule fois par valeur de eta
            for eta in eta_range:
                schedule.set_base_lr(eta)
                full_trajectory = self.compute_risk(name=name)
                all_trajectories.append(full_trajectory)
            
            all_trajectories = np.array(all_trajectories)

            # 2. Pour chaque t demandé, on cherche le meilleur eta très rapidement
            for t in t_values:
                risks_at_t = all_trajectories[:, t]
                best_idx = int(np.argmin(risks_at_t))
                
                results[t][name] = {
                    "best_eta": eta_range[best_idx],
                    "min_risk": risks_at_t[best_idx]
                }
            
            # 3. Restauration ou mise à jour du LR
            if change_eta:
                # Si on change le eta, on prend généralement celui optimisant le plus grand t (la fin)
                max_t = max(t_values)
                schedule.set_base_lr(results[max_t][name]["best_eta"])
            else:
                schedule.set_base_lr(original_lr)
            
        if save_results:
            filename = self._get_file_name("several_ts")
            self.last_optimization_file = filename
            save_optimization_results(results, filename=filename)

        return results

    def adapt_eta_from_file(self, filename=None, several_ts=False):
        if filename is None:
            filename = self.last_optimization_file
        results = read_optimization_results(filename)
        if several_ts:
            max_t = max(results.keys())
            state_optimization = results[max_t]
        else:            
            state_optimization = results
        for schedule_name, schedule in self.schedules.items():
            if schedule_name in state_optimization:
                best_eta = state_optimization[schedule_name]["best_eta"]
                schedule.set_base_lr(best_eta)
                print(f"Updated {schedule_name} with optimal eta: {best_eta:.2e}")
            else:
                print(f"No results found for {schedule_name} in the file.")

