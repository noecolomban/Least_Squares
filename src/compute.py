from src.SGD import SGD
import numpy as np 
from scheduled.schedules.base import ScheduleBase
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from datetime import datetime
import os
from src.utils import save_optimization_results, read_optimization_results

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

    def _make_sgd(self, x0, schedule):
        return self.sgd_class(self.model, x0, schedule)

    def _plot_risks(self, risks, title, ylabel, filename_suffix, log_scale=False, legend=True, line_styles=None):
        if not risks:
            return

        plt.figure(figsize=(8, 5))
        for idx, (schedule_name, risk) in enumerate(risks.items()):
            color = self._get_schedule_color(schedule_name)
            linestyle = line_styles[idx] if line_styles and idx < len(line_styles) else 'solid'
            plt.plot(risk, label=schedule_name, color=color, linestyle=linestyle)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title} ({self.class_name})")
        if log_scale:
            plt.yscale('log')
        if legend:
            plt.legend()
        plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename_suffix}.pdf")
        plt.show()
        plt.close()

    def compute_risk(self, x0, i_schedule=0):
        """Compute theoretical risk trajectory for a single schedule."""
        return self._make_sgd(x0, self.schedules[i_schedule]).compute_all_theoretical_risks()

    def compute_mean_empirical_risk(self, x0, n_runs=5, i_schedule=0):
        """Compute the mean empirical risk trajectory for a single schedule."""
        risks = [self._make_sgd(x0, self.schedules[i_schedule]).train(show=False) for _ in range(n_runs)]
        return np.mean(np.array(risks), axis=0)

    def compute_all_empirical_risks(self, x0, n_runs=5, plot=False, log_scale=False, legend=True):
        """Compute mean empirical risk trajectories for all schedules."""
        all_risks = {
            name: np.mean(
                np.array([self._make_sgd(x0, schedule).train(show=False) for _ in range(n_runs)]),
                axis=0
            )
            for name, schedule in zip(self.schedules_names, self.schedules)
        }

        if plot:
            self._plot_risks(
                all_risks,
                title="Mean Empirical Risk over Epochs",
                ylabel="Empirical Risk",
                filename_suffix="empirical_risks",
                log_scale=log_scale,
                legend=legend
            )

        return all_risks

    def compute_all_theoretical_risks(self, x0, plot=False, log_scale=False, legend=True):
        """Compute theoretical risk trajectories for all schedules."""
        all_risks = {
            name: self._make_sgd(x0, schedule).compute_all_theoretical_risks()
            for name, schedule in zip(self.schedules_names, self.schedules)
        }

        if plot:
            self._plot_risks(
                all_risks,
                title="Theoretical Risk over Epochs",
                ylabel="Theoretical Risk",
                filename_suffix="theoretical_risks",
                log_scale=log_scale,
                legend=legend
            )

        return all_risks

    def approx_all_theoretical_risks(self, x0, plot=False, log_scale=False, legend=True):
        """Compute approximate theoretical risk trajectories for all schedules."""
        all_risks = {
            name: self._make_sgd(x0, schedule).approx_all_theoretical_risks()
            for name, schedule in zip(self.schedules_names, self.schedules)
        }

        if plot:
            self._plot_risks(
                all_risks,
                title="Approximate Theoretical Risk over Epochs",
                ylabel="Theoretical Risk",
                filename_suffix="approx_theoretical_risks",
                log_scale=log_scale,
                legend=legend
            )

        return all_risks

    def compute_approx_vs_theoretical_risks(self, x0, plot=False, log_scale=False, legend=True):
        """Compute both theoretical and approximate risk trajectories for all schedules.
        approximation is lambda lambda^T = Lambda^2
        """

        theoretical_risks = self.compute_all_theoretical_risks(x0, plot=False, legend=False)
        approx_risks = self.approx_all_theoretical_risks(x0, plot=False, legend=False)

        if plot:
            plt.figure(figsize=(8, 5))
            for schedule_name in theoretical_risks:
                color = self._get_schedule_color(schedule_name)
                plt.plot(theoretical_risks[schedule_name], label=f"{schedule_name} Theoretical", color=color, linestyle='solid')
                plt.plot(approx_risks[schedule_name], label=f"{schedule_name} Approximate", color=color, linestyle='dashed')

            plt.xlabel("Epoch")
            plt.ylabel("Risk")
            plt.title(fr"Theoretical vs Approximate ($\lambda\lambda^T \leftarrow \Lambda^2$) Risk over Epochs ({self.class_name})")
            if log_scale:
                plt.yscale('log')
            if legend:
                plt.legend()
            plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_theoretical_vs_approx_risks.pdf")
            plt.show()
            plt.close()

        return theoretical_risks, approx_risks

    def compute_all_risks(self, x0, n_runs=5, plot=False, log_scale=False, legend=True):
        """Compute both empirical and theoretical risks for all schedules."""
        empirical_risks = self.compute_all_empirical_risks(x0, n_runs=n_runs, plot=False)
        theoretical_risks = self.compute_all_theoretical_risks(x0, plot=False)

        if plot:
            plt.figure(figsize=(8, 5))
            for schedule_name in empirical_risks:
                color = self._get_schedule_color(schedule_name)
                plt.plot(empirical_risks[schedule_name], label=f"{schedule_name} Empirical", color=color, linestyle='solid')
                plt.plot(theoretical_risks[schedule_name], label=f"{schedule_name} Theoretical", color=color, linestyle='dashed')

            plt.xlabel("Epoch")
            plt.ylabel("Risk")
            plt.title(f"Empirical vs Theoretical (with this model) Risk over Epochs ({self.class_name})")
            if log_scale:
                plt.yscale('log')
            if legend:
                plt.legend()
            plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_risks.pdf")
            plt.show()
            plt.close()

        return empirical_risks, theoretical_risks

    def _evaluate_eta(self, x0, i_schedule, eta, t_value):
        schedule = self.schedules[i_schedule]
        schedule.set_base_lr(eta)
        return self.compute_risk(x0, i_schedule=i_schedule)[t_value]

    def optimize_base_lr(self, x0, t_value=None, i_schedule=0, eta_range=None, plot=True, change_eta=True):
        if eta_range is None:
            eta_range = np.logspace(-4, 0.5, 50)
        eta_range = np.asarray(eta_range)

        schedule = self.schedules[i_schedule]
        original_lr = schedule.get_base_lr()

        if t_value is None:
            t_value = schedule._steps - 1
        assert 0 <= t_value < schedule._steps, "t_value must be a valid schedule index."

        try:
            final_risks = [self._evaluate_eta(x0, i_schedule, eta, t_value) for eta in eta_range]
        finally:
            schedule.set_base_lr(original_lr)

        best_idx = int(np.argmin(final_risks))
        best_eta = eta_range[best_idx]
        min_risk = final_risks[best_idx]

        if change_eta:
            schedule.set_base_lr(best_eta)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.loglog(eta_range, final_risks, marker='.', label=f'Risk at step $T={t_value}$')
            plt.axvline(best_eta, color='red', linestyle='--', label=fr'Optimum: $\eta \approx {best_eta:.2e}$')
            plt.xlabel(r'Base Learning Rate ($\eta$)')
            plt.ylabel(r'Final Risk $\mathbb{E}[f(x_T) - f^*]$')
            plt.title(fr'$\eta$ optimization for schedule={self.schedules_names[i_schedule]}')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.show()
            plt.close()

        return best_eta, min_risk

    def optimize_all_base_lrs(self, x0, t_value=None, eta_range=None, plot=True, change_eta=True, save_results=True):
        schedule_results = {}
        for i, name in enumerate(self.schedules_names):
            best_eta, min_risk = self.optimize_base_lr(
                x0,
                t_value=t_value,
                i_schedule=i,
                eta_range=eta_range,
                plot=plot,
                change_eta=change_eta
            )
            schedule_results[name] = {"best_eta": best_eta, "min_risk": min_risk}
        if save_results:
            save_optimization_results(schedule_results, additional_info=self.class_name, filename=f"optimize_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        return schedule_results

    def _plot_series(self, x_values, series, xlabel, ylabel, title, filename_suffix, log_scale=False):
        plt.figure(figsize=(8, 5))
        for name, values in series.items():
            plt.plot(x_values, values, marker='o', label=name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title} ({self.class_name})")
        plt.legend()
        plt.grid(True, ls='-', alpha=0.5)
        plt.savefig(f"images/{self.class_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename_suffix}.pdf")
        if log_scale:
            plt.yscale('log')
        plt.show()
        plt.close()

    def optimize_at_several_ts(self, x0, t_values, eta_range=None, plot=True, change_eta=True, log_scale=False, save_results=True):
        assert all(0 <= t < self.schedules[0]._steps for t in t_values), "All t values must be valid schedule indices."
        results = {}
        for t in t_values:
            schedule_results = self.optimize_all_base_lrs(
                x0,
                t_value=t,
                eta_range=eta_range,
                plot=False,
                change_eta=change_eta,
                save_results=False
            )
            results[t] = schedule_results

        if plot:
            risk_series = {name: [results[t][name]["min_risk"] for t in t_values] for name in self.schedules_names}
            eta_series = {name: [results[t][name]["best_eta"] for t in t_values] for name in self.schedules_names}

            self._plot_series(
                t_values,
                risk_series,
                xlabel=r'Time Steps ($t$)',
                ylabel='Minimum Risk',
                title=r'Minimum Risk at optimal $\eta$ for different $t$ values',
                filename_suffix='min_risk_vs_t_values',
                log_scale=log_scale
            )

            self._plot_series(
                t_values,
                eta_series,
                xlabel=r'Time Steps ($t$)',
                ylabel=r'Optimal Base Learning Rate ($\eta$)',
                title=r'Optimal $\eta$ at different $t$ values',
                filename_suffix='optimum_eta_vs_t_values'
            )
        if save_results:
            save_optimization_results(results, additional_info=self.class_name, filename=f"optimize_results_at_several_ts_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        return results

    def adapt_eta_from_file(self, filename, several_ts=False):
        results = read_optimization_results(filename)
        if several_ts:
            max_t = max(results.keys())
            state_optimization = results[max_t]
        else:            
            state_optimization = results
        for schedule_name, schedule in zip(self.schedules_names, self.schedules):
            if schedule_name in state_optimization:
                best_eta = state_optimization[schedule_name]["best_eta"]
                schedule.set_base_lr(best_eta)
                print(f"Updated {schedule_name} with optimal eta: {best_eta:.2e}")
            else:
                print(f"No results found for {schedule_name} in the file.")


if __name__ == "__main__":
    results = read_optimization_results(r"saved_files/optimize_results_13-04-2023_15-30-45.json")
    print(results)