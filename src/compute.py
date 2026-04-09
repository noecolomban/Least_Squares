from .SGD import SGD
import numpy as np 


class Simulations:
    def __init__(self, sim, schedule):
        self.sim = sim
        self.schedule = schedule

    def compute_risk(self, x0):
        sgd = SGD(self.sim, x0, self.schedule)
        return np.array([sgd.compute_theoretical_risk(t) for t in range(sgd.T)])

    def compute_mean_empirical_risk(self, x0, n_runs=5):
        sgd = SGD(self.sim, x0, self.schedule)
        risks = []  
        for i in range(n_runs):
            risk = sgd.train(show=False)
            risks.append(risk)
        return np.mean(np.array(risks), axis=0)