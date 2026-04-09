import numpy as np

from scheduled.schedules.base import ScheduleBase

class PolynomialSchedule(ScheduleBase):

    def __init__(self,
                steps: int=100,
                base_lr: float=1.0,
                exponent: float=0.5,
                **kwargs
    ):
        
        self._exponent = exponent

        super().__init__(steps=steps,
                         base_lr=base_lr,
                         cooldown_kwargs=None,
                         **kwargs
        )

    def _construct_main_schedule(self):
        return np.array([(t+1)**(-self._exponent) for t in range(self._steps)])
    
    @property
    def name(self):
        return 'polynomial'
