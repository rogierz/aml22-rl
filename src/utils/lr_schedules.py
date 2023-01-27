"""
This file holds the functions which handle the different learning rate schedules
"""

from typing import Callable


STEP = 0.5


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Wrapper for the no-LR-schedule

    :param initial_value: float: The initial value of the learning rate
    :return: A function that returns a constant value (param initial_value)
    """
    def func(progress_remaining: float) -> float:
        """
        This function returns the initial value of the progress bar.

        :param progress_remaining:float: The training progress remaining
        """
        return initial_value

    return func


def step_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Wrapper for a step learning rate schedule, i.e. the LR value over the progress will be a piecewise constant function.
    The initial value is multiplied by the global variable STEP (the decay factor) every 30% of the progress.

    :param initial_value:float: Set the initial value of the learning rate
    :return: A function that returns the current LR based on the progress remaining
    """
    def func(progress_remaining: float) -> float:
        if progress_remaining >= 0.7:
            # 1 <= pr <= 0.7 : just started
            return initial_value
        elif progress_remaining >= 0.4:
            # 0.7 < pr <= 0.4
            return initial_value * STEP
        elif progress_remaining >= 0.1:
            # 0.4 < pr <= 0.1
            return initial_value * STEP**2
        else:
            # 0.1 < pr <= 0 : last 10% of the progress
            return initial_value * STEP**3

    return func


LR_SCHEDULES = {"constant": constant_schedule, "step": step_schedule}
