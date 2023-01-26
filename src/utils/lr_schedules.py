from typing import Callable


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Wrapper for no LR schedule
    """
    def func(progress_remaining: float) -> float:
        return initial_value

    return func


def step_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Wrapper for step LR schedule
    """
    def func(progress_remaining: float) -> float:
        if progress_remaining >= 0.7:
            # 1 <= pr <= 0.7 : just started
            return initial_value
        elif progress_remaining >= 0.4:
            # 0.7 < pr <= 0.4
            return initial_value / 2
        elif progress_remaining >= 0.1:
            # 0.4 < pr <= 0.1
            return initial_value / 4
        else:
            # 0.1 < pr <= 0 : last 10% of the progress
            return initial_value / 8

    return func


LR_SCHEDULES = {"constant": constant_schedule, "step": step_schedule}