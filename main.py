#!/usr/bin/env python

from src.utils.parser import parser
from pprint import pprint
from src.steps.step2_2 import main as step2_2
from src.steps.step2_3 import main as step2_3
from src.steps.step3 import main as step3

STEPS = {
    "2_2": step2_2,
    "2_3": step2_3,
    "3": step3
}

if __name__ == '__main__':
    args = parser.parse_args()
    pprint(args)
    step = STEPS[args.step]

    # execute step
    step(args)
