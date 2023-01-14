#!/usr/bin/env python

from src.utils.parser import parser
from pprint import pprint
from src.steps.step2a import main as step2a
from src.steps.step3 import main as step3

STEPS = {
    2: step2a,
    3: step3
}

if __name__ == '__main__':
    args = parser.parse_args()
    pprint(args)
    step = STEPS[args.step]

    # execute step
    step(args)
