#!/usr/bin/env python

from pprint import pprint

from src.steps.step2_2 import main as step2_2
from src.steps.step2_3 import main as step2_3
from src.steps.step3_GS import main as step3
from src.steps.step4 import main as step4
from src.utils.parser import parser

STEPS = {
    "2_2": step2_2,
    "2_3": step2_3,
    "3": step3,
    "4": step4
}

if __name__ == '__main__':
    args = parser.parse_args()
    pprint(args)
    step = STEPS[args.step]

    # execute step
    step(args.base_prefix, bool(args.force))
