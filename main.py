#!/usr/bin/env python
"""
This file is the access point for the project code.

usage: parser.py [-h] --step STEP [--logdir BASE_PREFIX] [-f]

optional arguments:
  -h, --help            show this help message and exit
  --step STEP           The number of the step to execute
  --logdir BASE_PREFIX  The directory of log files
  -f, --force           Force the execution and overwrite the previous logs

"""


from src.steps.step2_2 import main as step2_2
from src.steps.step2_3 import main as step2_3
from src.steps.step3 import main as step3
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
    step = STEPS[args.step]

    # execute step
    step(args.base_prefix, bool(args.force))
