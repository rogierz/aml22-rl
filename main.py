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
from src.steps.step3 import \
    main as step3v0, \
    main_interval as step3v1, \
    main_relative as step3v2, \
    main_absolute as step3v3
from src.steps.step4 import \
    main as step4v0, \
    main_udr as step4v1, \
    main_no_udr as step4v2
from src.steps.step4_1 import main as step4_1
from src.steps.step4_2 import \
    main as step4_2v0, \
    main_nature_cnn as step4_2v1, \
    main_custom_cnn as step4_2v2, \
    main_custom_cnn_pretrained as step4_2v3
from src.utils.parser import parser

STEPS = {
    "2_2": [step2_2],
    "2_3": [step2_3],
    "3": [step3v0, step3v1, step3v2, step3v3],
    "4": [step4v0, step4v1, step4v2],
    "4_1": [step4_1],
    "4_2": [step4_2v0, step4_2v1, step4_2v2, step4_2v3]
}

if __name__ == '__main__':
    args = parser.parse_args()
    step = STEPS[args.step][args.v]

    # execute step
    step(args.base_prefix, bool(args.force))
