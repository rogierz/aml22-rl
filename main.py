import sys

from run.steps.step2a import main as step2a
from run.steps.step3 import main as step3

BASE_PREFIX = "./run/logs"


def main(args):
    if len(args) > 1:
        raise RuntimeWarning("Expected only one argument. Ignoring all the others.")

    step = args[0]

    if step == '--2':
        step2a(base_prefix=BASE_PREFIX)
    elif step == '--3':
        # TODO: cfg parameter
        step3('', base_prefix=BASE_PREFIX)


if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
