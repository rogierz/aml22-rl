import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--step", required=True, help="The number of the step to execute", type=str)
parser.add_argument(
    "--n", help="Split of the gridsearch to execute (1, 2 or 3)", default=0, type=int)
parser.add_argument("--t", help="The number of trials for the optimization study (suggested values: 100-1000) (default "
                                "is 10)", default=10, type=int)

parser.add_argument("--logdir", help="The directory of log files", default="logs", type=str, dest="base_prefix")
