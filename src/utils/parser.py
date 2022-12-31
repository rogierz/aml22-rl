import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--step", required=True, help="The number of the step to execute", type=int)
parser.add_argument(
    "--n", help="Split of the gridsearch to execute (1, 2 or 3)", default=0, type=int)
