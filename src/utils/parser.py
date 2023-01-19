import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--step", required=True, help="The number of the step to execute", type=str)
parser.add_argument("--logdir", help="The directory of log files", default="logs", type=str, dest="base_prefix")
