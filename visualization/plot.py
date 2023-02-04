#!/usr/bin/env python
"""
This script generates the plots based on the data in the `data` folder.

usage: plot.py [-h] -i INPUT_FILE (-s | -f) [-o OUTPUT_FOLDER] [-n FNAME]
               {tpe,gs}

positional arguments:
  {tpe,gs}              What you want to plot

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        The csv file containing the data to plot
  -s, --show            Show the plots in a window
  -f, --file            Save the plots in a png file (see param --fname)
  -o OUTPUT_FOLDER, --output-dir OUTPUT_FOLDER
                        The output folder
  -n FNAME, --fname FNAME
                        The output file name
"""

from argparse import ArgumentParser

from plotters.plot_step3_GS_results import main as plot_step3_gs
from plotters.plot_step3_TPE_results import main as plot_step3_tpe

PLOTTERS = {
    "tpe": plot_step3_tpe,
    "gs": plot_step3_gs
}


def main(p_name, args):
    """
    The main function of this script. It retrieves the correct plotter based on the command line argument and calls it.

    :param p_name: Determine which plotter to use
    :param args: The arguments dictionary passed by the parser
    """
    plotter = PLOTTERS[p_name]
    plotter(**args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("data_to_plot", help="What you want to plot", type=str, choices=PLOTTERS.keys())
    parser.add_argument("-i", "--input-file", required=True, help="The csv file containing the data to plot", type=str,
                        dest="input_file")
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("-s", "--show", action="count", help="Show the plots in a window")
    group1.add_argument("-f", "--file", action="count", help="Save the plots in a png file (see param --fname)")
    parser.add_argument("-o", "--output-dir", help="The output folder", type=str, default="plots", dest="output_folder")
    parser.add_argument("-n", "--fname", help="The output file name", type=str, default="plot", dest="fname")
    parser.add_argument("-m", "--smoothing", help="The smoothing weight", type=float, default=0)

    # parser.print_help()

    # extract arguments as dictionary
    argv = vars(parser.parse_args())

    plotter_name = argv['data_to_plot']
    argv["show"] = bool(argv["show"])
    argv["smoothing"] -= int(argv["smoothing"])

    del argv['data_to_plot']
    del argv['file']

    main(plotter_name, argv)
