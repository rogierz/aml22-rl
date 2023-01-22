from argparse import ArgumentParser
from pprint import pprint

from plot_GS_results import main as plot_gs
from plot_TPE_results import main as plot_tpe

PLOTTERS = {
    "tpe": plot_tpe,
    "gs": plot_gs
}


def main(p_name, args):
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

    argv = vars(parser.parse_args())

    plotter_name = argv['data_to_plot']
    argv["show"] = bool(argv["show"])

    del argv['data_to_plot']
    del argv['file']

    # pprint(argv)

    # parser.print_help()
    # argv = {'fname': 'plot',
    #         'input_file': 'data/TPE_data.csv',
    #         'output_folder': 'plots',
    #         'show': False}
    #
    # plotter_name = "tpe"

    main(plotter_name, argv)
