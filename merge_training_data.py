"""Merge training data from a couple of runs"""

from __future__ import print_function

import argparse

import training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='data.csv', help="Specify the output file name")
    parser.add_argument('--min-high-tile', '-m', type=int, default=1024, help="Specify the minimum highest tile for a game to be merged")
    parser.add_argument('--max-files', type=int, default=None, help="Specify the maximum number of files to merge")
    parser.add_argument('input', nargs='+', help="Specify input files to merge")
    args = parser.parse_args()

    data = training_data.training_data()

    accepted_input_files = 0
    for i in args.input:
        di = training_data.training_data()
        di.import_csv(i)
        high_tile = di.get_highest_tile()
        if high_tile >= args.min_high_tile:
          data.merge(di)
          accepted_input_files += 1
          if args.max_files and accepted_input_files >= args.max_files:
              print("Breaking out at maximum number of files {}".format(args.max_files))
              break
        else:
          print("Rejecting {} as highest tile ({}) was less than minimum".format(i, high_tile))

    print("Combined data has {} samples from {} files".format(data.size(), accepted_input_files))
    data.export_csv(args.output, add_returns=True)
