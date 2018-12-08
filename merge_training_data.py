"""Merge training data from a couple of runs"""

from __future__ import print_function

import argparse

import training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='data.csv', help="Specify the output file name")
    parser.add_argument('input', nargs='+', help="Specify input files to merge")
    args = parser.parse_args()

    data = training_data.training_data(True)

    for i in args.input:
        di = training_data.training_data(True)
        di.import_csv(i)
        data.merge(di)

    data.export_csv(args.output)
