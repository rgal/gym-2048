"""Merge training data from a couple of runs"""

from __future__ import print_function

import argparse

import training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='.', help="Set the directory for outputting files")
    parser.add_argument('input', nargs='+', help="Specify directories of input files to merge")
    args = parser.parse_args()

    data = training_data.training_data()

    for i in args.input:
        di = training_data.training_data()
        di.read(i)
        data.merge(di)

    data.write(args.output)
