"""Augment training data by flipping it horizontally."""

from __future__ import print_function

import argparse

import training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='output.csv', help="Set the output file")
    parser.add_argument('input', help="Specify input file")
    args = parser.parse_args()

    data = training_data.training_data()
    data.import_csv(args.input)
    c = data.copy()
    c.hflip()
    data.merge(c)
    data.export_csv(args.output)
