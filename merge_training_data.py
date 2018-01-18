"""Merge training data from a couple of runs"""

from __future__ import print_function

import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='.', help="Set the directory for outputting files")
    parser.add_argument('input', nargs='+', help="Specify directories of input files to merge")
    args = parser.parse_args()

    try:
        os.makedirs(args.output)
    except OSError:
        pass
    for vals in ['x', 'y']:
        data = None
        filename = '{}.npy'.format(vals)
        for i in args.input:
            with open(os.path.join(i, filename), 'r') as f:
                if data is not None:
                    data = np.concatenate((data, np.load(f)))
                else:
                    data = np.load(f)
        with open(os.path.join(args.output, filename), 'w') as f:
            print("Outputting {} with shape {}".format(filename, data.shape))
            np.save(f, data)
