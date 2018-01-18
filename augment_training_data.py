"""Augment training data by reflecting and rotating it."""

from __future__ import print_function

import argparse
import numpy as np
import os

def augment(inputs, outputs):
    """Flip the board horizontally, then add rotations to other orientations."""
    # Add horizontal flip of inputs and outputs
    flipped_inputs = np.concatenate((inputs, np.flip(inputs, 2)))

    # Swap directions 1 and 3
    temp = np.copy(outputs)
    temp[:,[1,3]] = temp[:,[3,1]]
    flipped_outputs = np.concatenate((outputs, temp))

    # Add 3 rotations of the previous
    augmented_inputs = np.concatenate((flipped_inputs,
        np.rot90(flipped_inputs, k=1, axes=(2, 1)),
        np.rot90(flipped_inputs, k=2, axes=(2, 1)),
        np.rot90(flipped_inputs, k=3, axes=(2, 1))))
    augmented_outputs = np.concatenate((flipped_outputs,
        np.roll(flipped_outputs, 1, axis=1),
        np.roll(flipped_outputs, 2, axis=1),
        np.roll(flipped_outputs, 3, axis=1)))
    return (augmented_inputs, augmented_outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='.', help="Set the directory for outputting files")
    parser.add_argument('input', nargs='+', help="Specify directories of input files to merge")
    args = parser.parse_args()

    with open(os.path.join(args.input[0], 'x.npy'), 'r') as f:
        x = np.load(f)
    with open(os.path.join(args.input[0], 'y.npy'), 'r') as f:
        y = np.load(f)
    try:
        os.makedirs(args.output)
    except OSError:
        pass
    with open(os.path.join(args.output, 'x.npy'), 'w') as f:
        x = np.save(f)
    with open(os.path.join(args.output, 'y.npy'), 'w') as f:
        y = np.save(f)
