"""Rotate and flip training data so that there is a good distribution of orientations"""

from __future__ import print_function

import argparse

import training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='outdata.csv', help="Set the output file name")
    parser.add_argument('input', help="Specify input file name")
    args = parser.parse_args()

    data = training_data.training_data()
    data.import_csv(args.input)
    a, e = data.split()
    a, c = a.split()
    a, b = a.split()
    c, d = c.split()
    e, g = e.split()
    e, f = e.split()
    g, h = g.split()
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())
    print(e.size())
    print(f.size())
    print(g.size())
    print(h.size())
    b.hflip()
    d.hflip()
    f.hflip()
    c.rotate(1)
    d.rotate(1)
    e.rotate(2)
    f.rotate(2)
    g.rotate(3)
    h.rotate(3)
    collect = training_data.training_data()
    collect.merge(a)
    collect.merge(b)
    collect.merge(c)
    collect.merge(d)
    collect.merge(e)
    collect.merge(f)
    collect.merge(g)
    collect.merge(h)
    collect.export_csv(args.output)
