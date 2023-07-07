#!/usr/bin/env python3

import argparse
import sqlite3
import sys
import itertools
import os


def fasta_iter(pathfa):
    fh = open(pathfa)

    faiter = (x[1] for x in itertools.groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())

        (name, _dummy, other) = headerStr.partition(" ")

        yield (name, other, seq)

def kmg_parse(ssize):
    ssize = ssize.strip().lower()

    if ssize[-1] in "kmg":
        num = ssize[:-1]
        suf = ssize[-1]

        if "." in num:
            num = float(num)
        elif "," in num:
            num = float(num.replace(",", "."))
        else:
            num = int(num)

        if suf == 'k':
            num *= 1000
        elif suf == 'm':
            num *= 1000 * 1000
        elif suf == 'g':
            num *= 1000 * 1000 * 1000
    else:
        num = int(ssize)

    return int(num)

def process_file(pathfa, target_read_mass):
    rlens = []

    for (header, _dummy, seq) in fasta_iter(pathfa):
        rlens.append( len(seq) )

    rlens.sort()

    read_mass = 0
    i = len(rlens) - 1
    while i >= 0:
        read_mass += rlens[i]
        i -= 1

        if read_mass > target_read_mass:
            break

    return rlens[i + 1]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("readmass", help = "target read mass", type = str)
    parser.add_argument("fasta", nargs = "+", type = str);

    args = parser.parse_args()

    target_read_mass = kmg_parse(args.readmass)

    for pathfa in args.fasta:
        cutoff = process_file(pathfa, target_read_mass)
        print(f"{pathfa} {cutoff}")

if __name__ == "__main__":
    main()

