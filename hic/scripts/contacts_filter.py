#!/usr/bin/env python

import sys
import argparse
import operator

def strip_contig(c):
    if c[0] in "+-":
        c = c[1:]

    return c.strip()

def main():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("-c", metavar = "file", help = "file containing the contigs to be kept", default = None)
    parser.add_argument("-o", dest = "ofile", metavar = "file", help = "contacts")
    parser.add_argument("hits", nargs="+", help = "hits")

    args = parser.parse_args()

    print( ">>" + args.ofile + "<<" )

    fout = open( args.ofile, "w" )

    # try to open the files

    try:
        for fpath in args.hits:
            open(fpath, "r")
    except:
        sys.exit(1)


    # read contig names

    setContigs = set()
    for line in open(args.c, "r"):
        line = line.strip()
        if len(line) > 0 and line[0] not in "#>":
            contig = strip_contig(line)
            setContigs.add(contig)

    print("{} contigs loaded".format( len(setContigs) ) )

    if len(setContigs) == 0:
        print("empty set of contigs")
        sys.exit(1)

    for fpath in args.hits:

        for line in open(fpath):
            line = line.strip()
            if line[0] == '#':
                continue

            items = line.split()

            if len(items) == 5:
                c1 = strip_contig(items[0])

                if c1 not in setContigs:
                    continue
            elif len(items) == 6:
                c1 = strip_contig(items[0])
                if c1 not in setContigs:
                    continue

                c2 = strip_contig(items[3])
                if c2 not in setContigs:
                    continue
            else:
                print("malformed line: " + line)
                break

            fout.write(line + "\n")

if __name__ == "__main__":
    main()
