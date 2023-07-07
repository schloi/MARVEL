#!/usr/bin/env python3

from __future__ import print_function

import sys
import string
import argparse
import os
import math

from marvel import DB
from marvel import LAS
import marvel.config

# PATH id end1 end2 v1-v2 v2-v3 ... v(n-1)-vn


def main():
    parser = argparse.ArgumentParser(description = "extract read ids from paths")

    parser.add_argument("paths", nargs = '+', help = "file with paths")
    parser.add_argument("readids", help = "file with read ids")

    args = parser.parse_args()

    if len(args.paths) != 1 or args.paths[0] != "-":
        for path in args.paths:
            if not os.path.exists(path):
                print("file {} does not exist".format(path))
                sys.exit(1)

    fout = open(args.readids, "w")
    rids = set()

    for path in args.paths:
        if path == "-":
            fin = sys.stdin
        else:
            fin = open(path, "r")

        for line in fin:
            assert(line.startswith("PATH "))
            items = line.split(" ", 4)

            # empty path
            if len(items) == 4:
                continue

            assert( len(items) == 5 )

            pairs = items[4].replace("-", " ").split(" ")

            for r in pairs:
                rids.add( int(r) )

        fin.close()

    for r in rids:
        fout.write("{}\n".format(r))

    fout.close()

if __name__ == "__main__":
    if not marvel.config.has_recent_python():
        sys.exit(1)

    main()
