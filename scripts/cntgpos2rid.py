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

def main():
    parser = argparse.ArgumentParser(description = "map contig relative position to underlying read id")

    parser.add_argument("contigs", help = "file with contigs")
    parser.add_argument("cname", help = "name of the contig")
    parser.add_argument("cpos", help = "contig relative position")

    args = parser.parse_args()

    cpath = args.contigs
    cname = args.cname
    cpos = int( args.cpos )
    items = None

    for line in open(cpath):
        if not line.startswith(">"):
            continue

        if not line[1:].startswith(cname):
            continue

        items = line.split()

        break

    if items == None:
        print(f"could not find contig {cname}")
        sys.exit(1)

    reads = []
    ureads = []

    for item in items:
        if item.startswith("reads="):
            print(item[6:])
            reads.extend( [int(x) for x in item[6:].split(",")] )
        elif item.startswith("ureads="):
            print(item[7:])
            ureads.extend( [int(x) for x in item[7:].split(",")] )

    curpos = 0
    prevpos = 0
    rid = None
    urid = None

    i = 0
    while i < len(reads):
        a = reads[i + 1]
        b = reads[i + 2]

        prevpos = curpos

        if a > b:
            curpos += a - b
        else:
            curpos += b - a

        if prevpos < cpos and cpos <= curpos:
            rid = reads[i]

            if len(ureads) > 0:
                urid = ureads[i]

            break

        i += 3

    if rid == None:
        print(f"could not find position {cpos} in {cname}")
        sys.exit(1)

    print(f"{cpos} in {cname} -> rid {rid} urid {urid}")

if __name__ == "__main__":
    if not marvel.config.has_recent_python():
        sys.exit(1)

    main()
