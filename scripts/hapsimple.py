#!/usr/bin/env python3

import argparse
from marvel import DB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type = float, default = 0.75)
    parser.add_argument("--high", type = float, default = 1.25)
    parser.add_argument("-s", type = int, default = 5)
    parser.add_argument("db")
    parser.add_argument("hap")
    args = parser.parse_args()

    minsupport = args.s
    ratiolow = args.low
    ratiohigh = args.high

    db = DB.DB(args.db)

    fhaps = open(args.hap, "r")

    setSupportReadIds = set()

    for line in fhaps:
        items = [ int(x) for x in line.strip().split() ]

        svid = items[0]
        aread = items[1]

        length_longest = 0
        longest_breaking = None

        breaking = 0
        crossing = 0
        i = 4
        while i < len(items):
            bread = items[i]

            if bread < 0:
                crossing += 1
                i += 1
            else:
                breaking += 1

                blen = db.length(bread)

                if blen > length_longest:
                    length_longest = blen
                    longest_breaking = bread

                i += 6

        if breaking < minsupport or crossing < minsupport:
            continue

        ratio = breaking / crossing
        if ratio < ratiolow or ratio > ratiohigh:
            continue

        setSupportReadIds.add( longest_breaking )

    for rid in sorted(setSupportReadIds):
        print(f"{rid}")

if __name__ == "__main__":
    main()
