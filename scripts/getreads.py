#!/usr/bin/env python3

import argparse
import os
import sys

from marvel import DB

def wrap_seq(strSeq):
    arrLines = []

    for i in range(0, len(strSeq), 50):
        arrLines.append( strSeq[i : i+50] )

    return "\n".join( arrLines )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", dest = "source", action = "store_true")
    parser.add_argument("db")
    parser.add_argument("reads", nargs = "+")
    args = parser.parse_args()

    reads = args.reads
    usesrc = args.source

    db = DB.DB(args.db)

    if usesrc:
        src2rid = {}
        tracksrc = db.track("source")

        for i in range( db.reads() ):
            (srid, _dummy, _dummy) = tracksrc.get(i)
            src2rid[srid] = i

    rids = set()

    for r in reads:
        if os.path.exists(r):
            rids = rids.union( [ int(x.strip()) for x in open(r) ] )
        elif "-" in r:
            (rfrom, _dummy, rto) = r.partition("-")
            rfrom = int(rfrom)
            rto = int(rto)

            rids.union( range(rfrom, rto + 1) )
        elif r.isdigit():
            rids.add( r )
        else:
            print(f"unknown argument {r}")
            sys.exit(1)

    for rid in sorted(rids):
        if usesrc:
            currid = src2rid[rid]
            seq = db.sequence(currid)
            print(">{} source={}\n{}".format(currid, rid, wrap_seq(seq)))
        else:
            seq = db.sequence(rid)
            print(">{}\n{}".format(rid, wrap_seq(seq)))

if __name__ == "__main__":
    main()
