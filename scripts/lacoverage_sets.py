#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import itertools
import os
import sys


from marvel import DB
from marvel import LAS
import marvel.config

# TODO: move paf/fasta functions to lib/something.py


@dataclass
class PafEntry:
    rname : str
    rlen : int
    rstart : int
    rend : int
    strand : str
    tname : str
    tlen : int
    tstart : int
    tend : int
    nmatches : int
    nalnlen : int

def paf_iter(pathpaf):
    fh = open(pathpaf)

    for line in fh:
        items = line.strip().split()[:11]

        rname = items[0]
        rlen = int(items[1])
        rstart = int(items[2])
        rend = int(items[3])
        strand = items[4]
        tname = items[5]
        tlen = int( items[6] )
        tstart = int(items[7])
        tend = int(items[8])
        nmatches = int(items[9])
        nalnlen = int(items[10])

        yield PafEntry( rname, rlen, rstart, rend, strand, tname, tlen, tstart, tend, nmatches, nalnlen )

def key_contig_name(a):
    if a.isdigit():
        return int(a)

    val = 0
    for i in range( len(a) - 1, -1, -1 ):
        val += ord(a[i]) * ( 10 ** (len(a) - 1 - i) )
    return val

def fasta_iter(pathfa):
    fh = open(pathfa)

    faiter = (x[1] for x in itertools.groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())

        (name, _dummy, other) = headerStr.partition(" ")

        args = {}

        for item in other.split(" "):
            if "=" in item:
                (argname, _dummy, argval) = item.partition("=")
                args[ argname ] = argval

        yield (name, args, seq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corrected", metavar = "corrected.db", default = None)
    parser.add_argument("asm_fa")
    parser.add_argument("asm_paf")
    parser.add_argument("sets_ids")
    parser.add_argument("sets_labels")
    args = parser.parse_args()

    if args.corrected:
        print("opening corrected db {}".format(args.corrected))
        dbc = DB.DB(args.corrected)
        tsrc = dbc.track("source")

        print("mapping read ids")
        crid2srid = [-1] * dbc.reads()

        for i in range( dbc.reads() ):
            (srid, dummy, dummy) = tsrc.get(i)
            crid2srid[i] = srid
    else:
        dbc = None
        crid2srid = None

    contig2rid = {}
    contig2chr = {}

    for pe in paf_iter(args.asm_paf):
        if pe.rname not in contig2chr or contig2chr[ pe.rname ][1] < pe.nmatches:
            if pe.tname == 'Y':
                print("here")

            contig2chr[ pe.rname ] = ( pe.tname, pe.nmatches )

    return

    for (name, seqargs, seq) in fasta_iter( args.asm_fa ):
        reads = [ int(x) for x in seqargs["reads"].split(",") ][0::3]

        if crid2srid != None:
            reads = [ crid2srid[x] for x in reads ]

        contig2rid[name] = reads

    chr2rid = {}

    for contig in contig2chr.keys():
        chrname = contig2chr[contig][0]
        rids = contig2rid[contig]

        if chrname not in chr2rid:
            chr2rid[chrname] = rids
        else:
            chr2rid[chrname].extend( rids )

    fout_ids = open(args.sets_ids, "w")
    fout_labels = open(args.sets_labels, "w")

    for chrname in sorted(chr2rid.keys(), key = key_contig_name):
        rids = chr2rid[chrname]
        fout_labels.write(f"{chrname}\n")
        fout_ids.write("{}\n".format( " ".join([str(x) for x in rids]) ))

    fout_labels.close()
    fout_ids.close()

if __name__ == "__main__":
    main()
