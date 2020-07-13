#!/usr/bin/env python3

from lib import readfai

import argparse
import glob
import itertools
import os
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("contigs", metavar = "path.fasta", help = "indexed fasta file")
    parser.add_argument("interactors", metavar="path.interactors", help = "best interactors for each contig")
    parser.add_argument("exclude", metavar="path.contigs.exclude", help = "list of contigs that should not be assigned")
    parser.add_argument("output", metavar="dir.output", help = "output directory")
    parser.add_argument("scaffolds", metavar = "dir.scaffolds", nargs = "+", help = "input directory")

    args = parser.parse_args()

    fai = readfai(args.contigs)

    dicCo2Scaf = {}
    dicScaf2Co = {}
    dicAssigned = {}
    setUnassigned = set()

    for fpath in args.scaffolds:
        fname = os.path.basename(fpath)
        scaf = fname[:-4]

        if scaf.endswith("_s"):
            scaf = scaf[:-2]

        dicScaf2Co[ scaf ] = []
        dicAssigned[ scaf ] = []

        for line in open(fpath, "r"):
            line = line.strip()
            if len(line) == 0 or line[0] in ">#":
                continue

            if line[0] in "+-":
                contig = line[1:].lstrip()
            else:
                contig = line

            dicCo2Scaf[ contig ] = scaf
            dicScaf2Co[ scaf ].append( contig )

        print("{} has {} contigs".format(scaf, len(dicScaf2Co[ scaf ]) ))

    setExclude = set()
    for line in open(args.exclude):
        cname = line.strip()
        setExclude.add(cname)

    print("{} excluded contigs".format(len(setExclude)))

    for cname in fai.entries.keys():
        if cname not in dicCo2Scaf and cname not in setExclude:
            setUnassigned.add(cname)

    print("{} assigned contigs".format(len(dicCo2Scaf)))
    print("{} unassigned contigs".format(len(setUnassigned)))


    assigned = None

    while assigned != 0:

        assigned = 0
        unassigned_cutoffs = 0
        unassigned_multi = 0
        unassigned_notarget = 0

        for line in open( args.interactors, "r" ):
            line = line.strip()
            if len(line) == 0:
                continue

            items = line.split()
            src = items[0]

            if src in dicCo2Scaf:
                continue

            dicScaf = {}
            for i in range(2, len(items), 2):
                trgt = items[i]
                cnt = int(items[i+1])

                if trgt in setExclude:
                    continue

                if trgt in dicCo2Scaf:
                    scaf = dicCo2Scaf[trgt]

                    if scaf in dicScaf:
                        dicScaf[scaf].append(cnt)
                    else:
                        dicScaf[scaf] = [ cnt ]

            arrDel = []
            for scaf, cnts in dicScaf.items():
                if sum(cnts) == 1:
                    arrDel.append(scaf)
            for scaf in arrDel:
                del dicScaf[scaf]

            if len(dicScaf) == 1:
                (scaf, cnts) = dicScaf.popitem()

                # TODO: hardcoded --- should probably always be the same value as passed to scaffold_graph using -l
                if sum(cnts) > 5:
                    dicAssigned[ scaf ].append(src)
                    dicCo2Scaf[src] = scaf

                    assigned += 1
                else:
                    unassigned_cutoffs += 1
            elif len(dicScaf) == 0:
                unassigned_notarget += 1
            else:
                unassigned_multi += 1

        print(f"assigned {assigned:5} cutoffs {unassigned_cutoffs:5} multi {unassigned_multi:5} no {unassigned_notarget:5}")

        # break

    unassigned = set()
    for c in setUnassigned:
        if c not in dicCo2Scaf:
            unassigned.add( c )

    total_sum = 0

    for scaf in dicScaf2Co.keys():
        total = sum( [  fai.entries[c].length for c in dicAssigned[scaf] ] ) + \
                sum( [  fai.entries[c].length for c in dicScaf2Co[scaf] ] )

        total_sum += total

        total_mb = total / 1000 / 1000

        print("{:10} {:6,} contigs / {:10.2f} mb".format(
                    scaf[:20],
                    len(dicAssigned[scaf]) + len(dicScaf2Co[scaf]),
                    total_mb ))

        fout = open( os.path.join(args.output, f"{scaf}_s.txt"), "w" )

        fout.write(f">{scaf}\n")

        for contig in dicScaf2Co[scaf]:
            fout.write(f"{contig}\n")

        for contig in dicAssigned[scaf]:
            fout.write(f"{contig}\n")

        fout.close()

    print("{:.2f} mb in {} scaffolds".format( total_sum / 1000 / 1000, len(dicScaf2Co) ))

if __name__ == "__main__":
    main()
