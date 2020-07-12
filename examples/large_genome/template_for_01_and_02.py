#!/usr/bin/env python3

import itertools

DB = "LARGEGENOME"
BLOCKS = 2948
COVERAGE = 30

def main():
    # pick some random block ranges from the db

    getiter = lambda : itertools.chain( range(1, 11), range(1400, 1411), range(2420, 2431) )

    # create overlaps for the selected blocks

    for i in getiter():
        for j in range(1, BLOCKS + 1, 8):

            emit = f"daligner -A -Poverlaps/ -T4 {DB}.{i}"
            for k in range(8):
                if j + k > BLOCKS + 1:
                    break

                emit += " {}.{}".format(DB, j+k)

            print(emit)

    # merge alignments and create conservative repeat annotation

    for i in getiter():
        print(f"LAmerge overlaps/{DB}.{i}.las overlaps/{i:05}")
        print(f"LArepeat -h 4 -l 3.5 -c {COVERAGE} {DB} overlaps/{DB}.{i}.las")

    print(f"TKcombine {DB} repeats " + " ".join([ str(x) + ".repeats" for x in getiter() ]) )

    # transitively transfer repeat annotation to all other blocks

    for i in getiter():
        print(f"TKhomogenize {DB} overlaps/{DB}.{i}.las")

    print(f"TKcombine {DB} hrepeats " + " ".join([ str(x) + ".hrepeats" for x in getiter() ]) )

    # overlap all blocks with repeats soft-masked

    for i in range(1, BLOCKS + 1):
        for j in range(1, BLOCKS + 1, 8):

            emit = f"daligner -A -Poverlaps/ -T4 -m hrepeats {DB}.{i}"
            for k in range(8):
                if j + k > BLOCKS + 1:
                    break

                emit += " {}.{}".format(DB, j+k)

            print(emit)

    # create final block alignment files

    for i in getiter():
        print(f"LAmerge overlaps/{DB}.{i}.las overlaps/{i:05}")

    ### only needed for stage 01

    # fix the reads

    for i in getiter():
        print(f"LAq {DB} overlaps/{DB}.{i}.las")

    print(f"TKmerge {DB} q")
    print(f"TKmerge {DB} trim")

    for i in range(1, BLOCKS + 1):
        print(f"LAfix -g -1 {DB} overlaps/{DB}.{i}.las fasta_fixed/{DB}.{i}.fa")

    print(f"cat fasta_fixed/{DB}*.fa > {DB}_FIXED.fa")

    # create new database using the fasta files
    # containing the fixed reads

if __name__ == "__main__":
    main()
