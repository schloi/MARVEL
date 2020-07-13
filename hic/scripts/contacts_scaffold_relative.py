#!/usr/bin/env python3

import sys
import os
import argparse

class node(object):
    def __init__(self, level, rc, length):
        self.rc = rc
        self.length = length
        self.childs = []
        self.level = level

    def __str__(self):
        output = "{} {} {} {}".format(self.level, self.rc, self.length, len(self.childs))

        for c in self.childs:
            output += "\n" + "  " * c.level + str(c)

        return output

    def leaf(self):
        return ( len(self.childs) == 0 )

def traverse(root, offlevel, output, rc):
    output.append( (root.level, root.rc, offlevel, offlevel + root.length) )

    if root.leaf():
        return offlevel + root.length

    if rc:
        if root.rc:
            thisrc = False
        else:
            thisrc = True
    else:
        if root.rc:
            thisrc = True
        else:
            thisrc = False

    if thisrc: # if root.rc:
        for n in reversed( root.childs ):
            offlevel = traverse(n, offlevel, output, thisrc)
    else:
        for n in root.childs:
            offlevel = traverse(n, offlevel, output, thisrc)

    return offlevel

def process_mtree(mtree):

    if len(mtree) == 0:
        return []

    root = node( *mtree[0] )
    stack = [ root ]

    for (level, rc, length) in mtree[1:]:
        if root.level + 1 != level:
            assert( level <= root.level )

            while level <= root.level:
                root = stack.pop()

        root.childs.append( node( level, rc, length ) )
        stack.append(root)

        root = root.childs[-1]

    output = []
    traverse(stack[0], 0, output, False)

    return output


def read_scaffolds(path):
    fin = open(path)

    arrScaffolds = []
    arrMergeTrees = []

    arrCurScaf = []
    arrCurMTree = []

    arrLevels = []

    for line in fin:
        line = line.strip()
        if len(line) == 0 or line[0] == '=':
            continue

        if line[0] == '#':
            items = line[1:].strip().split()
            if len(items) != 4:
                continue

            rc = items[0]
            seqname = items[1]
            level = int(items[2])
            seqlen = int(items[3])

            assert(rc in "+-")

            if rc == "+":
                rc = 0
            else:
                rc = 1


            while level >= len(arrLevels):
                arrLevels.append(0)

            # arrCurMTree.append( (level, rc, arrLevels[level], arrLevels[level] + seqlen) )
            arrCurMTree.append( (level, rc, seqlen) )

            arrLevels[level] += seqlen
        elif line[0] == '>':
            if len(arrCurScaf) > 0:
                arrScaffolds.append( (scaf_name, arrCurScaf) )
                arrCurScaf = []

            scaf_name = line[1:].lstrip()
            arrMergeTrees.append( arrCurMTree )
            arrCurMTree = []
            arrLevels = []
        else:
            if line[0] == '+':
                rc = False
                contig = line[1:].lstrip()
            elif line[0] == '-':
                rc = True
                contig = line[1:].lstrip()
            else:
                rc = False
                contig = line

            arrCurScaf.append( (rc, contig) )

    fin.close()

    if len(arrCurScaf) > 0:
        arrScaffolds.append( (scaf_name, arrCurScaf) )

    assert( len(arrScaffolds) == len(arrMergeTrees) )

    return ( arrScaffolds, arrMergeTrees )

def read_fai(path):
    if path.endswith(".fa"):
        path += ".fai"

    if not os.path.isfile(path):
        return []

    try:
        faifile = open(path, "r")
    except:
        print("failed to open {}".format(path))
        sys.exit(1)

    fai = {}
    for line in faifile:
        items = line.strip().split()
        contig = items[0]
        clen = int(items[1])

        fai[contig] = clen

    faifile.close()

    return fai

def compute_shift(arrContigs, dicFai):
    dicShift = {}
    nShift = 0

    for (rc, contig) in arrContigs:
        dicShift[contig] = (nShift, rc)
        nShift += dicFai[contig]

    return (dicShift, nShift)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", metavar = "file", help = "input scaffold")
    parser.add_argument("-f", metavar = "file", help = "indexed fasta containing the contigs")
    parser.add_argument("-o", metavar = "file", help = "scaffold relative links output")
    parser.add_argument("-i", metavar = "file", help = "links input", nargs = "+")
    parser.add_argument("-m", metavar = "file", help = "merge trees output", default = None)
    parser.add_argument("-I", metavar = "file", help = "input links used", default = None)
    parser.add_argument("-F", metavar = "file", help = "create fai for scaffold", default = None)
    parser.add_argument("-c", metavar = "file", help = "contig coordinates", default = None)

    args = parser.parse_args()

    pathContigCoordinates = args.c
    pathScaffold = args.s
    pathOutput = args.o
    arrPathInput = args.i
    pathFasta = args.f
    pathFai = args.F
    pathInputUsed = args.I
    pathMergeTrees = args.m

    if pathScaffold is None:
        print("no scaffolds specifiied")
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(pathScaffold):
        print("unknown file {}".format(pathScaffold))
        sys.exit(1)

    if pathFasta is None:
        print("no fasta file specified")
        sys.exit(1)

    dicFai = read_fai(pathFasta)

    if len(dicFai) == 0:
        print("unable to read fasta index {}".format(pathFasta))
        sys.exit(1)

    (arrScaffolds, arrMergeTrees) = read_scaffolds(pathScaffold)
    c2s = {}
    s2shift = {}

    if pathFai:
        fout = open(pathFai, "w")
    else:
        fout = None

    if pathContigCoordinates:
        fout_coord = open(pathContigCoordinates, "w")
    else:
        fout_coord = None

    if pathMergeTrees:
        fout_mtrees = open(pathMergeTrees, "w")
    else:
        fout_mtrees = None

    for i in range( len(arrScaffolds) ):
        (sname, sitems) = arrScaffolds[i]

        # TODO: handle no mtree case

        if fout_mtrees:
            mtree = process_mtree(arrMergeTrees[i])

            for (level, rc, beg, end) in mtree:
                fout_mtrees.write("{} {} {} {} {}\n".format(sname, level, rc, beg, end))

        (shift, slen) = compute_shift(sitems, dicFai)
        s2shift[sname] = shift

        nshift_prev = 0
        for cname in sorted(shift, key = shift.get):
            nshift = shift[cname][0]

            if fout_coord:
                if nshift > 0:
                    fout_coord.write(" {}\n".format(nshift))

                fout_coord.write("{} {} {}".format(sname, cname, nshift))

            nshift_prev = nshift

        if fout_coord:
            fout_coord.write(" {}\n".format(slen))

        if fout:
            fout.write("{} {} {} {} {}\n".format(sname, slen, "-", "-", slen + 1))

        for (rc, contig) in sitems:
            c2s[contig] = sname

    if fout_coord:
        fout_coord.close()

    if fout:
        fout.close()

    if fout_mtrees:
        fout_mtrees.close()

    arrFin = []
    for path in arrPathInput:
        try:
            arrFin.append( open(path, "r") )
        except:
            print("failed to open {}".format(path))
            sys.exit(1)

    fout = open(pathOutput, "w")

    if pathInputUsed:
        fout_used = open(pathInputUsed, "w")
    else:
        fout_used = None

    for fin in arrFin:
        for line in fin:
            items = line.strip().split()
            (c1, pos1, q1, c2, pos2, q2) = items

            if c1 in c2s and c2 in c2s and c2s[c1] == c2s[c2]:
                if fout_used:
                    fout_used.write(line)

                sname = c2s[c1]

                pos1 = int(pos1)
                pos2 = int(pos2)

                (shift1, rc1) = s2shift[sname][c1]
                (shift2, rc2) = s2shift[sname][c2]

                if rc1:
                    pos1 = dicFai[c1] - pos1

                if rc2:
                    pos2 = dicFai[c2] - pos2

                pos1 = pos1 + shift1
                pos2 = pos2 + shift2

                c1 = c2 = c2s[c1]

                fout.write( "{} {} {} {} {} {}\n".format(c1,pos1,q1,c2,pos2, q2) )

        fin.close()

    if fout_used:
        fout_used.close()

    fout.close()

if __name__ == "__main__":
    main()
