#!/usr/bin/env python3

import argparse
import glob
import itertools
import math
import os
import sys

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Set

@dataclass
class SplitAlignment:
    rid : int
    bpos1 : int
    epos1 : int
    bpos2 : int
    epos2 : int

    # TODO: missing comp

@dataclass
class Haplotype:
    FLAG_ACTIVE : ClassVar[int] = 0x1

    svid : int
    rid  : int
    bpos : int
    epos : int

    flags : int = 0

    split : List[ SplitAlignment ] = field(default_factory=list)

    split_rids : Set[ int ] = field(default_factory=set)
    cross_rids : Set[ int ] = field(default_factory=set)

    def __str__(self):
        strht = "@ {} {} {:6} {:6}".format(self.svid, self.rid, self.bpos, self.epos)

        for sa in self.split:
            strht += "\n  S {} {:6}..{:6} {:6}..{:6}".format(sa.rid, sa.bpos1, sa.epos1, sa.bpos2, sa.epos2)

        for crid in sorted(self.cross_rids):
            strht += "\n  C {}".format(crid)

        return strht

    def add_split(self, rp):
        self.split.append(rp)
        self.split_rids.add(rp.rid)

    def add_cross(self, rid):
        self.cross_rids.add(rid)

@dataclass
class LinkedHaplotypes:
    FLAG_ACTIVE : ClassVar[int] = 0x1

    flags : int = 0

    haplotypes : List[ Haplotype ] = field(default_factory = list)
    rids : List[ Set[ int ] ] = field(default_factory = list)

    def __str__(self):
        strlht = ""

        for ht in self.haplotypes:
            strlht += "@ {} {} {} {}\n".format(ht.svid, ht.rid, ht.bpos, ht.epos)

        trans = str.maketrans("01", "CS")
        digits = int( math.log2( len(self.rids) ) )

        for i in range(len(self.rids)):
            srids = self.rids[i]
            bi = bin(i)[2:].zfill(digits)
            bi = bi.translate(trans)

            strlht += "{} {}\n".format(bi, len(srids))

        return strlht

    def configurations(self):

        r2cfg = {}
        for rid in itertools.chain(self.rids[0], self.rids[1]):
            r2cfg[rid] = 0

        for i in range( len(self.rids) ):
            val = 0x1 << i

            for r in self.rids[i]:
                r2cfg[r] += val

        cfgs = set( r2cfg.values() )

        return len(cfgs)


    def contains(self, lht):
        if len(self.rids) < len(lht.rids):
            return False

        selfsvs = set( [ (x.svid, x.rid) for x in self.haplotypes ] )
        argsvs = set( [ (x.svid, x.rid) for x in lht.haplotypes ] )

        return argsvs.issubset(selfsvs)

    def equal(self, lht):
        if len(self.rids) != len(lht.rids):
            return False

        selfsvs = set( [ (x.svid, x.rid) for x in self.haplotypes ] )
        argsvs = set( [ (x.svid, x.rid) for x in lht.haplotypes ] )

        return selfsvs == argsvs

    def intersect(self, lht):

        srids = set()
        for so in self.rids:
            srids = srids.union(so)

        arids = set()
        for ao in lht.rids:
            arids = arids.union(ao)

        crids = srids.intersection(arids)

        if len(crids) == 0:
            return None

        lht_int = LinkedHaplotypes()

        lht_int.haplotypes.extend( self.haplotypes )
        lht_int.haplotypes.extend( lht.haplotypes )

        for so in self.rids:
            lht_int.rids.append( so.intersection(crids) )

        for ao in lht.rids:
            lht_int.rids.append( ao.intersection(crids) )

        return lht_int

    @staticmethod
    def from_haplotype(ht):
        lht = LinkedHaplotypes()
        lht.haplotypes.append(ht)
        lht.rids.append( ht.cross_rids )
        lht.rids.append( ht.split_rids )

        return lht

#
# remove duplicate LinkedHaplotypes
#
def remove_duplicates(lhts):
    dedupe = []
    for i in range(len(lhts)):
        ht1 = lhts[i]
        equal = False
        for j in range(i+1, len(lhts)):
            ht2 = lhts[j]

            if ht1.equal(ht2):
                equal = True
                break

        if not equal:
            dedupe.append( ht1 )

    return dedupe

def phase_haplotypes(haplotypes):

    ### singletons
    #

    haplotypes_final = []
    haplotypes_linked = []

    for i in range( len(haplotypes) ):
        ht1 = haplotypes[i]
        intersected = False

        for j in range(i + 1, len(haplotypes)):
            ht2 = haplotypes[j]

            ht_int = ht1.intersect(ht2)

            if ht_int != None:
                ht_int.flags |= LinkedHaplotypes.FLAG_ACTIVE
                ht2.flags |= Haplotype.FLAG_ACTIVE

                intersected = True

                haplotypes_linked.append( ht_int )

        if intersected:
            ht1.flags |= Haplotype.FLAG_ACTIVE

    haplotypes_final.extend( haplotypes_linked )

    print("linked {} singletons".format(len(haplotypes_linked)))

    #
    ###

    ### extend linked haplotypes
    #

    while True:
        haplotypes_new = []

        # count active singletons
        active = 0
        for lht in haplotypes:
            assert( len(lht.haplotypes) == 1 )
            if lht.flags & LinkedHaplotypes.FLAG_ACTIVE:
                active += 1

        print("linking {} active singletons (out of {}) to {}".format(active, len(haplotypes), len(haplotypes_linked)))

        for ht1 in haplotypes:
            if (ht1.flags & Haplotype.FLAG_ACTIVE) == 0:
                continue

            intersected = False

            for ht2 in haplotypes_linked:
                if ht2.contains(ht1):
                    continue

                ht_int = ht1.intersect(ht2)

                if ht_int != None:
                    haplotypes_new.append(ht_int)
                    intersected = True

            if not intersected:
                ht1.flags ^= Haplotype.FLAG_ACTIVE
                assert(ht1.flags == 0)

        haplotypes_new = remove_duplicates(haplotypes_new)

        print("linked {}".format(len(haplotypes_new)))

        haplotypes_final.extend( haplotypes_new )

        if len(haplotypes_new) == 0:
            break

        haplotypes_linked = haplotypes_new

    haplotypes_final.extend( haplotypes )

    #
    ###

    haplotypes_final.sort( key = lambda x: len(x.haplotypes) )

    return haplotypes_final

def read_haplotypes(pathhap, print_cs):

    haplotypes = []

    for line in open(pathhap):
        # (svid aread ab ab) (bread bb1 be1 bb2 be2 comp)*
        items = [int(x) for x in line.strip().split()]

        svid = items[0]
        aread = items[1]
        ab = items[2]
        ae = items[3]

        if print_cs:
            print(f"#### {svid} {aread} {ab:6} {ae:6}")

        ht = Haplotype(svid, aread, ab ,ae)

        i = 4
        while i < len(items):
            bread = items[i]

            if bread > 0:
                (bb1, be1, bb2, be2, comp) = items[i + 1 : i + 6]

                if bb2 < bb1:
                    (bb1, be1, bb2, be2) = (bb2, be2, bb1, be1)

                gap = abs(be1 - bb2)

                ht.add_split( SplitAlignment(bread, bb1, be1, bb2, be2) )

                if print_cs:
                    print( f"  S {bread:7} {bb1:6}..{be1:6} {bb2:6}..{be2:6} {gap:5} {comp}" )

                i+= 6
            else:
                bread = (-1) * bread

                ht.add_cross(bread)

                if print_cs:
                    print( f"  C {bread:7}" )

                i += 1

        haplotypes.append( LinkedHaplotypes.from_haplotype(ht) )

    return haplotypes

def read_validation_files(dirval):
    valfiles = glob.glob( os.path.join(dirval, "*.val") )

    if len(valfiles) == 0:
        print(f"could not find .val files in {dirval}")
        return None

    svvotes = {}

    for pathval in valfiles:
        fin = open(pathval, "r")

        for line in fin:
            (svid, aread, votes) = [int(x) for x in line.strip().split()]

            if (svid, aread) not in svvotes:
                svvotes[ (svid, aread) ] = 1
            else:
                svvotes[ (svid, aread) ] += 1

        fin.close()

    return svvotes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_hap")
    parser.add_argument("dir_val")
    parser.add_argument("-d", "--mindistance", default = 0)
    parser.add_argument("--print_cs", action = "store_true")
    args = parser.parse_args()

    pathhap = args.path_hap
    dirval = args.dir_val
    mindist = args.mindistance
    print_cs = args.print_cs

    if not os.path.exists(pathhap):
        print(f"could not open {pathhap}")
        sys.exit(1)

    if not os.path.exists(dirval):
        print(f"could not find {dirval}")
        sys.exit(1)

    votes = read_validation_files(dirval)

    print("reading haplotypes")

    haplotypes = read_haplotypes(pathhap, print_cs)

    print("{} variant locations".format( len(haplotypes) ))

    haplotypes_phased = phase_haplotypes(haplotypes)

    print("{} variant configurations".format(len(haplotypes_phased)))


    for lht in haplotypes_phased:
        if len(lht.haplotypes) < 2:
            continue

        print( len(lht.haplotypes), lht.configurations() )
        for ht in lht.haplotypes:
            print( "HT {} {} S/C {}/{} {}".format( ht.svid, ht.rid, len(ht.split_rids), len(ht.cross_rids), len(lht.rids[0]) + len(lht.rids[1]) ) )
        for rids in lht.rids:
            print(rids)

        print("------")

if __name__ == "__main__":
    main()
