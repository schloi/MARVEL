#!/usr/bin/env python

import argparse
import sqlite3
import sys
import itertools
import os

def fasta_iter(fh):
    faiter = (x[1] for x in itertools.groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        # drop the ">"
        header = header.next()[1:].strip()
        seq = "".join(s.strip() for s in faiter.next())
        yield header, seq

def kmg_format(nsize):
    if nsize < 1000:
        return nsize

    ssuf = "KMGT"
    suf = -1
    while nsize >= 1000:
        nsize /= 1000.0
        suf += 1

    return "{:.1f}{}".format(nsize, ssuf[suf])

def kmg_parse(ssize):
    ssize = ssize.strip().lower()

    if ssize[-1] in "kmg":
        num = ssize[:-1]
        suf = ssize[-1]

        if "." in num:
            num = float(num)
        elif "," in num:
            num = float(num.replace(",", "."))
        else:
            num = int(num)

        if ssize == 'k':
            num *= 1000
        elif ssize == 'm':
            num *= 1000 * 1000
        elif ssize == 'g':
            num *= 1000 * 1000 * 1000
    else:
        num = int(ssize)

    return int(num)

def sequence_decompress(cseq, slen):
    letter = ('a', 'c', 'g', 't')

    read = bytearray()

    for c in cseq:
        c = ord(c)

        read.append( letter[ (c >> 6) & 0x3 ] )
        read.append( letter[ (c >> 4) & 0x3 ] )
        read.append( letter[ (c >> 2) & 0x3 ] )
        read.append( letter[ c & 0x3 ] )

    read = read[:slen]

    return read.decode()

SEQ_L2N = [0] * 255
SEQ_L2N[ ord('a') ] = 0
SEQ_L2N[ ord('A') ] = 0
SEQ_L2N[ ord('c') ] = 1
SEQ_L2N[ ord('C') ] = 1
SEQ_L2N[ ord('g') ] = 2
SEQ_L2N[ ord('G') ] = 2
SEQ_L2N[ ord('t') ] = 3
SEQ_L2N[ ord('T') ] = 3

def sequence_compress(seq):
    cread = bytearray( (len(seq) + 3) / 4 )
    slen = len(seq)

    seq = bytearray(seq)

    i = 0
    j = 0
    while i + 3 < slen:
        cread[j] = (SEQ_L2N[seq[i]] << 6) + (SEQ_L2N[seq[i + 1]] << 4) + (SEQ_L2N[seq[i + 2]] << 2) + SEQ_L2N[seq[i + 3]]

        i += 4
        j += 1

    if i < slen:
        b = SEQ_L2N[seq[i]] << 6

        if i + 1 < slen:
            b += SEQ_L2N[seq[i + 1]] << 4

            if i + 2 < slen:
                b += SEQ_L2N[seq[i + 2]] << 2

        cread[j] = b

    return str(cread)

class DB(object):
    def __init__(self, dbname):
        if dbname.endswith(".sqlite"):
            self.dbname = dbname[:-7]
        else:
            self.dbname = dbname
            dbname += ".sqlite"

        self.dbname = dbname
        self.db = sqlite3.connect(dbname)

        self.initialize()

    def initialize(self):
        cur = self.db.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()

        if len(tables) == 0:
            statements = open("schema.sql").readlines()
            cur.executescript(" ".join(statements))

    def sequence_add(self, seq):
        cur = self.db.cursor()
        cseq = sequence_compress(seq)
        cur.execute("INSERT INTO read(sequence, length) VALUES (?, ?)", (buffer(cseq), len(seq)))

    def sequence_bulk_add(self, seqs):
        cur = self.db.cursor()

        for seq in seqs:
            cseq = sequence_compress(seq)
            cur.execute("INSERT INTO read(sequence, length) VALUES (?, ?)", (buffer(cseq), len(seq)))

    def sequence_get(self, sid):
        cur = self.db.cursor()
        cur.execute("SELECT sequence, length FROM read WHERE id = ?", (sid, ))
        (cseq, slen) = cur.fetchone()

        return sequence_decompress(str(cseq), slen)

    def partition(self, blocksize):
        cur = self.db.cursor()
        cur.execute("SELECT id, length FROM read ORDER BY id")

        blocks = 1
        bstart = 1
        bend = -1
        bases = 0
        for (rid, rlen) in cur.fetchall():
            if bases + rlen >= blocksize:
                cur.execute("INSERT INTO block(rid_from, rid_to, size) VALUES(?, ?, ?)", (bstart, bend, bases))
                bstart = rid
                bases = 0
                blocks += 1

            bend = rid
            bases += rlen

        cur.execute("INSERT INTO block(rid_from, rid_to, size) VALUES(?, ?, ?)", (bstart, bend, bases))

        return blocks

    def commit(self):
        self.db.commit()

def process_file(db, fin, minlength):
    bases = 0
    seqs = 0

    for (header, seq) in fasta_iter(fin):
        if len(seq) < minlength:
            continue

        seqs += 1
        bases += len(seq)

        seq = seq.lower()
        # assert( seq == sequence_decompress(sequence_compress(seq), len(seq)) )

        db.sequence_add(seq)

    return (seqs, bases)

def main():
    parser = argparse.ArgumentParser(description = "Tour overlap graph")

    parser.add_argument("database", help = "database name")
    parser.add_argument("fasta", nargs = "+", default = "-", type = str);
    parser.add_argument("-x", "--length", help = "min read length", default = 2000, type = str)
    parser.add_argument("-b", "--blocksize", help = "partition database into blocks", default = -1, type = str)

    args = parser.parse_args()

    db = DB(args.database)

    print("adding sequences")

    if args.fasta == "-":
        (seqs, bases) = process_file(db, sys.stdin)
        print("added {} in {} reads".format(kmg_format(bases), seqs))
    else:
        for fpath in args.fasta:
            if not os.path.exists(fpath):
                print("ERROR: file {} could not be found".format(fpath))
                sys.exit(1)

        for fpath in args.fasta:
            rlen = kmg_parse(args.length)

            (seqs, bases) = process_file(db, open(fpath, "r"), rlen)
            print("  added {} in {} reads from {}".format(kmg_format(bases), seqs, os.path.basename(fpath)))

    if args.blocksize > 0:
        print("partitioning database")
        bs = kmg_parse(args.blocksize)

        blocks = db.partition(bs)

        print("  create {} blocks of size {}".format(blocks, kmg_format(bs)))

    db.commit()

if __name__ == "__main__":

    import cProfile

    cProfile.run("main()")
