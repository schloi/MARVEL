
import struct
import os

from marvel.oflags import *

OM_AB = 0
OM_AE = 1
OM_BB = 2
OM_BE = 3
OM_FLAGS = 4
OM_AREAD = 5
OM_BREAD = 6
OM_DIFFS = 7
OM_OVH = 8

class Overlap(object):
    def __init__(self, ab, ae, bb, be, flags, aread, bread, diffs):
        self.ab = ab
        self.ae = ae
        self.bb = bb
        self.be = be

        self.flags = flags
        self.aread = aread
        self.bread = bread
        self.diffs = diffs

        self.ovh = None

    def __str__(self):
        return "{} x {} {}..{} x {}..{} o {} f {} d {}".format(self.aread, self.bread, self.ab, self.ae, self.bb, self.be, self.ovh, self.flags, self.diffs)

class LAS(object):
    STRUCT_LAS_HEADER = "@qi"
    STRUCT_LAS_RECORD = "@iiiiiiIii4x"

    BUFFER_SIZE       = 1 * 1024 * 1024
    TRACE_XOVR        = 125

    def __init__(self, path):
        self.nlas   = 0
        self.twidth = 0
        self.tbytes = 0

        self.lasPath = path

        self.fileLas = open(self.lasPath, "rb")

        self.rlen = struct.calcsize(self.STRUCT_LAS_RECORD)

        self.read_header()

        self.data = ""
        self.dcur = 0
        self.dlen = 0

    def __iter__(self):
        return self

    def path(self):
        return os.path.abspath(self.lasPath)

    def read_header(self):
        self.fileLas.seek(0)

        hlen = struct.calcsize(self.STRUCT_LAS_HEADER)

        header = self.fileLas.read(hlen)

        (nlas, twidth) = struct.unpack(self.STRUCT_LAS_HEADER, header)

        self.nlas = nlas
        self.twidth = twidth

        if twidth <= self.TRACE_XOVR:
            self.tbytes = 1
        else:
            self.tbytes = 2

    def next(self):
        rec = self.fileLas.read(self.rlen)

        if len(rec) == 0:
            raise StopIteration

        (tlen, diffs, ab, bb, ae, be, flags, aread, bread) = struct.unpack(self.STRUCT_LAS_RECORD, rec)

        self.fileLas.seek(self.tbytes * tlen, 1)

        return [ab, ae, bb, be, flags, aread, bread, diffs, -1]

    def rewind(self):
        self.fileLas.seek(0)

    def close(self):
        self.fileLas.close()

