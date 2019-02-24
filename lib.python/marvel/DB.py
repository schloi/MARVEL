
import os
import struct
import zlib
import sys

import array
# import numpy

class Track(object):
    STRUCT_TRACK_HEADER    = "@ii"

    STRUCT_TRACK_HEADER_V2 = "@HHLQQQQQQ"
    STRUCT_CHUNK_HEADER    = "@Q"

    def __init__(self):
        self.tName = None
        self.db = None

        self.anno = None
        self.data = None

    @classmethod
    def uncompress_chunks(cls, fin):
        hsize = struct.calcsize(Track.STRUCT_CHUNK_HEADER)
        data = bytearray()

        while True:
            chunk = fin.read(hsize)

            if len(chunk) == 0:
                break

            (clen, ) = struct.unpack(Track.STRUCT_CHUNK_HEADER, chunk)

            chunk = fin.read(clen)
            data.extend( zlib.decompress( chunk ) )

        return data

    @classmethod
    def from_data(cls, anno, data):
        t = cls()

        t.anno = anno
        t.data = data

        return t

    @classmethod
    def from_db(cls, db, tName):
        t = cls()

        t.tName = tName
        t.db = db

        pathAnno = os.path.join(db.dir(), ".{0}.{1}.anno".format(db.name(), tName))

        if not os.path.exists(pathAnno):
            pathAnno = os.path.join(db.dir(), ".{0}.{1}.a2".format(db.name(), tName))
            pathData = os.path.join(db.dir(), ".{0}.{1}.d2".format(db.name(), tName))
            version = 2
        else:
            pathData = os.path.join(db.dir(), ".{0}.{1}.data".format(db.name(), tName))
            version = 1

        if not os.path.exists(pathAnno) or not os.path.exists(pathData):
            return None

        # TODO - move to configure script

        if array.array("i").itemsize == 4:
            typecode4b = "i"
        elif array.array("l").itemsize == 4:
            typecode4b = "l"
        else:
            print("failed to find a 4-byte typecode")
            sys.exit(1)

        fileAnno = open(pathAnno, "rb")
        fileData = open(pathData, "rb")

        if version == 1:
            ht = fileAnno.read( struct.calcsize(Track.STRUCT_TRACK_HEADER) )
            (t.tlen, t.tsize) = struct.unpack(Track.STRUCT_TRACK_HEADER, ht)

            t.anno = array.array("Q")
            t.anno.frombytes( fileAnno.read() )

            # t.anno = numpy.fromfile(fileAnno, numpy.uint64).tolist()
            t.anno = [ int(x / 4) for x in t.anno ]

            t.data = array.array(typecode4b)
            t.data.frombytes( fileData.read() )

            # t.data = numpy.fromfile(fileData, numpy.int32).tolist()
        else:
            ht = fileAnno.read( struct.calcsize(Track.STRUCT_TRACK_HEADER_V2) )
            (t.version, t.size, dummy, t.tlen, t.clen, t.cdlen, dummy, dummy, dummy) = struct.unpack(Track.STRUCT_TRACK_HEADER_V2, ht)

            t.anno = array.array("Q")
            t.anno.frombytes( Track.uncompress_chunks(fileAnno) )

            # t.anno = numpy.frombuffer(Track.uncompress_chunks(fileAnno), numpy.uint64).tolist()
            t.anno = [ int(x / 4) for x in t.anno ]

            t.data = array.array(typecode4b)
            t.data.frombytes( Track.uncompress_chunks(fileData) )

            # t.data = numpy.frombuffer(Track.uncompress_chunks(fileData), numpy.int32).tolist()

        fileAnno.close()
        fileData.close()

        assert( len(t.anno) == db.reads() + 1 )
        assert( len(t.data) == t.anno[-1] )

        return t

    def name(self):
        return self.tName

    def has(self, rid):
        ob = self.anno[rid]
        oe = self.anno[rid + 1]

        return (ob < oe)

    def get(self, rid):
        ob = self.anno[rid]
        oe = self.anno[rid + 1]

        if ob < oe:
            return self.data[ ob : oe ]

        return []


class DB(object):
    STRUCT_HITS_DB    = "@iffffiqiiiPiPPP"
    STRUCT_HITS_READ  = "@iqqi4x" # pad to 32 byte

    def __init__(self, path):
        if not path.endswith(".db"):
            path += ".db"

        if not os.path.exists(path):
            raise FileNotFoundError("could not find {}".format(path))

        self.dbblocks = 0
        self.ureads = 0
        self.freq   = (0.0, 0.0, 0.0, 0.0)
        self.maxlen = 0
        self.totlen = 0

        for line in open(path):
            line = line.strip()
            if line.startswith("blocks"):
                self.dbblocks = int( line[ line.find("=")+1 : ].strip() )
                break

        (self.dbPath, self.dbName) = os.path.split(path)

        if self.dbName.endswith(".db"):
            self.dbName = self.dbName[:-3]

        self.fileBps = open( os.path.join(self.dbPath, "." + self.dbName + ".bps"), "rb" )
        self.fileIdx = open( os.path.join(self.dbPath, "." + self.dbName + ".idx"), "rb" )

        self.tracks = []

        self.__read_index()

    def __read_index(self):
        self.fileIdx.seek(0)

        dbheader = struct.calcsize(self.STRUCT_HITS_DB)
        hdb = self.fileIdx.read(dbheader)

        arrItems = struct.unpack(self.STRUCT_HITS_DB, hdb)

        self.ureads = arrItems[0]
        self.freq   = arrItems[1:5]
        self.maxlen = arrItems[5]
        self.totlen = arrItems[6]

        self.arrReads = []
        lenRead = struct.calcsize(self.STRUCT_HITS_READ)

        for i in range( self.ureads ):
            hr = self.fileIdx.read(lenRead)
            self.arrReads.append( struct.unpack(self.STRUCT_HITS_READ, hr) )

    def bases(self):
        return self.totlen

    def dir(self):
        return os.path.abspath( self.dbPath )

    def path(self):
        return os.path.abspath( os.path.join(self.dbPath, self.dbName) )

    def reads(self):
        return self.ureads

    def name(self):
        return self.dbName

    def blocks(self):
        return self.dbblocks

    def track(self, tName):
        for track in self.tracks:
            if track.name() == tName:
                return track

        track = Track.from_db(self, tName)

        if track != None:
            self.tracks.append(track)

        return track

    def length(self, rid):
        return self.arrReads[rid][0]

    def sequence(self, rid):
        (rlen, boff, dummy, dummy) = self.arrReads[rid]

        letter = (ord('a'), ord('c'), ord('g'), ord('t'))

        self.fileBps.seek(boff)

        clen = (rlen + 3) >> 2

        cread = self.fileBps.read(clen)
        read = bytearray()

        for c in cread:
            read.append( letter[ (c >> 6) & 0x3 ] )
            read.append( letter[ (c >> 4) & 0x3 ] )
            read.append( letter[ (c >> 2) & 0x3 ] )
            read.append( letter[ c & 0x3 ] )

        read = read[:rlen]

        return read.decode()

    def close(self):
        self.fileBps.close()
        self.fileIdx.close()
