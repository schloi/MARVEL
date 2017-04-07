#!/usr/bin/env python

from __future__ import print_function

import sys
import string
import argparse
import os

from constants import *

from marvel import DB
from marvel import LAS
import marvel.config

try:
    import networkx as nx
except:
    print("The networkx library doesn't seem to be installed on your system.")
    sys.exit(1)

###
#

def e_apply_attributes(g, aname, avalues):
    for (e, val) in avalues.items():
        g.edge[e[0]][e[1]][aname] = val

def v_apply_attributes(g, aname, avalues):
    for (n, val) in avalues.items():
        g.node[n][aname] = val

def e_source(e):
    return e[0]

def e_target(e):
    return e[1]

def v_all_edges(g, v):
    edges = []

    for (a, b) in g.edges():
        if a == v or b == v:
            edges.append( (a,b) )

    return edges

def v_in_edges(g, v):
    edges = []

    for (a, b) in g.edges():
        if b == v:
            edges.append( (a,b) )

    return edges

def v_out_edges(g, v):
    edges = []

    for (a, b) in g.edges():
        if a == v:
            edges.append( (a,b) )

    return edges

#
###

def print_path(path, vread):
    j = 0
    npath = len(path)
    for j in range(npath):
        (e, v, next_dir) = path[j]

        print("{}".format(vread[v]), end = "")

        if j < npath - 1:
            print(" -> ", end = "")
    print("")

def reverse(direction):
    if direction == 'l':
        return 'r'
    else:
        return 'l'

def vertexEdgeQuality(g, v, vread, eend, ediv):
    nSum = 0
    n = 0

    endl = 0
    ende = 0

    for e in v_all_edges(g, v):
        nSum += ediv[e]
        n += 1

        if eend[e] == 'l':
            endl += 1
        else:
            ende += 1

    if endl == 0 or ende == 0:
        nSum *= 2

    return float(nSum) / n

g_revcomp = string.maketrans("ACGTacgt", "TGCAtgca")
def revcomp(seq):
    seq = str(seq).translate(g_revcomp)
    return seq[::-1]

def wrapSeq(strSeq):
    arrLines = []

    for i in range(0, len(strSeq), 50):
        arrLines.append( strSeq[i : i+50] )

    return "\n".join( arrLines )

def write_path(fileOut, db, track_source, track_trim, vread, eflags, elen, eend, path, pathid, e1, e2):
    if len(path) == 0:
        return 0

    dir = eend[ path[0] ]

    if dir == 'l':
        bComp = True
    else:
        bComp = False

    v = e_source(path[0])

    rid = vread[v]
    (trim_b, trim_e) = track_trim.get(rid)

    seq = db.sequence(rid)[trim_b : trim_e]
    if bComp:
        seq = revcomp(seq)

        pathRid = [ (rid, trim_e, trim_b) ]
    else:
        pathRid = [ (rid, trim_b, trim_e) ]

    if track_source != None:
        pathSource = [ track_source.get(rid)[0] ]
    pathSeq = seq

    for e in path:
        assert(e_source(e) == v)

        flags = eflags[e]
        v = e_target(e)
        rid = vread[v]

        (trim_b, trim_e) = track_trim.get(rid)

        seq = db.sequence(rid)

        if flags & LAS.OVL_COMP:
            bComp = not bComp

        if bComp:
            ovh_end = trim_b + elen[e]

            pathRid.append( (rid, ovh_end, trim_b) )

            pathSeq += revcomp(seq[trim_b : ovh_end])
        else:
            ovh_beg = trim_e - elen[e]

            pathRid.append( (rid, ovh_beg, trim_e) )

            pathSeq += seq[ovh_beg : trim_e]

        if track_source != None:
            pathSource.append( track_source.get(rid)[0] )
        else:
            pathSource = []

    fileOut.write(">path_{} path={} ends={},{} reads={} sreads={}\n{}\n".format(pathid, pathid,
                    e1, e2,
                    ",".join( [ "{},{},{}".format(*x) for x in pathRid ] ),
                    ",".join( [ str(x) for x in pathSource ] ),
                    wrapSeq(pathSeq)))

    return len(pathSeq)

def process_graph(g, db, fileSeq, arrPaths, track_trim_name):
    vread_prop = { n : g.node[n]["read"] for n in g.nodes() }

    eflags_prop = { e : g.edge[e[0]][e[1]]["flags"] for e in g.edges() }
    elen_prop = { e : g.edge[e[0]][e[1]]["length"] for e in g.edges() }
    eend_prop = { e : g.edge[e[0]][e[1]]["end"] for e in g.edges() }

    track_source = db.track("source")
    track_trim = db.track(track_trim_name)

    total = 0
    for (pid, e1, e2, path) in arrPaths:
        plen = write_path(fileSeq, db, track_source, track_trim, vread_prop, eflags_prop, elen_prop, eend_prop, path, pid, e1, e2)

        total += plen
        print("wrote path {} length {} total {}".format(pid, plen, total))

def main():
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument("database", help = "database name")
    parser.add_argument("graph_path_pairs", nargs="+", help = "overlap graph and paths")
    parser.add_argument("-t", "--trimTrack",
                        dest = "trimTrack", type = str,
                        default = "trim",
                        help = "trim track (valid for filtered overlaps)")

    args = parser.parse_args()

    if len(args.graph_path_pairs) % 2:
        parser.error("graph_path_pairs should be pairs of .graphml and .path files")

    print("opening db {}".format(args.database))

    db = DB.DB(args.database)
    track_trim_name = args.trimTrack

    for fname in args.graph_path_pairs:
        if not os.path.exists(fname):
            print("file {} not found".format(fname))
            continue

    for i in range(0, len(args.graph_path_pairs), 2):
        fgraph = args.graph_path_pairs[i]
        fpaths = args.graph_path_pairs[i + 1]

        basename = fgraph.rpartition(".")[0]
        of = basename + ".fasta"

        print("loading graph {}".format(fgraph))

        try:
            g = nx.read_graphml(fgraph)
        except Exception as e:
            print("error: failed to load graph: " + str(e))
            sys.exit(1)

        fileSeq = open(of, "w")

        paths = []

        for line in open(fpaths):
            line = line.strip()

            if len(line) == 0:
                continue

            items = line.split(" ")
            pid = int(items[1])
            e1 = int(items[2])
            e2 = int(items[3])

            path = []
            for edge in items[4:]:
                (n1, n2) = edge.split("-")
                path.append( (n1, n2) )

            paths.append( (pid, e1, e2, path) )

        process_graph(g, db, fileSeq, paths, track_trim_name)

        fileSeq.close()

if __name__ == "__main__":
    if not marvel.config.has_recent_python():
        sys.exit(1)

    main()
