#!/usr/bin/env python3

#
# OGtour.py
#
# Overlap graph touring
#
# TODO: reimplement the whole thing in either C or Rust
#

from __future__ import print_function

import sys
import operator
import copy
import argparse
import logging

from constants import *
import colormap

from marvel import LAS
import marvel.config

try:
    import networkx as nx
except ImportError:
    logging.error("The networkx library doesn't seem to be installed on your system.")
    sys.exit(1)

### constants
#

COLORS_DIV_MAX = 30     # >=30% goes to same color
COLORS_DIV_MIN = 5      # <=5% goes to same color

COLOR_GREY     = "#F0F0F0"
COLOR_GREEN    = "#5FAA60"
COLOR_RED      = "#A02D31"
COLOR_WHITE    = "#FFFFFF"

#
###

###
#

def e_apply_attributes(g, aname, avalues):
    for (e, val) in avalues.items():
        if g.has_edge(e[0],e[1]):
            g.edges[e[0], e[1]][aname] = val

def v_apply_attributes(g, aname, avalues):
    for (n, val) in avalues.items():
        if n in g.nodes:
            g.nodes[n][aname] = val

def e_source(e):
    return e[0]

def e_target(e):
    return e[1]

def v_all_edges(g, v):
    for e in g.in_edges(v):
        yield e

    for e in g.out_edges(v):
        yield e

    # return g.in_edges(v) + g.out_edges(v)

def v_in_edges(g, v):
    return g.in_edges(v)

def v_out_edges(g, v):
    return g.out_edges(v)

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

def vertexEdgeQuality(g, v, eflags, eend, ediv):
    nSum = 0
    n = 0

    endl = 0
    endr = 0

    for e in v_all_edges(g, v):
        if eflags[e] & ( LAS.OVL_OPTIONAL | LAS.OVL_MODULE ):
            continue

        nSum += ediv[e]
        n += 1

        if eend[e] == 'l':
            endl += 1
        else:
            endr += 1

    if n == 0:
        return sys.maxsize

    return ( 1.0 / max(1, min(10, endl)) ) + ( 1.0 / max(1, min(10, endr)) )

def best_next_node_rec(g, vstart, vflags, vread, eflags, eend, next_dir, path, paths, lookahead, maxpaths):

    # max lookahead reached

    if len(path) == lookahead:
        paths.append( copy.copy(path) )

        if len(paths) > maxpaths:
            return False

        return True

    oedges = []

    for e in v_out_edges(g, vstart):
        vnext = e[1]
        cur_dir = eend[e]

        # stay on the same strand

        if next_dir != None and cur_dir != next_dir:
            continue

        if (eflags[e] & LAS.OVL_OPTIONAL):
            continue

        if (eflags[e] & E_INVERSION):
            continue

        if (eflags[e] & LAS.OVL_COMP):
            cur_dir = reverse(cur_dir)

        oedges.append( (e, vnext, cur_dir) )

    if len(path) == 0:
        # print("maxpaths {} len(oedges) {}".format(maxpaths, len(oedges)))
        maxpaths = max(1, maxpaths // ( len(oedges) + 1 ) )
        maxpaths_inc = maxpaths
    else:
        maxpaths_inc = 0

    for (e, vnext, cur_dir) in oedges:
        path.append( (e, vnext, cur_dir) )

        bStatus = best_next_node_rec(g, vnext, vflags, vread, eflags, eend, cur_dir, path, paths, lookahead, maxpaths)

        path.pop()

        maxpaths += maxpaths_inc

        if len(path) > 0 and not bStatus:
            return False

    return True

def best_next_node(g, vstart, vflags, vread, eflags, eend, next_dir, lookahead):
    path = []
    paths = []

    best_next_node_rec(g, vstart, vflags, vread, eflags, eend, next_dir, path, paths, lookahead, 1000)

    dicCounts = {}

    maxn = 0
    maxv = None

    for path in paths:

        setSeen = set()
        setSeen.add(vread[vstart])

        for (e, vnext, cur_dir) in path:
            if vnext in setSeen:
                eflags[e] |= E_INVERSION

                continue

            setSeen.add(vnext)

            if vnext not in dicCounts:
                count = 1
            else:
                count = dicCounts[vnext] + 1

            if count > maxn:
                maxn = count
                maxv = vnext

            dicCounts[vnext] = count

    maxv = None
    items = list( sorted( dicCounts.items(), key = lambda x: x[1], reverse = True ) )
    for (v, count) in items:
        if not (vflags[v] & V_MODULE):
            maxv = v
            break

    if maxv is None and len(items) > 0:
        maxv = items[0][0]

    fminn = lookahead + 1
    fminv = None

    fminn_mod = lookahead + 1
    fminv_mod = None

    if maxv != None:

        for path in paths:
            has_mod = False
            pos = None
            i = 0
            for (e, vnext, cur_dir) in path:
                if vflags[ vnext ] & V_MODULE:
                    has_mod = True

                if vnext == maxv:
                    pos = i
                    break

                i += 1

            if pos != None:

                if has_mod:
                    if pos < fminn_mod:
                        fminn_mod = pos
                        fminv_mod = path[0][1]
                else:
                    if pos < fminn:
                        fminn = pos
                        fminv = path[0][1]

    if fminv:
        return fminv
    else:
        return fminv_mod

def is_dead_end(g, v, eend, eflags):

    l = 0
    r = 0

    for e in v_all_edges(g, v):
        end = eend[e]
        eflag = eflags[e]

        if e[0] != v:
            if not (eflag & LAS.OVL_COMP):
                end = reverse(end)

        if end == 'l':
            l += 1
        else:
            r += 1

    return ( l == 0 or r == 0)

def mark_dead_ends(g, vflags, eend, eflags):
    nFlagged = 0

    for v in g:
        if is_dead_end(g, v, eend, eflags):
            vflags[v] |= V_DEAD_END
            nFlagged += 1

    return nFlagged

def update_bbdist(g, vflags, eflags, bbdist):
    stack = set()

    for v in g:

        if vflags[v] & V_BACKBONE and bbdist[v] == -1:
            stack.add(v)

    dist = 0

    while len(stack) > 0 and dist < MAX_BB_DISTANCE:
        stack_new = set()

        for v in stack:
            bbdist[v] = dist

        for v in stack:
            for e in v_all_edges(g, v):
                if eflags[e] & LAS.OVL_OPTIONAL:
                    continue

                if e_source(e) == v:
                    vnext = e_target(e)
                else:
                    vnext = e_source(e)

                distnext = bbdist[vnext]

                if distnext == -1 or distnext > dist:
                    stack_new.add(vnext)

        dist += 1
        stack = stack_new


def backbone(g, vstart, vflags, vread, eflags, eend, ediv, elen, vpath, bid, circular, lookahead, ilookahead):
    path = []

    vflags[vstart] |= V_BACKBONE | V_VISITED
    v = vstart

    next_dir = None

    first_dir = None
    other_dir = False

    ends = []

    while True:
        # get the best next node

        emin = None
        vmin = None

        retry = False

        if next_dir is None:
            vbestnext = best_next_node(g, v, vflags, vread, eflags, eend, 'l', lookahead)
        else:
            vbestnext = best_next_node(g, v, vflags, vread, eflags, eend, next_dir, lookahead)

        """
        if vbestnext != None:
            print("@ {} next best {}".format(vread[v], vread[vbestnext]))
        else:
            print("@ {} next best ?".format(vread[v]))
        """

        # TODO --- best sorting criterion ???

        out_edges = list(v_out_edges(g, v))

        bTerminate = False

        for e in sorted(out_edges, key = lambda x: elen[x], reverse = True):
            eflag = eflags[e]

            if eflag & ( E_INVERSION | LAS.OVL_OPTIONAL ):
                # print("skip {}->{} {} {}".format( vread[e[0]], vread[e[1]], eflag & E_INVERSION, eflag & LAS.OVL_OPTIONAL ))
                continue

            if e[0] == v:
                vnext = e[1]
                cur_dir = eend[e]
            else:
                vnext = e[0]

                cur_dir = eend[e]

                if not (eflag & LAS.OVL_COMP):
                    cur_dir = reverse(cur_dir)

            # make sure we don't turn around

            if next_dir != None and cur_dir != next_dir:
                # print("skip 2")
                continue

            # print("pass 1")

            if (eflag & LAS.OVL_COMP):
                cur_dir = reverse(cur_dir)

            if vflags[ vnext ] & V_VISITED:
                logging.debug("@ {} ... visited node encountered".format(vread[vnext]))

                if circular:
                    path_d1 = path
                    path = []

                    retry = False
                    other_dir = True

                    bTerminate = True

                    logging.debug("  -> circular")
                else:
                    # backtrack to last good and start inversion lookahead ...

                    back = 0
                    while len(path) > 1 and e_target(path[-1]) != vnext:
                        eprev = path[-1]
                        vprev = e_target(eprev)
                        vflags[vprev] &= ~(V_BACKBONE | V_VISITED)
                        eflags[eprev] &= ~(E_BACKBONE)
                        vpath[vprev] = -1
                        back += 1

                        path = path[:-1]

                    if len(path) > 0:
                        enext = path[-1]
                        v = e_target(enext)
                        next_dir = eend[enext]
                        if ( eflags[enext] & LAS.OVL_COMP ):
                            next_dir = reverse(next_dir)

                        logging.debug("  -> backtracked {} steps to edge {} -> {}".format(back, vread[e_source(enext)], vread[v]))
                    else:
                        eflags[e] |= E_INVERSION

                        logging.debug("  -> restart at {}".format(vread[v]))

                    if vflags[v] & V_RETRY:
                        vmin = None
                    else:
                        vflags[v] |= V_RETRY

                        best_next_node(g, v, vflags, vread, eflags, eend, next_dir, ilookahead)
                        retry = True

                break

            if vmin is None or \
               ( vflags[vnext] & V_BACKBONE ) or    \
               ( vnext == vbestnext and not ( vflags[vmin] & V_BACKBONE ) ):
                emin = e
                vmin = vnext
                next_dir_min = cur_dir

        if retry:
            continue

        # exhausted the first direction, now turn around and go the other way

        if not bTerminate and vmin != None:
            vflag = vflags[vmin]
            vminpath = vpath[vmin]

            logging.debug("V {:7} E {:7} -> {:7} {} {} {:3}".format(vread[vmin],
                    vread[emin[0]], vread[emin[1]],
                    next_dir, next_dir_min,
                    ediv[emin]))

            vflags[vmin] |= V_BACKBONE | V_VISITED
            eflags[emin] |= E_BACKBONE

            vpath[vmin] = bid

            path.append( emin )

            v = vmin
            next_dir = next_dir_min

            # remember the first direction we took

            if first_dir is None:
                if (eflags[emin] & LAS.OVL_COMP):
                    first_dir = reverse(next_dir)
                else:
                    first_dir = next_dir
        else:
            vflag = 0
            vminpath = -1


        if vmin is None or (vflag & V_BACKBONE) or bTerminate:
            ends.append(vminpath)

            if vmin is None:
                logging.debug("  -> dead end")
            elif vflag & V_BACKBONE:
                logging.info("  -> backbone (path {})".format(vminpath))

            if not other_dir:
                path_d1 = path
                path = []

                other_dir = True
                v = vstart
                next_dir = reverse(first_dir)

                vmin = None
            else:
                if len(ends) == 1:
                    ends.append( ends[0] )

                break

    path_rev = []
    for e in path:
        v1 = e_source(e)
        v2 = e_target(e)

        bFound = False
        for e2 in v_out_edges(g, v2):
            if e_target(e2) == v1:
                path_rev.append(e2)
                bFound = True
                break

        if not bFound:
            logging.error("ERROR: could not find reverse edge for {} -> {}".format(vread[v1], vread[v2]))

        assert(bFound is True)

    path = path_rev[::-1] + path_d1

    if len(path) > 0:
        for e in path:
            vflags[ e[0] ] ^= V_VISITED
        vflags[ e[1] ] ^= V_VISITED

    assert( len(ends) == 2 )

    return (path, ends)

def path_ends(path, eflags, eend):
    if len(path) == 0:
        return None

    e = path[0]
    v = e_source(e)

    if eend[e] == 'l':
        end1 = ( v, "r" )
    else:
        end1 = ( v, "l" )

    e = path[-1]
    v = e_target(e)

    if eflags[e] & LAS.OVL_COMP:
        end2 = ( v, reverse(eend[e]) )
    else:
        end2 = ( v, eend[e] )

    return ( end1, end2 )

def analyze_path_ends(g, paths, vread, eflags, eend):

    ends = {}

    for (path, pid, vstart, dummy) in paths:
        pends = path_ends(path, eflags, eend)

        if pends is None:
            continue

        ( (v1, dir1), (v2, dir2) ) = pends

        ends[v1] = ( dir1, pid )
        ends[v2] = ( dir2, pid )

        logging.debug("path {} ends {} {} {} {}".format(pid, v1, dir1, v2, dir2))

    logging.info("inspecting {} ends".format(len(ends)))

    potential_edges = []
    ends_used = set()

    ends_targets = {}
    details = {}

    for v in ends:
        (edir1, pid1) = ends[v]

        ends_targets[v] = set()

        for e in v_out_edges(g, v):
            if not (eflags[e] & LAS.OVL_OPTIONAL):
                continue

            assert( v == e[0] )

            # potential_edges.append( e )

            vtarget = e[1]

            if vtarget in ends:
                if vtarget in ends_targets[v]:
                    continue

                (edir2, pid2) = ends[vtarget]

                if eflags[e] & LAS.OVL_COMP:
                    comp = "C"
                else:
                    comp = " "

                ends_targets[v].add(vtarget)
                details[ (v, vtarget) ] = (edir1, pid1, edir2, pid2, e)

    while True:

        drop_v = set()

        print("pass end analysis")
        temp_edges = []

        for end in ends_targets:
            vtargets = ends_targets[end]

            if len(vtargets) != 0:
                logging.debug("target {} ({}) -> {} ({})".format(
                            vread[end], len(ends_targets[end]),
                            " ".join(vtargets), len(vtargets)))

            if len(vtargets) != 1:
                continue

            vtarget = vtargets.pop()

            if len( ends_targets[vtarget] ) != 1:
            # if len( ends_targets[vtarget] ) == 0:
                continue

            (edir1, pid1, edir2, pid2, e) = details[ (end, vtarget) ]

            potential_edges.append(e)

            # XXX --- CHANGE
            potential_edges.append( (e[1], e[0]) )

            ends_used.add(end)

            rid_from = vread[end]
            rid_to = vread[vtarget]

            drop_v.add( vtarget )

            logging.debug("potential connection {}/{} {} {} {} {}/{}".format(
                        pid1, rid_from, edir1, comp, edir2, pid2, rid_to))



        if len(drop_v) > 0:
            for v in drop_v:
                for (end, vtargets) in ends_targets.items():
                    if v in vtargets:
                        vtargets.remove(v)
        else:
            break

    return (ends_used, potential_edges)

def analyze_path_ends_v2(g, paths, vread, eflags, eend, ends_used):

    ends = {}

    for (path, pid, vstart, dummy) in paths:
        pends = path_ends(path, eflags, eend)

        if pends is None:
            continue

        ( (v1, dir1), (v2, dir2) ) = pends

        logging.debug("path {} ends {} {} {} {}".format(pid, v1, dir1, v2, dir2))

        if v1 not in ends_used:
            ends[v1] = ( dir1, pid )

        if v2 not in ends_used:
            ends[v2] = ( dir2, pid )


    logging.info("inspecting {} ends".format(len(ends)))

    end2targets = {}
    direct_links = {}

    for vend in ends:
        vtargets = set()

        for e in v_out_edges(g, vend):
            if not ( eflags[e] & LAS.OVL_OPTIONAL ):
                continue

            assert( vend == e[0] )
            target = e[1]

            vtargets.add(target)

            if target in ends:
                logging.debug("{} -> {} direct link".format( vread[vend], vread[target] ))

                direct_links[ (vend, target) ] = e
                direct_links[ (target, vend) ] = e

        end2targets[ vend ] = vtargets

    potential_edges = set()

    for vend1 in ends:
        if vend1 in ends_used:
            continue

        targets1 = end2targets[vend1]

        choices = set()

        for vend2 in ends:
            if vend2 in ends_used:
                continue

            if vend1 == vend2:
                continue

            targets2 = end2targets[vend2]

            shared = targets1.intersection(targets2)

            if len(shared) > 10:
                logging.debug("{} -> {} supported by {}".format( vread[vend1], vread[vend2], len( shared ) ))

                if ( vend1, vend2 ) in direct_links:
                    logging.debug("  + direct link")
                    # choices.add( direct_links[ (vend1, vend2) ] )

        if len(choices) == 1:
            potential_edges |= choices

    return potential_edges

def graph_compute_paths(g, arrStartVertices, bbdist_prop,
                        vflags_prop, vread_prop, vpath_prop,
                        eflags_prop, eend_prop, ediv_prop, elen_prop,
                        circular, lookahead, ilookahead):

    path_num = 0
    vcur = 0
    arrPaths = []

    while True:
        # look for a good start vertex

        while vcur < len(arrStartVertices):
            (v, fQuality) = arrStartVertices[vcur]
            vcur += 1

            dist = bbdist_prop[v]

            if dist != -1 and dist < MAX_BB_DISTANCE:
                continue

            flags = vflags_prop[v]

            if flags & ( V_BACKBONE | V_DISCARD | V_DEAD_END | V_MODULE | V_OPTIONAL ):
                continue

            break

        if vcur == len(arrStartVertices):
            break

        logging.debug("path {} @ read {}".format(path_num, vread_prop[v]))

        vpath_prop[v] = path_num

        (path, ends) = backbone(g, v,
                        vflags_prop, vread_prop, eflags_prop, eend_prop, ediv_prop, elen_prop, vpath_prop,
                        path_num, circular, lookahead, ilookahead)
        arrPaths.append( (path, path_num, v, ends) )

        if len(path) > 0:
            start = 0
            end = 0

            logging.info("found path consisting of {} reads (@ {} of {})".format(len(path) + 1,
                    vcur, len(arrStartVertices)))

            while len(path) > 0 and eflags_prop[ path[0] ] & (LAS.OVL_MODULE | LAS.OVL_OPTIONAL):
                del path[0]
                start += 1

            while len(path) > 0 and eflags_prop[ path[-1] ] & (LAS.OVL_MODULE | LAS.OVL_OPTIONAL):
                del path[-1]
                end += 1

            vflags_prop[ e_source(path[0]) ] |= V_PATH_END
            vflags_prop[ e_target(path[-1]) ] |= V_PATH_END

            logging.debug("path trimmed by {}..{}".format(start, end))

            if len(path) == 0:
                logging.debug("path empty after trimming")

            logging.debug("update backbone distance")

            update_bbdist(g, vflags_prop, eflags_prop, bbdist_prop)

        path_num += 1

    return arrPaths

def process_graph(g, dropinversions, circular, lookahead, ilookahead):
    smark_prop = {}     # source marker
    tmark_prop = {}     # target marker
    ecolor_prop = {}
    estyle_prop = {}
    vlabel_prop = {}
    elabel_prop = {}
    vcolor_prop = {}
    vstyle_prop = {}
    bbdist_prop = { key : -1 for key in g }
    vpath_prop = { key : -1 for key in g }

    vflags_prop = { key : 0 for key in g }

    edir_prop = {}
    ewidth_prop = { key : 1.0 for key in g.edges() }

    vsize_prop = { key : 0.2 for key in g }

    vquality_prop = { key : 0.0 for key in g }

    vread_prop = { n : g.nodes[n]["read"] for n in g }
    vopt_prop = {  n : int(g.nodes[n]["optional"]) for n in g }

    seen = set()

    for e in g.edges():
        key = ( e[0], e[1] )
        if key in seen:
            logging.debug("DUPE " + str(key))
        else:
            seen.add(key)

    eflags_prop = { e : int(g.edges[e[0], e[1]]["flags"]) for e in g.edges() }
    ediv_prop = { e : int(g.edges[e[0], e[1]]["divergence"]) for e in g.edges() }
    elen_prop = { e : int(g.edges[e[0], e[1]]["length"]) for e in g.edges() }
    eend_prop = { e : g.edges[e[0], e[1]]["end"] for e in g.edges() }
    epath_prop = { key : -1 for key in g.edges() }


    for e in g.edges():
        if vopt_prop[ e[0] ]:
            eflags_prop[e] |= LAS.OVL_OPTIONAL
            vflags_prop[ e[0] ] |= V_OPTIONAL

        if vopt_prop[ e[1] ]:
            eflags_prop[e] |= LAS.OVL_OPTIONAL
            vflags_prop[ e[1] ] |= V_OPTIONAL

    ### DEBUG - remove
    #

    if False:
        drop_n = set()
        drop_e = set()
        for e in g.edges():
            if vopt_prop[ e[0] ]:
                eflags_prop[e] |= LAS.OVL_OPTIONAL
                drop_n.add( e[0] )
                drop_e.add(e);

            if vopt_prop[ e[1] ]:
                eflags_prop[e] |= LAS.OVL_OPTIONAL
                drop_n.add( e[1] )
                drop_e.add(e);

        for e in drop_e:
            g.remove_edge(*e)

        for n in drop_n:
            g.remove_node(n)

    #
    ###

    if dropinversions:
        (v, e) = drop_inversions(g, eend_prop, eflags_prop)

        logging.debug("removed {0} vertices and {1} edges due to inversion".format(v, e))

    for e in g.edges():
        flags = eflags_prop[e]
        ediv_prop[e] = ediv_prop[e]
        edir_prop[e] = "both"

        end = eend_prop[e]

        if flags & LAS.OVL_MODULE:
            vflags_prop[e_source(e)] |= V_MODULE
            vflags_prop[e_target(e)] |= V_MODULE

        if end == 'l':
            smark_prop[e] = "inv"

            if flags & LAS.OVL_COMP:
                tmark_prop[e] = "normal"
            else:
                tmark_prop[e] = "inv"
        else:
            smark_prop[e] = "normal"

            if flags & LAS.OVL_COMP:
                tmark_prop[e] = "inv"
            else:
                tmark_prop[e] = "normal"

    # color edges according to the diff score

    logging.info("overlap divergence [{:2}%, {:2}%]".format(min( ediv_prop.values() ), max( ediv_prop.values() )))

    for e in g.edges():
        c = ediv_prop[e]
        c = tuple( [ int(c * 255.0) for c in colormap.map(c, COLORS_DIV_MIN, COLORS_DIV_MAX) ] )
        ecolor_prop[e] = "#%.2x%.2x%.2x" % c[:3]

    # flag dead ends

    nFlagged = mark_dead_ends(g, vflags_prop, eend_prop, eflags_prop)

    logging.info("found {} dead ends".format(nFlagged))

    # calculate quality of vertices

    logging.info("assigning read quality")

    arrStartVertices = []

    for v in g:
        fQuality = vertexEdgeQuality(g, v, eflags_prop, eend_prop, ediv_prop)
        vquality_prop[v] = fQuality

        arrStartVertices.append( (v, fQuality) )

    arrStartVertices.sort( key = operator.itemgetter(1) )

    ### compute paths

    arrPaths = graph_compute_paths(g, arrStartVertices, bbdist_prop,
                                   vflags_prop, vread_prop, vpath_prop,
                                   eflags_prop, eend_prop, ediv_prop, elen_prop,
                                   circular, lookahead, ilookahead)

    ### try to connect paths using optional edges

    (ends_used, potential_edges) = analyze_path_ends(g, arrPaths, vread_prop, eflags_prop, eend_prop)

    # potential_edges.extend( analyze_path_ends_v2(g, arrPaths, vread_prop, eflags_prop, eend_prop, ends_used) )
    # sys.exit(0)

    if len(arrPaths) > 0 and len(potential_edges) > 0:
        ### reset touring data

        bbdist_prop = { key : -1 for key in g }
        vpath_prop = { key : -1 for key in g }
        vflags_prop = { key : vflags_prop[key] & ~( V_PATH_END | V_BACKBONE | V_VISITED | V_RETRY ) for key in vflags_prop }
        # eflags_prop = { key : eflags_prop[key] & ~( E_INVERSION | E_BACKBONE ) for key in eflags_prop }
        eflags_prop = { key : eflags_prop[key] & ~( E_BACKBONE ) for key in eflags_prop }

        ### better start vertices

        temp = []
        arrPaths.sort( key = len, reverse = True )
        for (path, pid, vstart, ends) in arrPaths:
            if len(path) == 0:
                continue

            temp.append( (e_source( path[(len(path) - 1) // 2] ), 0) )

        arrStartVertices = temp + arrStartVertices

        ### enable optional edges/vertices

        for e in potential_edges:
            eflags_prop[e] &= ~ ( LAS.OVL_OPTIONAL )

        ### compute paths (again)

        arrPaths = graph_compute_paths(g, arrStartVertices, bbdist_prop,
                                    vflags_prop, vread_prop, vpath_prop,
                                    eflags_prop, eend_prop, ediv_prop, elen_prop,
                                    circular, lookahead, ilookahead)

    #### apply styles

    for v in g:
        vstyle_prop[v] = "filled"

        if vflags_prop[v] & V_DISCARD:
            vcolor_prop[v] = COLOR_GREY
        elif vflags_prop[v] & V_PATH_END:
            vcolor_prop[v] = COLOR_GREEN
        else:
            vcolor_prop[v] = COLOR_WHITE

        if vflags_prop[v] & V_BACKBONE:
            vlabel_prop[v] = "{:,} P{}".format(vread_prop[v], vpath_prop[v])
        else:
            vlabel_prop[v] = "{:,} D{}".format(vread_prop[v], bbdist_prop[v])

    for e in g.edges():
        estyle_prop[e] = "dotted"
        ewidth_prop[e] = 1.0

        if eflags_prop[e] & E_BACKBONE:
            estyle_prop[e] = "solid"
            ewidth_prop[e] = 2.0
            elabel_prop[e] = str( elen_prop[e] )

            vs = e_source(e)
            vt = e_target(e)

            if vpath_prop[vs] == vpath_prop[vt]:
                epath_prop[e] = vpath_prop[vs]
            else:
                if vflags_prop[vs] & V_PATH_END:
                    epath_prop[e] = vpath_prop[vs]
                elif vflags_prop[vt] & V_PATH_END:
                    epath_prop[e] = vpath_prop[vt]
                else:
                    logging.debug("WARNING edge path ambiguous n:{} t:{}".format(vs,vt))
                    epath_prop[e] = vpath_prop[vs]

        if eflags_prop[e] & LAS.OVL_OPTIONAL:
            estyle_prop[e] = "dashed"
            ecolor_prop[e] = "yellow"

        if eflags_prop[e] & LAS.OVL_MODULE:
            estyle_prop[e] = "dashed"
            ecolor_prop[e] = "purple"

        if eflags_prop[e] & E_INVERSION:
            ecolor_prop[e] = "magenta"

    #####

    e_apply_attributes(g, "color", ecolor_prop)
    e_apply_attributes(g, "dir", edir_prop)
    e_apply_attributes(g, "end", eend_prop)
    e_apply_attributes(g, "flags", eflags_prop)
    e_apply_attributes(g, "label", elabel_prop)
    e_apply_attributes(g, "style", estyle_prop)
    e_apply_attributes(g, "weight", ewidth_prop)
    e_apply_attributes(g, "path", epath_prop)

    v_apply_attributes(g, "color", vcolor_prop)
    v_apply_attributes(g, "flags", vflags_prop)
    v_apply_attributes(g, "label", vlabel_prop)
    v_apply_attributes(g, "path", vpath_prop)
    v_apply_attributes(g, "style", vstyle_prop)
    v_apply_attributes(g, "width", vsize_prop)

    return (g, arrPaths, bbdist_prop)

def drop_inversions(g, eend, eflags):
    setBadEdges = set()

    for v in g:
        dicTargetStats = {}

        for e in v_all_edges(g, v):

            if e_source(e) == v:
                t = e_target(e)
                end = eend[e]
            else:
                t = e_source(e)

                end = eend[e]

                if not (eflags[e] & LAS.OVL_COMP):
                    end = reverse(end)

            if t not in dicTargetStats:
                dicTargetStats[t] = [0, 0]

            if end == 'l':
                dicTargetStats[t][0] += 1
            else:
                dicTargetStats[t][1] += 1

        for t in dicTargetStats:
            (l, r) = dicTargetStats[t]

            if l > 0 and r > 0:
                setBadEdges.add( (v, t) )

    setDiscard = set()
    for e in g.edges():
        v1 = e_source(e)
        v2 = e_target(e)

        if (v1,v2) in setBadEdges or (v2,v1) in setBadEdges:
            setDiscard.add(e)

    for e in setDiscard:
        g.remove_edge(e[0],e[1])

    nEdges = len(setDiscard)

    setDiscard = set()

    for v in g:
        if len( list( v_all_edges(g, v) ) ) == 0: # v.in_degree() == 0 and v.out_degree() == 0:
            setDiscard.add(v)

    for v in reversed(sorted(setDiscard)):
        g.remove_node(v)

    nVertices = len(setDiscard)

    return (nVertices, nEdges)

def paths_write(paths, fpath_paths, fpath_rids):
    filePaths = open(fpath_paths, "w")
    setRids = set()

    for (p, pid, v, ends) in paths:
        filePaths.write("PATH {} {} {}".format(pid, ends[0], ends[1]))

        for e in p:
            filePaths.write(" {}-{}".format(e[0], e[1]))
            setRids.add( e[0] )
            setRids.add( e[1] )

        filePaths.write("\n")

    filePaths.close()

    fileRids = open(fpath_rids, "w")
    for rid in setRids:
        fileRids.write("{}\n".format(rid))

    fileRids.close()

def main():
    parser = argparse.ArgumentParser(description = "Tour overlap graph")

    parser.add_argument("database", help = "database name")
    parser.add_argument("graph", nargs = "+", help = "overlap graph")

    parser.add_argument("-c", "--circular", help = "allow circular paths", default = "true", action = "store_true")
    parser.add_argument("-d", "--dropinversions", help = "remove edges suspected to be due to inversions", action = "store_true")
    parser.add_argument("-l", "--lookahead", metavar = "n", help = "lookahead during touring (default {0})".format(DEF_PATH_LOOKAHEAD), default = DEF_PATH_LOOKAHEAD, type = int)
    parser.add_argument("--debug", help = argparse.SUPPRESS, action = "store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logformat = logging.Formatter("%(message)s")

        handler = logging.FileHandler("OGtour.log")
        handler.setFormatter(logformat)
        logging.getLogger().addHandler(handler)

        handler = logging.StreamHandler()
        handler.setFormatter(logformat)
        logging.getLogger().addHandler(handler)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(format = "%(message)s")


    ilookahead = args.lookahead + PATH_LOOKAHEAD_INCREASE_INVERSION

    # print("opening db {}".format(args.database))
    # db = DB.DB(args.database)

    for ig in args.graph:
        basename = ig.rpartition(".")[0]
        og = basename + ".tour.graphml"
        op = basename + ".tour.paths"
        orids = basename + ".tour.rids"

        logging.info("loading graph {}".format(ig))

        try:
            g = nx.read_graphml(ig)
        except Exception as e:
            logging.error("error: failed to load graph: " + str(e))
            sys.exit(1)

        logging.info("touring...")

        (g, paths, bbdist) = process_graph(g, args.dropinversions, args.circular, args.lookahead, ilookahead)

        setContigEnds = set()

        for (p, pid, v, ends) in paths:
            if len(p) == 0:
                continue

            setContigEnds.add( p[0][0] )
            setContigEnds.add( p[-1][1] )

        if len(paths) > 0:
            paths_write(paths, op, orids)

            logging.info("saving tour {}".format(og))

            """
            drop = []

            for e in g.edges():
                #if e[0] in setContigEnds or e[1] in setContigEnds:
                #    continue

                eflags = g.edges[e[0], e[1]]["flags"]

                if (eflags & LAS.OVL_OPTIONAL) or ((eflags & LAS.OVL_MODULE) and not (eflags & E_BACKBONE)):
                    drop.append(e)

            for e in drop:
                g.remove_edge(*e)

            drop = []

            for v in g:
                if len( v_all_edges(g, v) ) == 0 or bbdist[v] > 2:
                    drop.append(v)

            for v in drop:
                g.remove_node(v)
            """

            nx.write_graphml(g, og)

if __name__ == "__main__":
    if not marvel.config.has_recent_python():
        sys.exit(1)

    main()
