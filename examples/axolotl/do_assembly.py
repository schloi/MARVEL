#!/usr/bin/env python

import multiprocessing
import marvel
import marvel.config
import marvel.queue

### settings

DB         = "AXOLOTL_FIX"
COVERAGE   = 30

PARALLEL   = multiprocessing.cpu_count()

### patch raw reads

q = marvel.queue.queue(DB, COVERAGE, PARALLEL)

### run daligner to create overlaps 
q.plan("planDalign{db}")

# after all daligner jobs are finshied the dynamic repeat masker has to be shut down
# !!!!!!!!!!!!!!!!!!!!!!!!!! replace SERVER and PORT !!!!!!!!!!!!!!!!!!!!!!!!!!
q.single("{path}/DMctl -h HOST -p PORT shutdown")

### run LAmerge to merge overlap blocks  
q.plan("planMerge{db}")

### start scrubbing pipeline

##### for larger genomes (> 100MB) LAstitch can be run with the -L option (preload reads)
##### with the -L option two passes over the overlap files are performed: 
##### first to buffer all reads and a second time to stitch them 
##### otherwise the random file access can make LAstitch pretty slow. 
##### Another option would be, to keep the whole db in cache (/dev/shm/) 
q.block("{path}/LAstitch -f 50 {db} {db}.{block}.las {db}.{block}.stitch.las")

##### create quality and trim annotation (tracks) for each overlap block and merge them
q.block("{path}/LAq -T trim0 -s 5 -b {block} {db} {db}.{block}.stitch.las")
q.single("{path}/TKmerge -d {db} q")
q.single("{path}/TKmerge -d {db} trim0")
##### create a repeat annotation (tracks) for each overlap block and merge them
q.block("{path}/LArepeat -c {coverage} -b {block} {db} {db}.{block}.stitch.las")
q.single("{path}/TKmerge -d {db} repeats")
##### combine repeat track and maskr, maskc track from dynamic masking server 
q.single("{path}/TKcombine {db} frepeats repeats maskc maskr")

##### detects "borders" in overlaps due to bad regions within reads that were not detected 
##### in LAfix. Those regions can be quite big (several Kb). If gaps within a read are
##### detected LAgap chooses the longer part oft the read as valid range. The other part(s) are
##### discarded
##### option -L (see LAstitch) is also available     
q.block("{path}/LAgap -s 100 -t trim0 {db} {db}.{block}.stitch.las {db}.{block}.gap.las")

##### create a new trim1 track, (trim0 is kept)
q.block("{path}/LAq -u -t trim0 -T trim1 -b {block} {db} {db}.{block}.gap.las")
q.single("{path}/TKmerge -d {db} trim1")

##### based on different filter critera filter out: local-alignments, repeat induced-alifnments
##### previously discarded alignments, ....
##### -r repeats, -t trim1 ... use repeats and trim1 track
##### -n 500  ... overlaps must be anchored by at least 500 bases (non-repeats) 
##### -u 10   ... overlaps with more than 10 unaligned bases according to the trim1 intervall are discarded
##### -o 2000 ... overlaps shorter than 2k bases are discarded
##### -p      ... purge overlaps, overlaps are not written into the output file    
##### option -L (see LAstitch) is also available
q.block("{path}/LAfilter -p -s 100 -n 300 -r frepeats -t trim1 -o 1000 -u 10 {db} {db}.{block}.gap.las {db}.{block}.filtered.las")

##### merge all filtered overlap files into one overlap file
q.single("{path}/LAmerge -S filtered {db} {db}.filtered.las")

##### create overlap graph
q.single("!mkdir -p components")
q.single("{path}/OGbuild -t trim1 -s {db} {db}.filtered.las components/{db}")

##### tour the overlap graph and create contigs paths 
first = True
for fpath in glob.glob("components/*.graphml"):
    q.single("{path_scripts}/OGtour.py -c {db} {graph}", graph = fpath)
    if not first:
        q.merge()
    else:
        first = False

q.process()

##### create contig fasta files 
first = True
for fpath in glob.glob("components/*.paths"):
    fpath_graph = fpath.replace(".tour.paths", ".graphml")
    q.single("{path_scripts}/tour2fasta.py -t trim1 {db} {graph} {paths}", graph = fpath_graph, paths = fpath)

    if not first:
        q.merge()
    else:
        first = False

q.process()
