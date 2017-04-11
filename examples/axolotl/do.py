#!/usr/bin/env python

import multiprocessing
import marvel
import marvel.config
import marvel.queue

### settings

DB         = "AXOLOTL"
COVERAGE   = 30

DB_FIX     = DB + "_FIX"
PARALLEL   = multiprocessing.cpu_count()

### patch raw reads

q = marvel.queue.queue(DB, COVERAGE, PARALLEL)

### run daligner to create initial overlaps 
q.plan("planDalign{db}")
### run LAmerge to merge overlap blocks  
q.plan("planMerge{db}")

# create quality and trim annotation (tracks) for each overlap block
q.block("{path}/LAq -b {block} {db} {db}.{block}.las")
# merge quality and trim tracks 
q.single("{path}/TKmerge -d {db} q")
q.single("{path}/TKmerge -d {db} trim")
# run LAfix to patch reads based on overlaps
q.block("{path}/LAfix -c -x 2000 {db} {db}.{block}.las {db}.{block}.fixed.fasta")
# create a new Database of fixed reads
q.single("{path}/DB2fasta -v -x 2000 {db_fixed} {db}.*.fixed.fasta", db_fixed = DB_FIX)
q.single("{path}/DBsplit {db_fixed}", db_fixed = DB_FIX)
q.single("{path}/DBdust {db_fixed}", db_fixed = DB_FIX)
# merge contained and repeat track 
q.single("{path}/TKcombine {db_fixed} mask maskr maskc dust", db_fixed = DB_FIX)
# create daligner and merge plans, replace SERVER and PORT
q.single("{path}/HPCdaligner -v -t 100 -D SERVER:PORT -m mask -r2 -j16 --dal 32 --mrg 32 -o {db_fixed} {db_fixed}", db_fixed = DB_FIX)

q.process()

##### assemble patched reads

q = marvel.queue.queue(DB_FIX, COVERAGE, PARALLEL)

### run daligner to create overlaps 
q.plan("planDalign{db}")
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
q.block("{path}/LAfilter -p -n 500 -r repeats -t trim1 -o 2000 -u 10 {db} {db}.{block}.gap.las {db}.{block}.filtered.las")

##### merge all filtered overlap files into one overlap file
q.single("{path}/LAmerge -S filtered {db} {db}.filtered.las")

##### create overlap graph
q.single("{path}/OGbuild -t trim1 {db} {db}.filtered.las {db}.graphml")
##### tour the overlap graph and create contigs paths 
q.single("{path_scripts}/OGtour.py -c {db} {db}.graphml")
##### create contig fasta files 
q.single("{path_scripts}/tour2fasta.py -t trim1 {db} {db}.tour.graphml {db}.tour.paths")

### optional: create a layout of the overlap graph which can viewed in a Browser (svg) or Gephi (dot)
q.single("{path}/OGlayout -R {db}.tour.graphml {db}.tour.layout.svg")
# q.single("{path}/OGlayout -R {db}.tour.graphml {db}.tour.layout.dot")

q.process()
