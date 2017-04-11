#!/usr/bin/env python

import multiprocessing
import marvel
import marvel.config
import marvel.queue

### settings

DB         = "SMED_P6"
COVERAGE   = 29

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
q.block("{path}/LAfix -c -x 4000 {db} {db}.{block}.las {db}.{block}.fixed.fasta")

q.process()