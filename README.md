
# UPDATE 2023-07

Here we go, a fully working GPU/CUDA-based overlapper for blazing fast alignments. The code has been sitting on my laptop for a while now and since I most likely won't write a paper on the GPU-based alignment code, here it is for your enjoyment.

PS. the build system switched to cmake. I will update the build instructions below when I get around to it.

# The MARVEL Assembler

MARVEL consists of a set of tools that facilitate the overlapping, patching, correction and assembly of noisy (not so noisy ones as well) long reads.

## CONTACT

For questions, suggestions and complaints please contact [s.schloissnig@gmail.com](s.schloissnig@gmail.com)

## REQUIREMENTS

- GTK3 3.x (optional)
- libhdf5 (optional)
- networkx 2.1+ (Python library)

## SOURCE CODE ORGANIZATION

Most of the codebase is in C with some utility scripts written in Python. Often Python is used to develop prototypes and which are then moved to a C implementation in case performance is sub-par or spare time is available. Due to this two track development Python based interfaces to access the most commonly used file formats the various tools produce are offered.

For production purposes the use of Python to deal with las files is discouraged. Due to the performance overhead involved.

MARVEL is largely self-contained, meaning that external code packages/libraries are rarely relied on. The only external dependencies are the HDF5 library, GTK3 and networkx. All of which are available through the package management system of the most popular linux distributions. If that is not the case for our platform, you need to have a look at [https://www.hdfgroup.org/downloads/hdf5/](https://www.hdfgroup.org/downloads/hdf5/), [https://developer.gnome.org/gtk3/3.0/](https://developer.gnome.org/gtk3/3.0/) and [https://networkx.github.io/](https://networkx.github.io/). Please note that the build will not fail of those libraries are not present, but rather skip the compilation of the tools depending on them.

The build system is based on automake/conf and build utility scripts are located in the build/ subdir.

    build/          build utility scripts
    corrector/      read correction
    dalign/         fork of Gene's daligner
    db/             fork of the DB module underlying Gene's daligner package
    docs/           documentation
    examples/       example datasets and assembly scripts
    hic/            various tools for dealing with Hi-C data
    lib/            utility functions
    lib.ext/        utility functions based on external code
    lib.python/     Python modules for interacting with the assembler's data files
    m4/             autoconf macros
    msa/            multiple sequence aligner
    scripts/        various Python based utility scripts
    scrub/          read scrubbing
    touring/        overlap graph construction and touring
    utils/          various C based utilities

For specific information on the db and dalign packages have a look at their respective READMEs.

## INSTALLATION

1. in case your source package didn't come with a "configure" script you need to run "autoreconf".
2. run "configure --prefix=<marvel.install.dir>" replacing <marvel.install.dir> with the path where MARVEL should be installed
3. make
4. make install
5. add <marvel.install.dir>/lib.python to your python's module search path PYTHONPATH

In case configure is not able to locate your hdf5 and/or gtk installation you might need to set

    export CPPFLAGS='-I/path/to/include'
    export LDFLAGS='-L/path/to/lib'

Alternatively you can pass the location of the h5cc/h5pcc binary to configure using the --with-hdf5 argument.

After performing the above steps you will find MARVEL installed in <marvel.install.dir> with all binaries in the bin/, scripts in the scripts/ and python modules in the lib.python/ subdir. Note that these three directories are symlinks to dirs inside another directory in <marvel.install.dir>. This subdir is named either according to the current git revision hash (in case you checked the code out directly from the git) or contains the version number.

## USAGE

The assembly process can be summarized as follows:

1. overlap
2. patch reads
3. overlap (again)
4. scrubbing
5. assembly graph construction and touring
6. optional read correction
7. fasta file creation

Example: Assembly of an E.coli dataset (see the included examples/e.coli_pacbio)

This assembly can savely be run a reasonbly fast laptop.

In order to initially set up the database containing the reads you need to run

<marvel.install.dir>/scripts/DBprepare.py ECOL p6.25x.fasta

This will create ECOL.db and two hidden files .ECOL.idx and .ECOL.bps in the current working directory and two plans containing the statements neccessary to perform the overlapping.

The assembly can now be performed by running the included do.py script. Which will use all available processor cores on the machine it is running at to perform the assembly.

The do.py script only serves as an example of how to perform an assembly using MARVEL and can be changed to your liking or replaced altogether with a custom setup of your own.
```python
# set various constants

DB         = "ECOL"                         # database name
COVERAGE   = 25                             # coverage of the dataset
DB_FIX     = DB + "_FIX"                    # name of the database containing the patched reads
PARALLEL   = multiprocessing.cpu_count()    # number of available processors

### patch raw reads

q = marvel.queue.queue(DB, COVERAGE, PARALLEL)

# run daligner to create initial overlaps
q.plan("{db}.dalign.plan")

# run LAmerge to merge overlap blocks
q.plan("{db}.merge.plan")

# create quality and trim annotation (tracks) for each overlap block
q.block("{path}/LAq -b {block} {db} {db}.{block}.las")

# since q and trim tracks have been produced for each block, we need to merge them
q.single("{path}/TKmerge -d {db} q")
q.single("{path}/TKmerge -d {db} trim")

# run LAfix to patch reads based on overlaps
# the -g parameter specifies the maximum gap size that gets patched
q.block("{path}/LAfix -g -1 {db} {db}.{block}.las {db}.{block}.fixed.fasta")

# join all fasta files containing the repaired reads
q.single("!cat {db}.*.fixed.fasta > {db}.fixed.fasta")

# create a new database with them
# -s ... block size in megabytes/bases
# -r ... run id
# -j ... threads used by the overlapper
# -g ... genome size
q.single("{path_scripts}/DBprepare.py -s 50 -r 2 -j 4 -g 4600000 {db_fixed} {db}.fixed.fasta", db_fixed = DB_FIX)

# run the commands build using the above statements
q.process()

# alternatively you can dump the set of statements build to a text file
# that you then lateron could use for running jobs on a queuing system
# q.dump("statements.txt")

# build a new queue for the assembly of the patched reads
q = marvel.queue.queue(DB_FIX, COVERAGE, PARALLEL)

# put the daligner statements in the queue
q.plan("{db}.dalign.plan")

# put the LAmerge statements in the queue
q.plan("{db}.merge.plan")

# stitch reads, this will repair/join alignments that have split into two (or more)
# due to leftover weak regions in the reads
q.block("{path}/LAstitch -f 50 {db} {db}.{block}.las {db}.{block}.stitch.las")

# create quality and trim annotation for each overlap block
q.block("{path}/LAq -s 5 -T trim0 -b {block} {db} {db}.{block}.stitch.las")

# and merge them
q.single("{path}/TKmerge -d {db} q")
q.single("{path}/TKmerge -d {db} trim0")

# create a repeat annotation for each overlap block
q.block("{path}/LArepeat -c {coverage} -b {block} {db} {db}.{block}.stitch.las")

# and merge them
q.single("{path}/TKmerge -d {db} repeats")

# detect chimeric breaks and other leftover structural problems with the reads/alignments
# that result in "gap", ie. regions that are not spanned by any alignment and discard
# all alignments on one side of the gap
q.block("{path}/LAgap -t trim0 {db} {db}.{block}.stitch.las {db}.{block}.gap.las")

# create a new trim annotation, since LAgap might have resulted in regions of some reads
# not being overed by an alignments (ie. appearing like "dead" sequence)
q.block("{path}/LAq -s 5 -u -t trim0 -T trim1 -b {block} {db} {db}.{block}.gap.las")

# merge the block-based tracks
q.single("{path}/TKmerge -d {db} trim1")

# filter all non true-overlap induced alignments
# -r ... which repeat track to use
# -t ... which trim track to use
# -T ... actually apply the trim (ie. update alignments based on its information)
# -o ... discard all alignments shorter than 2k
# -u ... discard all local alignments with >0 unaligned bases
# -n ... require alignments that span a repeat to cover at least 300 non-repeat annotated bases
# -p ... actually discard the filtered alignments, and not just flag them as discarded
q.block("{path}/LAfilter -n 300 -r repeats -t trim1 -T -o 2000 -u 0 {db} {db}.{block}.gap.las {db}.{block}.filtered.las")

# merge all filtered overlap files
q.single("{path}/LAmerge -S filtered {db} {db}.filtered.las")

# create the overlap graph
q.single("{path}/OGbuild -t trim1 {db} {db}.filtered.las {db}.graphml")

# tour the overlap graph and create contigs paths
q.single("{path_scripts}/OGtour.py -c {db} {db}.graphml")

# correct the reads used in the touring of the overlap graph
# and put them into a new database
# -j ... number of threads to use
# -r ... only correct the reads with the ids contained in ECOL_FIX.tour.rids
# Note: LAcorrect could be run directly after the initial overlapping in order to produce a set
#       of corrected reads
q.single("{path}/LAcorrect -j 4 -r {db}.tour.rids {db} {db}.filtered.las {db}.corrected")
q.single("{path}/FA2db -c {db}_CORRECTED [expand:{db}.corrected.*.fasta]")

# output fasta files of the paths found in the touring
# -c ... use the corrected reads, of not present the contigs would be built using the
#        the uncorrected (patched) reads
# -t ... which trim track to use
q.single("{path_scripts}/tour2fasta.py -c {db}_CORRECTED -t trim1 {db} {db}.tour.graphml {db}.tour.paths")

# optional: calculate a layout (arrange the graph's nodes in such a way that they are easy to look at)
#           for the toured overlap graph and write it to a dot file
q.single("{path}/OGlayout -R {db}.tour.graphml {db}.tour.layout.dot")

# run the commands
q.process()
```

## CONCEPTS & NOMENCLATURE

### Database

The files (.db .idx and .bps) containing the sequences of the read and keeping track of the partitioning
of them into blocks.

### LAS files

Contains the alignment records and their trace points and status flags.

### Annotation tracks

Keep track of read specific information like quality values, repeat annotation, trim information ...

### A/B reads

In the context of alignments and overlaps the two reads participating in it are refered to as the A
and B read.

### Alignment trace points

Since storing the alignments explicitely would result in excessive storage requirements, only the position of the alignments at so called trace points is kept track of. For the A read they are evenly spaced with the trace width (usually 100bp) and give the offset from the previous trace point in B and the number of differences (edit distance).

For example, the alignment starts at (a.begin, b.begin) the first trace point would contain
(a.begin + 100, b.begin + offset.b.1, diffs.1),
the second (a.begin + 200, b.begin + offset.b.1 + offset.b.2, diffs.2), ... and so forth.

### Read quality

Based on the trace points and all alignments to a given A read be can now use the trace points of all alignments to infer the quality of the A read by averaging across them. Thereby assigning quality values to trace-width-sized segments of the A read.

### Pre-loading (the -L parameter)

Some tools scan the las files sequentially and require both the sequence of the A and B read. The A read sequence essentially can be read sequentially from the file containing the sequences. But reading the B read induces a random-access pattern, which results in sub-par performance on many (distributed) file systems. For the tools which require the B reads, the -L option is offered, often called the pre-loading option. When this option is used, an addition sequential scan of the las file is performed, in which the identifiers of all reads that need to be loaded are collected, then subsequently fetched from the database in one sequential pass and cached in memory.

### Stitching

A->B alignments can be split into multiple A->B records due to bad regions in either one of the reads, causing the overlapper to stop aligning. LAstitch looks for such records (with a maximum gap of -f), joins the two alignments into a single record, discards the superfluous one, and (re-)computes pass-through points & diffs for the segments surrounding the break point.

### Uncorrected assembly

MARVEL breaks with the established paradigm of correcting the reads to >99% identity prior to assembly. It is vital for read correction that the reads used to derive the consensus (i.e. corrected) sequence are not taken from repeat induced alignments. However, given the knowledge of which alignments constitute true overlaps, assembling immediately without prior correction is achievable. This has the added advantage of not constructing hybrid sequences, that represent essentially the “average” of the repeats. The probability of creating hybrid sequences is correlated with the repeat content of the genome. For genomes featuring a non-excessive repeat content, hybrid sequences can largely be avoided by employing heuristics that enrich for true overlaps, such as using the longest alignments for correction. But this approach starts breaking down rapidly with >50% repeat content, massive heterozygosity or polyploidy. The resulting hybrid sequences subsequently cause problems in the assembly pipeline, resulting in sub-par assemblies.

### Patching (in lieu of correction)

Instead of correcting reads, a process called patching patching is employed. Therein, large scale errors (missed adaptors, large “random” inserts, high-error regions, missing sequence) that result in a stop of the alignments one it reaches said erroneous regions are repaired. The main point here is, that when restoring large scale errors with uncorrected parts of others reads, the sequence used for the correction only needs to be good enough to allow the alignment to continue at the native error rate and doesn’t actually have to be the right one.

Since the chance of encountering a large-scale error in a read goes up with its length the most valuable long reads are most afflicted by them and therefore patching represents an important way to preserve the availability of long reads for use in assembly. Patching essentially works by finding regions in reads that are

a. of low quality. The quality (i.e. error rate) across a read is derived from the identity of the alignments to it.
b. not spanned by any alignment.

a. Is corrected by taking the sequence of a read that spans the low-quality region and replacing the low-quality region with it.
b. Look to the left/right of the break and see if the same reads are on both sides and that the size of the gap from the perspective of those reads is roughly the same. The region in the read is then replaced by sequence taken from one of the reads that align to it.

### Post-assembly correction

MARVEL performs read correction post-assembly. This ensures that true overlaps are used for the correction and only the reads used to build the contigs need to be corrected, thereby resulting in often significant compute time savings. The correction module can also be used without performing an assembly to produce a complete set of corrected reads, if that is desired

### Dynamic masking

Large repeat-rich genomes often result in excessive CPU and storage requirements. We added an on-the-fly repeat masker that works jointly with the overlapper to lower said requirements. On the server side you need to launch the dynamic masking server DMserver on a dedicated node (with enough main memory) and on the client side the overlapper is told where the server runs using the -D argument. When the overlapper starts processing a new block, it retrieves a masking track from the server (network traffic is negligible) and uses the track to soft mask intervals in the block, thereby excluding them from the k-mer seeding. When a block comparison is finished, the overlapper notifies the server of the location of the resulting .las file(s) (note that overlapper and masking server need to have a shared filesystem for that to work). The server processes the alignments and updates internal data structures which maintain data on the reads' repetitiveness.

For additional information please refer to [docs/HOWTO-dynamic-masking.txt](docs/HOWTO-dynamic-masking.txt).
