Various tools for dealing with Hi-C data in the context of genome assembly.

Contacts, often refered to as links in this code bundle, are expected to be stored in a text having the following format:

	seq.name.1 position.1 mapq.1 seq.name.2 position.2 mapq.2

**assign**

Produces the list of sequences' best interactors based on the contacts data.

Usage:

    assign [-s n] -L links.in [... links.in] -f fasta.in -o counts.out
      -s n ... minimum sequence length (default 20000)
      -L f ... links files

**dump_matrix**

Takes an indexed fasta file and a file containing the contacts and outputs a binned matrix representation of the contact counts.

Usage:

    dump_matrix [-qb n] [-s seqname] [[-s seqname] ...] <fasta.in> <contacts.in> <matrix.out>
      -q n    ... MAPQ cutoff (default 1)
      -b n    ... bin size (default 1000)
      -n      ... include non-self hits

**misjoin**

Detects potential misjoins and writes the coordinates of the misjoined segments to segments.out.

Usage:

    misjoins [-qs n] <fasta.in> <links.in> <segments.out>
      -q n    ... MAPQ cutoff (default 1)
      -s n    ... minimum sequence length (default 20000)

**scaffold_graph**

Scaffolds the sequences as given in fasta.in using their associated contact data in links.in. Links need to be passed to the scaffolder with the recognition sequence of the restriction enzyme used.

The list of digest sites needs to be passed to scaffolder as well and is expected to be formatted like:

	sequence_name recognication_sequence position.1 position.2 ...
	...

Output format:

	# merge tree
	# [+-] path_id/contig <depth> <length of path/contig>
	# ...
	> top_level_path_1
	[+-] sequence
	...
	# merge tree
	> top_level_path_2
	[+-] sequence
	...

Usage:

	scaffold_graph [-clms n] [-g file.in] [-p string] [-o path.prefix] -L string links.in [... links.in] [-L ...] -f fasta.in -d digest_sites.in
	  -c n ... min amount of clusters
	  -l n ... minimum number of links (default 10)
	  -m n ... MAPQ cutoff (default 1)
	  -s n ... minimum sequence length (default 20000)
	  -g f ... guide scaffold
	  -w n ... guide scaffold window size
	  -p s ... scaffold names prefix
	  -o f ... output scaffolds to f
	  -L s f . digest site (e.g. GATC) and associated links files
