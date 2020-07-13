
bunch of helper scripts

**scripts/contacts_filter.py**

Takes a file containing contacts and only keeps the ones that link contigs as specified in the passed file.

**scripts/contacts_scaffold_relative.py**

Given a scaffold and contig relative contacts produces a new set of contacs relative to the scaffold.

**scripts/contigs_assign.py**

Given a fasta file containing the list of contigs, the top interactors of each contig (created by the assign tool) and a list of existing scaffolds, assigns contigs to scaffolds if they uniquely belong to it. Updates scaffolds are written to the specified directory.
