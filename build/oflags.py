#!/usr/bin/env python

import sys
import re

if len(sys.argv) != 3:
    print("usage: oflags.py input.h output.py")
    sys.exit(1)

maxshift = 0

fout = open(sys.argv[2], "w")

for line in open(sys.argv[1]):
    m = re.match(r"#define\s+(OVL_\w+)\s+\(\d+\s+<<\s+(\d+)\)", line)

    if m is None:
        continue

    name = m.group(1)
    shift = int( m.group(2) )

    if shift > maxshift:
        maxshift = shift

    fout.write("%s = (1 << %d)\n" % (name, shift))

fout.write("OVL_CUSTOM = (1 << %d)\n" % (maxshift + 1))

fout.close()

