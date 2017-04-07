#!/usr/bin/env python

#
# print block number for a given read id
#

import sys

if len(sys.argv) != 3:
    print("usage: <db> <read.id>")
    sys.exit(1)

strDb = sys.argv[1]

if not strDb.endswith(".db"):
    strDb = strDb + ".db"

nId = int( sys.argv[2] )

nBlock = 0
nOffReal_p = nOffTrim_p = 0

for strLine in open(strDb):
    if nBlock == 0:
        if strLine.startswith("size = "):
            nBlock = 1

        continue

    nOffReal = int(strLine.strip())

    if  nId >= nOffReal_p and nId < nOffReal:
        print( format(nBlock - 1) )
        break

    nOffReal_p = nOffReal

    nBlock += 1

