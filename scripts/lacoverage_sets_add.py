#!/usr/bin/env python3

import argparse
import sys

def main():
    data = []

    for pathin in sys.argv[1:]:
        fin = open(pathin)

        for line in fin:
            items = line.strip().split()
            ltype = items[0]
            lvalues = [ int(x) for x in items[1:] ]

            if ltype == "MAX":
                continue

            if lvalues[0] == len(data):
                data.append( lvalues[1:] )
            else:
                for i in range( len(lvalues) - 1 ):

                    data[ lvalues[0] ][i] += lvalues[i + 1]

    values_idx = [0] * len( data[0] )
    values_max = [0] * len( data[0] )

    i = 0
    for values in data:
        for j in range(len(values)):
            if values[j] > values_max[j]:
                values_max[j] = values[j]
                values_idx[j] = i

        i += 1

    print("MAX {}".format(" ".join([str(x) for x in values_idx])))

    i = 0
    for values in data:
        print("HIST {} {}".format(i, " ".join([str(x) for x in values])))
        i += 1

if __name__ == "__main__":
    main()