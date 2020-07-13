
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FaiEntry:
    length : int = 0
    offset : int = 0

@dataclass
class Fai:
    entries : Dict[ str, FaiEntry ] = field( default_factory = dict )

def readfai(pathfa):
    fai = Fai()

    if not pathfa.endswith(".fai"):
        pathfai = pathfa + ".fai"
    else:
        pathfai = pathfa

    try:
        faifile = open(pathfai, "r")
    except:
        print("failed to open".format(pathfai))
        return fai

    for line in faifile:
        items = line.strip().split()
        contig = items[0]
        clen = int(items[1])
        coff = int(items[2])

        fai.entries[ contig ] = FaiEntry(clen, coff)

    return fai

