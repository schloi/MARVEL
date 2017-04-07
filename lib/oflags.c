
#include <stdlib.h>

#include "oflags.h"

OverlapFlag2Label oflag2label[] = {
                                    {OVL_COMP,      "complement",          '<'},

                                    {OVL_DISCARD,   "discarded",           'x'},

                                    {OVL_REPEAT,    "repeat",              '*'},
                                    {OVL_STITCH,    "stitched",            '+'},
                                    {OVL_TRIM,      "trimmed",             '-'},
                                    {OVL_CONT,      "contained",           'c'},
                                    {OVL_GAP,       "gap",                 '|'},
                                    {OVL_OLEN,      "overlap length",      'o'},
                                    {OVL_RLEN,      "read length",         'r'},
                                    {OVL_LOCAL,     "local alignment",     'l'},
                                    {OVL_DIFF,      "divergence",          'd'},

                                    {OVL_SYMDISCARD, "symmetric discard",  '#'},
                                    {OVL_TEMP,      "temporary flag",      't'},

                                    {OVL_MODULE,    "module",              'm'},
                                    {OVL_OPTIONAL,  "optional",            '?'},

                                    {0,             NULL,                  '\0'}
                                 };

void flags2str(char* pc, int flags)
{
    int i;
    int last = -1;
    for (i = 0; oflag2label[i].mask; i++)
    {
        if ( flags & oflag2label[i].mask )
        {
            pc[i] = oflag2label[i].indicator;
            last = i;
        }
        else
        {
            pc[i] = ' ';
        }
    }

    pc[last+1] = '\0';
}

