/*******************************************************************************************
 *
 *  Date  :  January 2014
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <assert.h>

#include "lib/stats.h"
#include "lib/tracks.h"
#include "lib/oflags.h"
#include "lib/colors.h"
#include "lib/anno.h"

#include "borders.h"
#include "lib/pass.h"


// switches

#undef BORDER_EXTEND

#ifdef BORDER_EXTEND
#define BORDER_EXTEND_DISTANCE 100
#endif


void find_borders(Border** ppBorder, int* bmax, int* bcur,
                  Event** pEvents, int l, int r, 
                  float min_density, int min_events, int max_dist)
{
    int len = r - l + 1;
    
    /*
    static int indent = -1;
    
    indent++;
    printf("%*sfind_borders %d %d %d..%d %d\n", 
                indent, "", 
                l, r, 
                pEvents[l]->pos, pEvents[r]->pos, 
                pEvents[r]->pos - pEvents[l]->pos);
    
    if (len < min_events)
    {
        printf("%*smin_events\n", indent, "");
        indent--;
        return ;
    }
    */

    if (len < min_events)
    {
        return ;
    }

    int dist = pEvents[r]->pos - pEvents[l]->pos;
    double density;
    
    if (dist == 0)
    {
        density = min_density + 1;
    }
    else
    {
        density = (double)len / dist;
    }
    
    if (density > min_density && dist < max_dist)
    {
        if ( *bcur == *bmax )
        {
            *bmax = *bmax * 2 + 20;
            *ppBorder = (Border*)realloc(*ppBorder, sizeof(Border) * (*bmax));
        }
 
#ifdef BORDER_EXTEND
        if ( (*bcur) > 0 )
        {
            Border* prev = (*ppBorder) + (*bcur) - 1;
            
            if ( pEvents[l]->pos - pEvents[ prev->ee ]->pos < BORDER_EXTEND_DISTANCE )
            {
                prev->ee = r;
                
                return;
            }
        }
#endif

        Border* b = (*ppBorder) + (*bcur);

        *bcur += 1;

        b->peb = pEvents[l];
        b->pee = pEvents[r];
        
        b->eb = l;
        b->ee = r;
        b->link = -1;
        b->type = pEvents[l]->type;
        b->done = 0;
        
        // printf("%*sborder\n", indent, "");
    }
    else
    {    
        // recurse around pair with max distance
    
        int mid = l + 1;
        int dist_max = pEvents[mid]->pos - pEvents[mid-1]->pos;
    
        int i;
        for (i = mid+1; i <= r; i++)
        {
            if ( pEvents[i]->pos - pEvents[i-1]->pos > dist_max )
            {
                mid = i;
                dist_max = pEvents[i]->pos - pEvents[i-1]->pos;
            }
        }
    
        find_borders(ppBorder, bmax, bcur,
                     pEvents, l, mid-1, min_density, min_events, max_dist);
                 
        find_borders(ppBorder, bmax, bcur,
                     pEvents, mid, r, min_density, min_events, max_dist);
    }
    
    // indent--;
}
