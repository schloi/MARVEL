
#include <stdlib.h>
#include <sys/param.h>
#include <assert.h>
#include <string.h>

#include "trim.h"
#include "lib/oflags.h"
#include "lib/utils.h"
#include "lib/colors.h"

#define DEBUG_VALIDATE
#undef DEBUG_TRIM

static void align(TRIM* trim, Overlap* ovl, int ab, int ae, int bb, int be)
{
#ifdef DEBUG_TRIM
    printf("align %d %d %d %d", ab, ae, bb, be);
#endif

    Alignment* align = &(trim->align);
    Path* path = trim->align.path;

    if(trim->rl)
    {
    	Read_Loader* rl = trim->rl;

    	rl_load_read(rl, ovl->aread, align->aseq, 0);
    	rl_load_read(rl, ovl->bread, align->bseq, 0);
    }
    else
    {
    	Load_Read(trim->db, ovl->aread, align->aseq, 0);
    	Load_Read(trim->db, ovl->bread, align->bseq, 0);
    }

    align->alen = DB_READ_LEN(trim->db, ovl->aread);
    align->blen = DB_READ_LEN(trim->db, ovl->bread);

    if (ovl->flags & OVL_COMP)
    {
        Complement_Seq(trim->align.bseq,align->blen);
    }

    path->diffs = (ae - ab) + (be - bb);

    path->abpos = ab;
    path->aepos = ae;
    path->bbpos = bb;
    path->bepos = be;

    Compute_Trace_ALL(align, trim->align_work);

#ifdef DEBUG_TRIM
    printf(" -> %d\n", path->diffs);
#endif

}


static void eval_trace(TRIM* trim, int stop_b, int* stop_a, int* diff_left, int* diff_right)
{
    // int twidth = trim->twidth;
    int a = trim->align.path->abpos;
    int b = trim->align.path->bbpos;
    int p, t;
    int diffs = 0;
    int matches = 0;

    // int aprev = a;
    // int bprev = b;
    // int tcur = 0;

    for (t = 0; t < trim->align.path->tlen; t++)
    {
        if ((p = ((int*)(trim->align.path->trace))[t]) < 0)
        {
            p = -p - 1;
            while (a < p)
            {
                if (trim->align.aseq[a] != trim->align.bseq[b]) diffs++;
                else matches++;

                a += 1;
                b += 1;

                if (b == stop_b)
                {
                    *stop_a = a;
                    *diff_left = diffs;
                    diffs = 0;
                }
            }

            diffs++;
            b += 1;

            if (b == stop_b)
            {
                *stop_a = a;
                *diff_left = diffs;
                diffs = 0;
            }
        }
        else
        {
            p--;

            while (b < p)
            {
                if (trim->align.aseq[a] != trim->align.bseq[b]) diffs++;
                else matches++;

                a += 1;
                b += 1;

                if (b == stop_b)
                {
                    *stop_a = a;
                    *diff_left = diffs;
                    diffs = 0;
                }
            }

            diffs++;
            a += 1;
        }
    }

    p = trim->align.path->aepos;
    while (a < p)
    {
        if (trim->align.aseq[a] != trim->align.bseq[b]) diffs++;
        else matches++;

        a += 1;
        b += 1;

        if (b == stop_b)
        {
            *stop_a = a;
            *diff_left = diffs;
            diffs = 0;
        }
    }

    *diff_right = diffs;
}


#ifdef DEBUG_TRIM

static void print_trace(Overlap* ovl)
{
    ovl_trace* trace = ovl->path.trace;
    int tlen = ovl->path.tlen;

    int b = ovl->path.bbpos;

    int i;
    for (i = 0; i < tlen; i += 2)
    {
        if ( i > 0 && i%10 == 0 )
        {
            printf("\n");
        }

        printf("%5d (%3d %3d) ", b, trace[i], trace[i+1]);

        b += trace[i+1];
    }

    printf("%5d\n", b);
}

#endif

void trim_overlap(TRIM* trim, Overlap* ovl)
{
    ovl_trace* trace = ovl->path.trace;

    trim->nOvls++;
    trim->nOvlBases += (ovl->path.aepos - ovl->path.abpos) + (ovl->path.bepos - ovl->path.bbpos);

    int a = ovl->aread;
    int b = ovl->bread;

    int trim_a_left, trim_a_right;
    get_trim(trim->db, trim->track, a, &trim_a_left, &trim_a_right);

    int trim_b_left, trim_b_right;
    get_trim(trim->db, trim->track, b, &trim_b_left, &trim_b_right);

    if (trim_a_left >= trim_a_right || trim_b_left >= trim_b_right)
    {
        ovl->flags |= OVL_DISCARD | OVL_TRIM;
        return ;
    }

    if (ovl->flags & OVL_COMP)
    {
        int tmp = trim_b_left;

        int ovlBLen = DB_READ_LEN(trim->db, ovl->bread);
        trim_b_left = ovlBLen - trim_b_right;
        trim_b_right = ovlBLen - tmp;
    }

    int abt = MAX(trim_a_left, ovl->path.abpos);
    int aet = MIN(trim_a_right, ovl->path.aepos);
    int bbt = MAX(trim_b_left, ovl->path.bbpos);
    int bet = MIN(trim_b_right, ovl->path.bepos);

    if (abt >= aet || bbt >= bet)
    {
        ovl->flags |= OVL_DISCARD | OVL_TRIM;
        return ;
    }

    if (abt == ovl->path.abpos && aet == ovl->path.aepos &&
        bbt == ovl->path.bbpos && bet == ovl->path.bepos)
    {
        return ;
    }

#ifdef DEBUG_TRIM
    printf("%6d (%5d) %c %6d (%5d) " ANSI_COLOR_GREEN "OVL" ANSI_COLOR_RESET " %5d..%5d %5d..%5d " ANSI_COLOR_GREEN "TRIM" ANSI_COLOR_RESET " %5d..%5d %5d..%5d\n",
            a, DB_READ_LEN(trim->db, ovl->aread), ovl->flags & OVL_COMP ? 'c' : 'n', b, DB_READ_LEN(trim->db, ovl->bread),
            ovl->path.abpos, ovl->path.aepos,
            ovl->path.bbpos, ovl->path.bepos,
            trim_a_left, trim_a_right,
            trim_b_left, trim_b_right);
#endif

    if (abt != ovl->path.abpos || aet != ovl->path.aepos)
    {
        int seg_old = ovl->path.abpos / trim->twidth;
        int seg_new = abt / trim->twidth;

        ovl->path.tlen -= 2 * (seg_new - seg_old);

        int j = 0;
        while (seg_new != seg_old)
        {
            ovl->path.bbpos += trace[j+1];

            j += 2;
            seg_old++;
        }

        trace += j;
        ovl->path.trace += sizeof(ovl_trace) * j;

        seg_old = (ovl->path.aepos - 1) / trim->twidth;
        seg_new = (aet - 1) / trim->twidth;

        j = ovl->path.tlen - 2;
        while (seg_new != seg_old)
        {
            ovl->path.bepos -= trace[j+1];

            j -= 2;
            seg_old--;
        }

        ovl->path.tlen = j + 2;

        ovl->path.abpos = MAX(trim_a_left, ovl->path.abpos);
        ovl->path.aepos = MIN(trim_a_right, ovl->path.aepos);

#ifdef DEBUG_TRIM
        printf(ANSI_COLOR_BLUE);
#endif
    }

#ifdef DEBUG_TRIM
    printf("%35s %5d..%5d %5d..%5d\n", "",
            ovl->path.abpos, ovl->path.aepos,
            ovl->path.bbpos, ovl->path.bepos);
    printf(ANSI_COLOR_RESET);
#endif

    trim->nTrimmedOvls++;

    bbt = MAX(trim_b_left, ovl->path.bbpos);
    bet = MIN(trim_b_right, ovl->path.bepos);

    if (bbt < bet && (bbt != ovl->path.bbpos || bet != ovl->path.bepos))
    {
        int abpos, aepos, bbpos, bepos, j;

        // align segment on the left

        bbpos = ovl->path.bbpos;

        if (bbpos < bbt)
        {
            // printf("left trim\n");

            abpos = ovl->path.abpos;
            aepos = (abpos / trim->twidth + 1) * trim->twidth;
            bepos = bbpos + trace[1];

            // print_trace(ovl);

            for (j = 2; j < ovl->path.tlen; j += 2)
            {
                if (bbpos <= bbt && bepos > bbt)
                {
                    // printf("L %d %5d..%5d %5d..%5d\n", j, abpos, aepos, bbpos, bepos);
                    break;
                }

                abpos  = aepos;
                aepos += trim->twidth;

                bbpos  = bepos;
                bepos += trace[j+1];
            }

            if (bbpos != bbt)
            {
                align(trim, ovl, abpos, aepos, bbpos, bepos);

                int stop_a, diffs_left, diffs_right;

                eval_trace(trim, bbt, &stop_a, &diffs_left, &diffs_right);

                trace[j-1] = bepos - bbt;
                trace[j-2] = diffs_right;

                // rare case, see e.coli dataset 729 -> 15957 for an example

                if (aepos == stop_a) stop_a--;

                trim->nTrimmedBases += (stop_a - ovl->path.abpos) + (bbt - ovl->path.bbpos);

                ovl->path.abpos = stop_a;
                ovl->path.bbpos = bbt;

                // printf("STOP @ %5d %5d DIFFS %2d %2d\n", stop_a, bbt, diffs_left, diffs_right);
            }
            else
            {
            	trim->nTrimmedBases += (abpos - ovl->path.abpos) + (bbpos - ovl->path.bbpos);

                ovl->path.abpos = abpos;
                ovl->path.bbpos = bbpos;
            }

            trace += j - 2;
            ovl->path.trace += sizeof(ovl_trace) * (j - 2);
            ovl->path.tlen -= j - 2;

            // print_trace(ovl);
        }

        // align segment on the right

        bepos = ovl->path.bepos;

        if (bepos > bet)
        {
            aepos = ovl->path.aepos;
            abpos = (aepos % trim->twidth) ? aepos - (aepos % trim->twidth) : aepos - trim->twidth;
            bbpos = bepos - trace[ ovl->path.tlen - 1 ];

            // print_trace(ovl);

            for (j = ovl->path.tlen - 4; j >= 0; j -= 2)
            {
                if (bbpos < bet && bepos >= bet)
                {
                    // printf("R %d %5d..%5d %5d..%5d\n", ovl->path.tlen - j-2, abpos, aepos, bbpos, bepos);
                    break;
                }

                aepos  = abpos;
                abpos -= trim->twidth;

                bepos  = bbpos;
                bbpos -= trace[j+1];
            }

            if (bepos != bet)
            {
                align(trim, ovl, abpos, aepos, bbpos, bepos);

                int stop_a, diffs_left, diffs_right;

                eval_trace(trim, bet, &stop_a, &diffs_left, &diffs_right);

                trim->nTrimmedBases += (ovl->path.aepos - stop_a) + (ovl->path.bepos - bet);

                ovl->path.aepos = stop_a;
                ovl->path.bepos = bet;

                trace[j+2] = diffs_left;
                trace[j+3] = bet - bbpos;

                // printf("STOP @ %5d %5d DIFFS %2d %2d\n", stop_a, bet, diffs_left, diffs_right);
            }
            else
            {
            	trim->nTrimmedBases += (ovl->path.aepos - aepos) + (ovl->path.bepos - bepos);

                ovl->path.aepos = aepos;
                ovl->path.bepos = bepos;
            }

            ovl->path.tlen = j + 4;

            // print_trace(ovl);
        }

#ifdef DEBUG_TRIM
        printf(ANSI_COLOR_BLUE);
#endif
    }

#ifdef DEBUG_TRIM
    printf("%35s %5d..%5d %5d..%5d nTOvl(%lld) nTBas(%lld)\n", "",
            ovl->path.abpos, ovl->path.aepos,
			ovl->path.bbpos, ovl->path.bepos,
			trim->nTrimmedOvls, trim->nTrimmedBases
			);
    printf(ANSI_COLOR_RESET);
#endif

    // this can happen if we trim the alignment to a very short length
    // covering mostly noisy regions in either a or b
    if (ovl->path.abpos >= ovl->path.aepos || ovl->path.bepos < trim_b_left || ovl->path.bbpos > trim_b_right)
    {
        ovl->flags |= OVL_DISCARD | OVL_TRIM;
    }
    else
    {
        ovl->path.diffs = 0;
        int j;
        for (j = 0; j < ovl->path.tlen; j += 2)
        {
            ovl->path.diffs += trace[j];
        }
    }

#ifdef DEBUG_VALIDATE
    {
        int cura = ovl->path.abpos;
        int curb = ovl->path.bbpos;
        int j;
        for (j = 0; j < ovl->path.tlen; j += 2)
        {
            cura = (cura/trim->twidth + 1) * trim->twidth;
            curb += trace[j+1];
        }

        assert(curb == ovl->path.bepos);
    }
#endif
}

TRIM* trim_init(HITS_DB* db, ovl_header_twidth twidth, HITS_TRACK *track, Read_Loader *rl)
{
    TRIM* trim = malloc( sizeof(TRIM) );

    trim->db = db;
    trim->twidth = twidth;
    trim->track = track;

    trim->align_work = New_Work_Data();

    trim->align.path = &(trim->path);
    trim->align.aseq = New_Read_Buffer(db);
    trim->align.bseq = New_Read_Buffer(db);

    trim->nTrimmedBases = 0;
    trim->nTrimmedOvls  = 0;
    trim->nOvls         = 0;
    trim->nOvlBases     = 0;

    trim->rl = rl;

    return trim;
}

void trim_close(TRIM* trim)
{
//    track_close(trim->track);

    Free_Work_Data(trim->align_work);
    free(trim->align.aseq - 1);
    free(trim->align.bseq - 1);

    bzero(trim, sizeof(TRIM));
    free(trim);
}
