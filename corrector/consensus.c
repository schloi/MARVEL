
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "consensus.h"
#include "lib/colors.h"

#define CONS_MIN_COV_FRACTION   0.5 // base calling threshold

#undef DEBUG_SHOW_ADD

typedef struct
{
    char* B;                        // sequence that gets aligned to
    profile_entry* A;               // the profile B

    int abpos, aepos;               // regions to be aligned in A
    int bbpos, bepos;               // in B

    int* pPtPoints;                 // pairs of pass-through points <profile, sequence>
    int  nPtPoints;                 // number of pass-through points ( pairs * 2 )

    int diffs_a;                    // number of insertions in A
    int diffs_b;                    // number of insertions in B

    v3_consensus_alignment_ctx* pCtx;   // points to context, working storage
} v3_consensus_alignment;


static int match(char* a, int ia, profile_entry* p, int ip)
{
    profile_entry* pEntry = (p + ip);

    char nMax = pEntry->counts[0];
    char cMax = 0;
    char nCount;

    // loop unrolled

    nCount = pEntry->counts[1];
    if (nCount > nMax ) { nMax = nCount; cMax = 1; }

    nCount = pEntry->counts[2];
    if (nCount > nMax ) { nMax = nCount; cMax = 2; }

    nCount = pEntry->counts[3];
    if (nCount > nMax ) { nMax = nCount; cMax = 3; }

    nCount = pEntry->counts[4];
    if (nCount >= nMax ) { return 0; }

    return (cMax == a[ia]);
}

static int v3_align_onp(v3_consensus_alignment* pAlign,
                        profile_entry* A, int M, char* B, int N)
{
    int**  PVF = pAlign->pCtx->PVF; // wave->PVF;
    int**  PHF = pAlign->pCtx->PHF; // wave->PHF;
    int    D;
    int    del = M - N;

    {
        int*  F0, *F1, *F2;
        int*  HF;
        int   low, hgh;

        if (del >= 0)
        {
            low = 0;
            hgh = del;
        }
        else
        {
            low = del;
            hgh = 0;
        }

        F1 = PVF[-2];
        F0 = PVF[-1];

        for (D = low - 1; D <= hgh + 1; D++)
        {
            F1[D] = F0[D] = -2;
        }

        F0[0] = -1;

        low += 1;
        hgh -= 1;

        for (D = 0; 1; D += 1)
        {
            int   k, i, j;
            int   am, ac, ap;
            profile_entry* a;

            F2 = F1;
            F1 = F0;
            F0 = PVF[D];
            HF = PHF[D];

            if ((D & 0x1) == 0)
            {
                hgh += 1;
                low -= 1;
            }

            F0[hgh + 1] = F0[low - 1] = -2;

#define FS_MOVE(mdir,pdir)          \
  ac = F1[k]+1;                     \
  if (ac < am)                      \
    if (ap < am)                    \
      { HF[k] = mdir;               \
        j = am;                     \
      }                             \
    else                            \
      { HF[k] = pdir;               \
        j = ap;                     \
      }                             \
  else                              \
    if (ap < ac)                    \
      { HF[k] = 0;                  \
        j = ac;                     \
      }                             \
    else                            \
      { HF[k] = pdir;               \
        j = ap;                     \
      }                             \
                                    \
  if (N < i)                        \
    while (j < N && match(B,j,a,j)) \
      j += 1;                       \
  else                              \
    while (j < i && match(B,j,a,j)) \
      j += 1;                       \
  F0[k] = j;

            j = -2;
            a = A + hgh;
            i = M - hgh;

            for (k = hgh; k > del; k--)
            {
                ap = j + 1;
                am = F2[k - 1];
                FS_MOVE(-1, 4)
                a -= 1;
                i += 1;
            }

            j = -2;
            a = A + low;
            i = M - low;

            for (k = low; k < del; k++)
            {
                ap = F2[k + 1] + 1;
                am = j;
                FS_MOVE(2, 1)
                a += 1;
                i -= 1;
            }

            ap = F0[del + 1] + 1;
            am = j;
            FS_MOVE(2, 4)

            if (F0[del] >= N)
            {
                break;
            }
        }
    }

    {
        int   k, h, m, e, c;
        profile_entry* a;
        int   ap = (pAlign->pCtx->Aabs - A) - 1; // (wave->Aabs - A) - 1;
        int   bp = (B - pAlign->pCtx->Babs) + 1; // (B - wave->Babs) + 1;

        PHF[0][0] = 3;

        c = N;
        k = del;
        e = PHF[D][k];
        PHF[D][k] = 3;

        while (e != 3)
        {
            h = k + e;

            if (e > 1)
            {
                h -= 3;
            }
            else if (e == 0)
            {
                D -= 1;
            }
            else
            {
                D -= 2;
            }

            if (h < k)       // => e = -1 or 2
            {
                a = A + k;

                if (k < 0)
                {
                    m = -k;
                }
                else
                {
                    m = 0;
                }

                if (PVF[D][h] <= c)
                {
                    c = PVF[D][h] - 1;
                }

                while (c >= m && match(B,c,a,c)) // a[c] == B[c])
                {
                    c -= 1;
                }

                if (e < 1)  //  => edge is 2, others are 1, and 0
                {
                    if (c <= PVF[D + 2][k + 1])
                    {
                        e = 4;
                        h = k + 1;
                        D = D + 2;
                    }
                    else if (c == PVF[D + 1][k])
                    {
                        e = 0;
                        h = k;
                        D = D + 1;
                    }
                    else
                    {
                        PVF[D][h] = c + 1;
                    }
                }
                else      //   => edge is 0, others are 1, and 2 (if k != del), 0 (otherwise)
                {
                    if (k == del)
                    {
                        m = D;
                    }
                    else
                    {
                        m = D - 2;
                    }

                    if (c <= PVF[m][k + 1])
                    {
                        if (k == del)
                        {
                            e = 4;
                        }
                        else
                        {
                            e = 1;
                        }

                        h = k + 1;
                        D = m;
                    }
                    else if (c == PVF[D - 1][k])
                    {
                        e = 0;
                        h = k;
                        D = D - 1;
                    }
                    else
                    {
                        PVF[D][h] = c + 1;
                    }
                }
            }

            m = PHF[D][h];
            PHF[D][h] = e;
            e = m;

            k = h;
        }

        k = D = 0;
        e = PHF[D][k];

        while (e != 3)
        {
            h = k - e;
            c = PVF[D][k];

            if (e > 1)
            {
                h += 3;
            }
            else if (e == 0)
            {
                D += 1;
            }
            else
            {
                D += 2;
            }

            if (h > k)
            {
                *(pAlign->pCtx->Stop) = bp + c;
                (pAlign->pCtx->Stop)++;

                pAlign->diffs_b++;

                // *wave->Stop++ = bp + c;
            }
            else if (h < k)
            {
                *(pAlign->pCtx->Stop) = ap - (c + k);
                (pAlign->pCtx->Stop)++;

                pAlign->diffs_a++;

                // *wave->Stop++ = ap - (c + k);
            }

            k = h;
            e = PHF[D][h];
        }

    }

    return (D + abs(del));
}

static int v3_align(v3_consensus_alignment* align)
{
    v3_consensus_alignment_ctx* pCtx = align->pCtx;
    char* bseq;
    profile_entry* aseq;
    int*  points, tlen;
    int   ab, bb;
    int   ae, be;
    int   diffs;

    aseq   = align->A;
    bseq   = align->B;

    tlen   = align->nPtPoints;
    points = align->pPtPoints;

    {
        int d, s;
        int M, N;
        int dmax, nmax, mmax;
        int** PVF, **PHF;

        M = align->aepos - align->abpos;
        N = align->bepos - align->bbpos;

        s = M>N ? M : N;

        if (s > pCtx->ntrace)
        {
            pCtx->ntrace = 1.2 * pCtx->ntrace + s;
            pCtx->trace = (int*)realloc(pCtx->trace, sizeof(int)*pCtx->ntrace);
        }

        mmax = 0;
        nmax = 0;
        dmax = 0;
        ab = align->abpos;
        bb = align->bbpos;

        for (d = 0; d < tlen; d += 2)
        {
            ae = points[d];
            be = points[d + 1];

            M  = ae - ab;
            N  = be - bb;

            if (M < N)
            {
                diffs = M;
            }
            else
            {
                diffs = N;
            }

            if (diffs > dmax)
            {
                dmax = diffs;
            }

            if (M > mmax)
            {
                mmax = M;
            }

            if (N > nmax)
            {
                nmax = N;
            }

            ab = ae;
            bb = be;
        }

        ae = align->aepos;
        be = align->bepos;
        M  = ae - ab;
        N  = be - bb;

        if (M < N)
        {
            diffs = M;
        }
        else
        {
            diffs = N;
        }

        if (diffs > dmax)
        {
            dmax = diffs;
        }

        if (M > mmax)
        {
            mmax = M;
        }

        if (N > nmax)
        {
            nmax = N;
        }

        s = (dmax + 3) * 2 * ((mmax + nmax + 3) * sizeof(int) + sizeof(int*));

        if (s > pCtx->vecmax)
        {
            if (s > 256*1024*1024)
            {
                printf("v3_align> skipping alignment. excessive amount of memory needed.\n");
                printf("dmax = %d mmax = %d nmax = %d s = %d\n", dmax, mmax, nmax, s);

                return 0;
            }

            pCtx->vecmax = s * 1.2 + 10000;

            pCtx->vector = (int*)realloc(pCtx->vector, pCtx->vecmax);
        }

        pCtx->PVF = PVF = ((int**) (pCtx->vector)) + 2;
        pCtx->PHF = PHF = PVF + (dmax + 3);

        // wave.PVF = PVF = ((int**) (work->vec.data())) + 2;
        // wave.PHF = PHF = PVF + (dmax + 3);

        s = mmax + nmax + 3;
        PVF[-2] = ((int*) (PHF + (dmax + 1))) + (nmax + 1);

        for (d = -1; d <= dmax; d++)
        {
            PVF[d] = PVF[d - 1] + s;
        }

        PHF[-2] = PVF[dmax] + s;

        for (d = -1; d <= dmax; d++)
        {
            PHF[d] = PHF[d - 1] + s;
        }
    }

    pCtx->Stop = pCtx->trace;
    pCtx->Aabs = aseq;
    pCtx->Babs = bseq;

    {
        int i;

        diffs = 0;
        ab = align->abpos;
        bb = align->bbpos;

        for (i = 0; i < tlen; i += 2)
        {
            be = points[i + 1];
            ae = points[i];

            diffs += v3_align_onp(align, aseq + ab, ae - ab, bseq + bb, be - bb);

            ab = ae;
            bb = be;
        }

        ae = align->aepos;
        be = align->bepos;

        diffs += v3_align_onp(align, aseq + ab, ae - ab, bseq + bb, be - bb);
    }

    pCtx->ntrace = pCtx->Stop - pCtx->trace;

    // path->diffs = diffs;

    return 1;
}

static char decode_base(unsigned char c)
{
    static const char* pcTable = "ACGT-";
    return pcTable[c];
}

consensus* consensus_init()
{
    consensus* c = (consensus*)malloc(sizeof(consensus));

    c->profile = NULL;
    c->maxprof = 0;
    c->added = 0;

    c->seq = NULL;
    c->nseq = 0;

    c->aln_ctx = (v3_consensus_alignment_ctx*)malloc(sizeof(v3_consensus_alignment_ctx));
    c->aln_ctx->ntrace = 0;
    c->aln_ctx->trace = NULL;
    c->aln_ctx->vecmax = 0;
    c->aln_ctx->vector = NULL;

    return c;
}

void consensus_free(consensus* c)
{
    free(c->profile);

    free(c->aln_ctx->trace);
    free(c->aln_ctx->vector);

    free(c->aln_ctx);

    free(c);
}

void consensus_print_profile(consensus* c, FILE* fileOut, int colorize)
{
    profile_entry* pEntry;

    // counts

    int i;

    if (colorize) printf(ANSI_COLOR_BLUE);

    printf("  ");
    for (i = 0; i < c->curprof; i++)
    {
        pEntry = c->profile + i;
        fprintf(fileOut, "%3d", i%100);
    }

    if (colorize) printf(ANSI_COLOR_RESET);

    fprintf(fileOut, "\n");

    int nBase;
    for (nBase = 0; nBase < 5; nBase++)
    {
        fprintf(fileOut, "%c ", decode_base(nBase));

        for (i = 0; i < c->curprof; i++)
        {
            pEntry = c->profile + i;
            fprintf(fileOut, "%3d", pEntry->counts[nBase]);
        }
        fprintf(fileOut, "\n");
    }

    // consensus

    if (colorize) printf(ANSI_COLOR_RED);

    char* pcCons = consensus_sequence(c, 1);

    printf("# ");
    for (i = 0; i < c->curprof; i++)
    {
        fprintf(fileOut, "%3c", pcCons[i]);
    }

    if (colorize) printf(ANSI_COLOR_RESET);

    fprintf(fileOut, "\n");
}

void consensus_reset(consensus* c)
{
    c->curprof = 0;
    c->added = 0;
}

static void consensus_add_first(consensus* c, char* seq, int len)
{
    c->alen = len;

    if (c->maxprof < len)
    {
        c->maxprof = 1.2 * len + 100;
        c->profile = realloc(c->profile, sizeof(profile_entry)*c->maxprof);

    }

    c->curprof = len;
    bzero(c->profile, sizeof(profile_entry)*c->maxprof);

    int i;
    for (i = 0; i < len; i++)
    {
        c->profile[i].counts[ (unsigned char)seq[i] ] = 1;
    }
}

void consensus_add(consensus* cns, char* seq, int sb, int se) // , int pb, int pe)
{
    int i;

#ifdef DEBUG_SHOW_ADD
    printf("add %5d %5d ", sb, se);
    for (i = sb; i < se; i++)
    {
        printf("%c", decode_base(seq[i]));
    }
    printf("\n");
#endif

    if (cns->curprof == 0)
    {
        consensus_add_first(cns, seq+sb, se-sb);
        cns->added++;

        // consensus_print_profile(cns, stdout);

        return;
    }

    v3_consensus_alignment aln;
    aln.pCtx = cns->aln_ctx;

    aln.diffs_a = 0;
    aln.diffs_b = 0;

    aln.B = seq;
    aln.bbpos = sb;
    aln.bepos = se;

    aln.A = cns->profile;

    aln.abpos = 0;
    aln.aepos = cns->curprof;

    assert( aln.bbpos >= 0 );
    assert( aln.bbpos < aln.bepos );

    // prepare pass through points
    aln.nPtPoints = 0;
    aln.pPtPoints = NULL;

    if (!v3_align(&aln))
    {
        return ;
    }

    int c, p, b, n;

    // enlarge profile if necessary
    if (cns->curprof + aln.diffs_a > cns->maxprof)
    {
        cns->maxprof = cns->maxprof * 2 + aln.diffs_a;

        cns->profile = (profile_entry*)realloc(cns->profile, sizeof(profile_entry)*cns->maxprof);
        bzero(cns->profile + cns->curprof, sizeof(profile_entry)*aln.diffs_a);
    }

    // apply trace to profile and sequence in reverse order
    // thereby allowing us to perform the update in place


    // profile overhang right
    n = cns->curprof - 1 + aln.diffs_a;
    p = cns->curprof - 1;

    while (p > aln.aepos-1)
    {
        cns->profile[n] = cns->profile[p];
        p--;
        n--;
    }

    b = aln.bepos - 1;

    for (i = aln.pCtx->ntrace - 1; i >= 0; i--)
    {
        c = aln.pCtx->trace[i];

        if (c > 0)      // dash before B[-c]
        {
            c--;

            while (c <= b)
            {
                cns->profile[n] = cns->profile[p];
                cns->profile[n].counts[ (unsigned char)aln.B[b] ]++;

                n--;

                b--;
                p--;
            }

#ifdef CONSENSUS_KEEP_SEQUENCES
            apply_gap(c + m_vecPositionTracking[nProfileStart] - nSeqStart, m_nSeqsAdded, m_nSeqsAdded+1);
#endif

            cns->profile[n] = cns->profile[p];
            cns->profile[n].counts[4]++;
            n--;

            p--;
        }
        else        // dash before P[c]
        {
            c = -c;
            c--;

            while (c <= p)
            {
                cns->profile[n] = cns->profile[p];
                cns->profile[n].counts[ (unsigned char)aln.B[b] ]++;

                n--;

                b--;
                p--;
            }

#ifdef CONSENSUS_KEEP_SEQUENCES
            apply_gap(c, 0, m_nSeqsAdded);
#endif

            memset(cns->profile + n, 0, sizeof(profile_entry));
            cns->profile[n].counts[ (unsigned char)aln.B[b] ]++;
            cns->profile[n].counts[4] = cns->added; // ++;

            n--;

            b--;
        }
    }

    // aligned leftovers
    while (b >= aln.bbpos)
    {
        cns->profile[n].counts[ (unsigned char)aln.B[b] ]++;

        n--;

        b--;
        p--;
    }

    cns->curprof += aln.diffs_a;

    cns->added++;

    // consensus_print_profile(cns, stdout, 1);

}

int consensus_added(consensus* c)
{
    return c->added;
}

char* consensus_sequence(consensus* c, int dashes)
{
    profile_entry* pEntry;

    if (c->nseq <= c->curprof)
    {
        c->nseq = c->curprof + 1;
        c->seq = (char*)realloc(c->seq, c->nseq);
    }

    int curseq = 0;
    unsigned char nMax, cMax;
    unsigned short nCov;
    // unsigned short nCovGaps;

    int i, b;

    for (i = 0; i < c->curprof; i++)
    {
        pEntry = c->profile + i;
        nMax = cMax = 0;

        // find out whether ACTG or - received the most "votes"
        for (b = 0; b < 5; b++)
        {
            if (pEntry->counts[b] > nMax)
            {
                nMax = pEntry->counts[b];
                cMax = b;
            }
        }

        nCov = pEntry->counts[0] + pEntry->counts[1] + pEntry->counts[2] + pEntry->counts[3];
        // nCovGaps = nCov + pEntry->counts[4];

        if (c->added <= 2)
        {
            if ( c->added != nCov )
            {
                cMax = 4;
            }
        }
        else if (nCov < c->added * CONS_MIN_COV_FRACTION)
        {
            cMax = 4;
        }

        // if it is a dash, but we don't have to add "-" to the consensus sequence
        if ( cMax == 4 && !dashes )
        {
            continue;
        }

        c->seq[ curseq++ ] = decode_base(cMax);
    }

    c->seq[ curseq ] = '\0';

    return c->seq;
}



