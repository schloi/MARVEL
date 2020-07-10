
#include <stdlib.h>
#include <string.h>

#include "align.h"
#include "db/DB.h"
#include "filter.h"
#include "ovlbuffer.h"

extern int Binshift;
extern int Hitmin;
extern int Kmer;
extern int MG_self;
extern HITS_DB* MR_ablock;
extern HITS_DB* MR_bblock;
extern SeedPair* MR_hits;
extern Align_Spec* MR_spec;
extern int MR_tspace;

void* report_thread( void* arg )
{
    Report_Arg* data        = (Report_Arg*)arg;
    SeedPair* hits          = MR_hits;
    Double* hitd            = (Double*)MR_hits;
    DAZZ_READ* bread        = MR_bblock->reads;
    DAZZ_READ* aread        = MR_ablock->reads;
    char* aseq              = (char*)( MR_ablock->bases );
    char* bseq              = (char*)( MR_bblock->bases );
    int* score              = data->score;
    int* scorp              = data->score + 1;
    int* scorm              = data->score - 1;
    int* lastp              = data->lastp;
    int* lasta              = data->lasta;
    Overlap_IO_Buffer* obuf = data->iobuf;

    int afirst = MR_ablock->ufirst; // tfirst;
    int bfirst = MR_bblock->ufirst; // tfirst;

#ifndef ENABLE_OVL_IO_BUFFER
    FILE* ofile1 = data->ofile1;
    FILE* ofile2 = data->ofile2;
#endif

    Work_Data* work = data->work;
    int maxdiag     = ( MR_ablock->maxlen >> Binshift );
    int mindiag     = ( -MR_bblock->maxlen >> Binshift );
    int areads      = MR_ablock->nreads;
#ifdef PROFILE
    int* profyes = data->profyes;
    int* profno  = data->profno;
    int maxhit, isyes;
#endif

    Overlap _ovlb, *ovlb     = &_ovlb;
    Overlap _ovla, *ovla     = &_ovla;
    Alignment _align, *align = &_align;
    Path* apath = &( ovla->path );
    Path* bpath;
    char* bcomp;

    int AOmax, BOmax;
    int novla, novlb;
    Path *amatch, *bmatch;

    Trace_Buffer _tbuf, *tbuf = &_tbuf;
    int small, tbytes;

    Double* hitc;
    int minhit;
    uint64 cpair;
    uint64 npair = 0;
    int64 nidx, eidx;

    int64 nfilt = 0;
    int64 nlas  = 0;
    int64 ahits = 0;
    int64 bhits = 0;

    //  In ovl and align roles of A and B are reversed, as the B sequence must be the
    //    complemented sequence !!

    align->path = apath;
    bcomp       = New_Read_Buffer( MR_bblock );

    if ( MR_tspace <= TRACE_XOVR )
    {
        small  = 1;
        tbytes = sizeof( uint8 );
    }
    else
    {
        small  = 0;
        tbytes = sizeof( uint16 );
    }

    AOmax = BOmax = MATCH_CHUNK;
    amatch        = Malloc( sizeof( Path ) * AOmax, "Allocating match vector" );
    bmatch        = Malloc( sizeof( Path ) * BOmax, "Allocating match vector" );

    tbuf->max   = 2 * TRACE_CHUNK;
    tbuf->trace = Malloc( sizeof( short ) * tbuf->max, "Allocating trace vector" );

    if ( amatch == NULL || bmatch == NULL || tbuf->trace == NULL )
        Clean_Exit( 1 );

#ifndef ENABLE_OVL_IO_BUFFER
    fwrite( &ahits, sizeof( int64 ), 1, ofile1 );
    fwrite( &MR_tspace, sizeof( int ), 1, ofile1 );
    if ( MR_two )
    {
        fwrite( &bhits, sizeof( int64 ), 1, ofile2 );
        fwrite( &MR_tspace, sizeof( int ), 1, ofile2 );
    }
#endif

#ifdef PROFILE
    {
        int i;
        for ( i = 0; i <= MAXHIT; i++ )
            profyes[ i ] = profno[ i ] = 0;
    }
#endif

    minhit = ( Hitmin - 1 ) / Kmer + 1;
    hitc   = hitd + ( minhit - 1 );
    eidx   = data->end - minhit;
    nidx   = data->beg;
    for ( cpair = hitd[ nidx ].p1; nidx <= eidx; cpair = npair )
        if ( hitc[ nidx ].p1 != cpair )
        {
            nidx += 1;
            while ( ( npair = hitd[ nidx ].p1 ) == cpair )
                nidx += 1;
        }
        else
        {
            int ar, br, bc;
            int alen, blen;
            int doA, doB;
            int setaln, amark, amark2;
            int apos, bpos, diag;
            int64 lidx, sidx;
            int64 f, h2;

            ar = hits[ nidx ].aread;
            br = hits[ nidx ].bread;
            if ( ar >= areads )
            {
                bc = 1;
                ar -= areads;
            }
            else
                bc = 0;
            alen = aread[ ar ].rlen;
            blen = bread[ br ].rlen;
            if ( alen < HGAP_MIN && blen < HGAP_MIN )
            {
                nidx += 1;
                while ( ( npair = hitd[ nidx ].p1 ) == cpair )
                    nidx += 1;
                continue;
            }
#ifdef PROFILE
            maxhit = 0;
            isyes  = 0;
#endif

#ifdef TEST_GATHER
            printf( "%5d vs %5d%c : %5d x %5d\n", ar + afirst, br + bfirst, bc ? 'c' : 'n', alen, blen );
            fflush( stdout );
#endif
            setaln = 1;
            doA = doB = 0;
            amark2    = 0;
            novla = novlb = 0;
            tbuf->top     = 0;
            for ( sidx = nidx; hitd[ nidx ].p1 == cpair; nidx = h2 )
            {
                amark  = amark2 + PANEL_SIZE;
                amark2 = amark - PANEL_OVERLAP;

                h2 = lidx = nidx;
                do
                {
                    apos  = hits[ nidx ].apos;
                    npair = hitd[ ++nidx ].p1;
                    if ( apos <= amark2 )
                        h2 = nidx;
                } while ( npair == cpair && apos <= amark );

                if ( nidx - lidx < minhit )
                    continue;

                for ( f = lidx; f < nidx; f++ )
                {
                    apos = hits[ f ].apos;
                    diag = hits[ f ].diag >> Binshift;
                    if ( apos - lastp[ diag ] >= Kmer )
                        score[ diag ] += Kmer;
                    else
                        score[ diag ] += apos - lastp[ diag ];
                    lastp[ diag ] = apos;
                }

#ifdef TEST_GATHER
                printf( "  %6lld upto %6d", nidx - lidx, amark );
                fflush( stdout );
#endif

                for ( f = lidx; f < nidx; f++ )
                {
                    apos = hits[ f ].apos;
                    diag = hits[ f ].diag;
                    bpos = apos - diag;
                    diag = diag >> Binshift;
#ifdef PROFILE
                    if ( score[ diag ] + scorp[ diag ] > maxhit )
                        maxhit = score[ diag ] + scorp[ diag ];
                    if ( score[ diag ] + scorm[ diag ] > maxhit )
                        maxhit = score[ diag ] + scorm[ diag ];
#endif
                    if ( apos > lasta[ diag ] &&
                         ( score[ diag ] + scorp[ diag ] >= Hitmin || score[ diag ] + scorm[ diag ] >= Hitmin ) )
                    {
                        if ( setaln )
                        {
                            setaln      = 0;
                            align->aseq = aseq + aread[ ar ].boff;
                            align->bseq = bseq + bread[ br ].boff;
                            if ( bc )
                            {
                                CopyAndComp( bcomp, align->bseq, blen );
                                align->bseq = bcomp;
                            }
                            align->alen  = alen;
                            align->blen  = blen;
                            align->flags = ovla->flags = ovlb->flags = bc;
                            ovlb->bread = ovla->aread = ar + afirst;
                            ovlb->aread = ovla->bread = br + bfirst;
                            doA                       = ( alen >= HGAP_MIN );
                            doB                       = ( SYMMETRIC && blen >= HGAP_MIN && !( ar == br && MG_self ) );
                        }
#ifdef TEST_GATHER
                        else
                            printf( "\n                    " );

                        if ( scorm[ diag ] > scorp[ diag ] )
                            printf( "  %5d.. x %5d.. %5d (%3d)",
                                    bpos, apos, apos - bpos, score[ diag ] + scorm[ diag ] );
                        else
                            printf( "  %5d.. x %5d.. %5d (%3d)",
                                    bpos, apos, apos - bpos, score[ diag ] + scorp[ diag ] );
                        fflush( stdout );
#endif
                        nfilt += 1;

#ifdef DO_ALIGNMENT
                        bpath = Local_Alignment( align, work, MR_spec, apos - bpos, apos - bpos, apos + bpos, -1, -1 );

                        {
                            int low, hgh, ae;

                            Diagonal_Span( apath, &low, &hgh );
                            if ( diag < low )
                                low = diag;
                            else if ( diag > hgh )
                                hgh = diag;
                            ae = apath->aepos;
                            for ( diag = low; diag <= hgh; diag++ )
                                if ( ae > lasta[ diag ] )
                                    lasta[ diag ] = ae;
#ifdef TEST_GATHER
                            printf( " %d - %d @ %d", low, hgh, apath->aepos );
                            fflush( stdout );
#endif
                        }

                        if ( ( apath->aepos - apath->abpos ) + ( apath->bepos - apath->bbpos ) >= MINOVER )
                        {
                            if ( doA )
                            {
                                if ( novla >= AOmax )
                                {
                                    AOmax  = 1.2 * novla + MATCH_CHUNK;
                                    amatch = Realloc( amatch, sizeof( Path ) * AOmax,
                                                      "Reallocating match vector" );
                                    if ( amatch == NULL )
                                        Clean_Exit( 1 );
                                }
                                if ( tbuf->top + apath->tlen > tbuf->max )
                                {
                                    tbuf->max   = 1.2 * ( tbuf->top + apath->tlen ) + TRACE_CHUNK;
                                    tbuf->trace = Realloc( tbuf->trace, sizeof( short ) * tbuf->max,
                                                           "Reallocating trace vector" );
                                    if ( tbuf->trace == NULL )
                                        Clean_Exit( 1 );
                                }
                                amatch[ novla ]       = *apath;
                                amatch[ novla ].trace = (void*)( tbuf->top );
                                memmove( tbuf->trace + tbuf->top, apath->trace, sizeof( short ) * apath->tlen );
                                novla += 1;
                                tbuf->top += apath->tlen;
                            }
                            if ( doB )
                            {
                                if ( novlb >= BOmax )
                                {
                                    BOmax  = 1.2 * novlb + MATCH_CHUNK;
                                    bmatch = Realloc( bmatch, sizeof( Path ) * BOmax,
                                                      "Reallocating match vector" );
                                    if ( bmatch == NULL )
                                        Clean_Exit( 1 );
                                }
                                if ( tbuf->top + bpath->tlen > tbuf->max )
                                {
                                    tbuf->max   = 1.2 * ( tbuf->top + bpath->tlen ) + TRACE_CHUNK;
                                    tbuf->trace = Realloc( tbuf->trace, sizeof( short ) * tbuf->max,
                                                           "Reallocating trace vector" );
                                    if ( tbuf->trace == NULL )
                                        Clean_Exit( 1 );
                                }
                                bmatch[ novlb ]       = *bpath;
                                bmatch[ novlb ].trace = (void*)( tbuf->top );
                                memmove( tbuf->trace + tbuf->top, bpath->trace, sizeof( short ) * bpath->tlen );
                                novlb += 1;
                                tbuf->top += bpath->tlen;
                            }

#ifdef TEST_GATHER
                            printf( "  [%5d,%5d] x [%5d,%5d] = %4d",
                                    apath->abpos, apath->aepos, apath->bbpos, apath->bepos, apath->diffs );
                            fflush( stdout );
#endif
#ifdef SHOW_OVERLAP
                            printf( "\n\n                    %d(%d) vs %d(%d)\n\n",
                                    ovla->aread, ovla->alen, ovla->bread, ovla->blen );
                            Print_ACartoon( stdout, align, ALIGN_INDENT );
#ifdef SHOW_ALIGNMENT
                            Compute_Trace_ALL( align, work );
                            printf( "\n                      Diff = %d\n", align->path->diffs );
                            Print_Alignment( stdout, align, work,
                                             ALIGN_INDENT, ALIGN_WIDTH, ALIGN_BORDER, 0, 5 );
#endif
#endif // SHOW_OVERLAP
                        }
#ifdef TEST_GATHER
                        else
                            printf( "  No alignment %d",
                                    ( ( apath->aepos - apath->abpos ) + ( apath->bepos - apath->bbpos ) ) / 2 );
                        fflush( stdout );
#endif
#endif // DO_ALIGNMENT
                    }
                }

                for ( f = lidx; f < nidx; f++ )
                {
                    diag          = hits[ f ].diag >> Binshift;
                    score[ diag ] = lastp[ diag ] = 0;
                }
#ifdef TEST_GATHER
                printf( "\n" );
                fflush( stdout );
#endif
            }

            for ( f = sidx; f < nidx; f++ )
            {
                int d;

                diag = hits[ f ].diag >> Binshift;
                for ( d = diag; d <= maxdiag; d++ )
                    if ( lasta[ d ] == 0 )
                        break;
                    else
                        lasta[ d ] = 0;
                for ( d = diag - 1; d >= mindiag; d-- )
                    if ( lasta[ d ] == 0 )
                        break;
                    else
                        lasta[ d ] = 0;
            }

            {
                int i;

#ifdef TEST_CONTAIN
                if ( novla > 1 || novlb > 1 )
                    printf( "\n%5d vs %5d:\n", ar, br );
#endif

                if ( novla > 1 )
                {
                    if ( novlb > 1 )
                        novla = novlb = Handle_Redundancies( amatch, novla, bmatch, align, work, tbuf );
                    else
                        novla = Handle_Redundancies( amatch, novla, NULL, align, work, tbuf );
                }
                else if ( novlb > 1 )
                    novlb = Handle_Redundancies( bmatch, novlb, NULL, align, work, tbuf );

                for ( i = 0; i < novla; i++ )
                {
                    ovla->path       = amatch[ i ];
                    ovla->path.trace = tbuf->trace + ( uint64 )( ovla->path.trace );
                    if ( small )
                        Compress_TraceTo8( ovla, 1 );

#ifdef ENABLE_OVL_IO_BUFFER
                    AddOverlapToBuffer( obuf, ovla, tbytes );
#else
                    if ( Write_Overlap( ofile1, ovla, tbytes ) )
                    {
                        fprintf( stderr, "%s: Cannot write to %s too small?\n", SORT_PATH, Prog_Name );
                        Clean_Exit( 1 );
                    }
#endif
                }
                for ( i = 0; i < novlb; i++ )
                {
                    ovlb->path       = bmatch[ i ];
                    ovlb->path.trace = tbuf->trace + ( uint64 )( ovlb->path.trace );
                    if ( small )
                        Compress_TraceTo8( ovlb, 1 );

#ifdef ENABLE_OVL_IO_BUFFER
                    AddOverlapToBuffer( obuf, ovlb, tbytes );
#else
                    if ( Write_Overlap( ofile2, ovlb, tbytes ) )
                    {
                        fprintf( stderr, "%s: Cannot write to %s, too small?\n", SORT_PATH, Prog_Name );
                        Clean_Exit( 1 );
                    }
#endif
                }
                if ( doA )
                    nlas += novla;
                else
                    nlas += novlb;
                ahits += novla;
                bhits += novlb;
#ifdef PROFILE
                isyes = ( novla + novlb > 0 );
#endif
            }

#ifdef PROFILE
            if ( maxhit > MAXHIT )
                maxhit = MAXHIT;
            if ( isyes )
                profyes[ maxhit ] += 1;
            else
                profno[ maxhit ] += 1;
#endif
        }

    free( tbuf->trace );
    free( bmatch );
    free( amatch );
    free( bcomp - 1 );

    data->nfilt = nfilt;
    data->nlas  = nlas;

#ifndef ENABLE_OVL_IO_BUFFER
    if ( MR_two )
    {
        rewind( ofile2 );
        fwrite( &bhits, sizeof( int64 ), 1, ofile2 );
        fclose( ofile2 );
    }
    else
        ahits += bhits;

    rewind( ofile1 );
    fwrite( &ahits, sizeof( int64 ), 1, ofile1 );
    fclose( ofile1 );
#endif

    return ( NULL );
}
