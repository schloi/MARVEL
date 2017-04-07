/*******************************************************************************************
 *
 *  Consensus based read correction
 *
 *  Date   : revisited July 2016
 *
 *  Author : Marvel Team
 *
 *******************************************************************************************/

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

#include "consensus.h"
#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include "dalign/align.h"
#include "db/DB.h"

// settings

#define MIN_TWIDTH 20

#define MAX_COVERAGE 20
#define MAX_TILES 100

#undef ADJUST_OFFSETS      // use mid-points to adjust pass through points
#undef FIX_BOUNDARY_ERRORS // perform multi-tile alignment around segment boundary
#define USE_A_TILES        // add the A tile to the tile-pile

#undef WRITE_STATISTICS

#define TRACK_POSITIONS

#ifdef FIX_BOUNDARY_ERRORS
#define CE_CENTER_DISTANCE 0.4 // consensus error fixing, max fraction of twidth from tile boundary
#endif

// defaults

#define DEF_ARG_X 0 // min length in order for corrected sequences to written to file
#define DEF_ARG_Q TRACK_Q

// development only

#define VERBOSE

#undef DEBUG
#undef DEBUG_FIX_CONSENSUS
#undef DEBUG_MULTI
#undef DEBUG_OFFSETS

#ifdef DEBUG_MULTI
#include "msa/msa.h"
#endif

// parameters for each correction thread

typedef struct
{
    int thread;     // thread number
    int twidth;     // spacing between the alignment trace points
    off_t start;    // start offset in the overlap file
    off_t end;      // end offset in the overlap
    FILE* fileOvls; // overlaps
    FILE* fileOut;  // output file

#ifdef WRITE_STATISTICS
    FILE* fileStats; // statistics file
#endif

    HITS_DB db;            // database
    HITS_TRACK* qtrack;    // quality track
    char* fastaHeader;

    int minRLen;
} corrector_arg;

// relative to the A read the overlaps are decomposed into so called tiles.
// where each tile represents the interval between two alignment trace points

typedef struct
{
    int a;
    int b;            // read id the tile comes from
    int read;         // index of the sequence
    int bbpos, bepos; // b relative interval of the tile
    int qv;           // tile's quality value

    int apos; //  0 ... if the tile covers twidth bases in A
              // <0 ... position in a * -1, if the tile ends before a trace point in A
              // >0 ... position in a, if the tile starts after a trace point in A
} tile_overlap;

// each thread maintains its state using a corrector_context struct

typedef struct
{
    consensus* cons; // consensus state

#ifdef DEBUG_MULTI
    msa* malign;       // multi-align state
    int malign_indent; // multi-align indentation (display only)
#endif

    int thread; // thread #

    int twidth; // spacing of trace points

    int ntoff;
    int* toff;

    int ntovl;
    tile_overlap* tovl;

    int maxreads;
    int nreads;
    char** reads;

    FILE* fileOut;
    HITS_DB* db;

    char* seqcons;
    int maxcons;
    int curcons;

    int* tiles;
    int curtiles;

    int ncorrected;

    track_anno* qtrack_offset; // quality track offsets
    track_data* qtrack_data;   // quality track data
                               // read a -> qtrack_data[ qtrack_offset[a] .. qtrack_offset[a+1]-1 ]

    int* mtc_data;
    int mtc_dmax;
    int** mtc_dsort;

    int stats_tiles_single;
    int stats_tiles_multi;

    Work_Data* align_work_data; // alignment work data

    // base pair position tracking
    int* track;

    // single/multi-tile alignment and boundary artefact elimination
    int ce_smax;
    char* ce_seq_singles;

    int ce_first_tile;

    int* ce_tiles;
    int ce_tcur;

    char* fastaHeader; // keeps pointer to base_out

    int minRLen;

} corrector_context;

// needed for getopt

extern char* optarg;
extern int optind, opterr, optopt;

static int round_down( int n, int f )
{
    return n - ( n % f );
}

static int has_valid_pt_points( Overlap* ovl )
{
    int bb, be, t;
    unsigned short* trace = ovl->path.trace;

    be = ovl->path.bbpos + trace[ 1 ];

    for ( t = 2; t < ovl->path.tlen; t += 2 )
    {
        bb = be;

        if ( t == ovl->path.tlen - 1 )
        {
            be = ovl->path.bepos + 1;
        }
        else
        {
            be += trace[ t + 1 ];
        }

        if ( bb > be ||
             bb < ovl->path.bbpos || bb > ovl->path.bepos ||
             be < ovl->path.bbpos || be > ovl->path.bepos + 1 )
        {
            return 0;
        }
    }

    return 1;
}

static off_t* partition_overlaps( FILE* fileOvls, int parts )
{
    Overlap ovl;
    ovl_header_novl novl;
    ovl_header_twidth twidth;

    off_t* offsets = (off_t*)malloc( sizeof( off_t ) * ( parts + 1 ) );
    bzero( offsets, sizeof( off_t ) * parts );

    ovl_header_read( fileOvls, &novl, &twidth );

    if ( parts == 1 )
    {
        offsets[ 0 ] = ftello( fileOvls );
    }
    else
    {
        int64 ovls_chunk = novl / parts;

        off_t offprev = ftello( fileOvls );
        int aprev, chunk;
        aprev = chunk = 0;
        int64 novl    = 0;

        while ( !Read_Overlap( fileOvls, &ovl ) )
        {
            if ( ovl.aread != aprev )
            {
                offprev = ftello( fileOvls ) - OVERLAP_IO_SIZE;
                aprev   = ovl.aread;
            }

            fseek( fileOvls, TBYTES( twidth ) * ovl.path.tlen, SEEK_CUR );

            if ( ( novl % ovls_chunk ) == 0 )
            {
                offsets[ chunk++ ] = offprev;
            }

            novl++;
        }
    }

    struct stat info;
    fstat( fileno( fileOvls ), &info );
    offsets[ parts ] = info.st_size;

    return offsets;
}

static int cmp_tovl_qv( const void* a, const void* b )
{
    tile_overlap* tovl1 = (tile_overlap*)a;
    tile_overlap* tovl2 = (tile_overlap*)b;

    return tovl1->qv - tovl2->qv;
}

#ifdef DEBUG

/*
static void print_read(char* seq, int b, int e)
{
    static char nucl[] = {'A', 'C', 'G', 'T', '*'};

    while (b != e)
    {
        putc(nucl[ (int)seq[b] ], stdout);
        b++;
    }

    putc('\n', stdout);
}
*/

static void print_tiles( corrector_context* cctx, int tile )
{
    int j;

    printf( "TILE %3d\n", tile );

    for ( j = cctx->toff[ tile ]; j < cctx->toff[ tile + 1 ]; j++ )
    {
        printf( "(" ANSI_COLOR_BLUE "%8d %8d" ANSI_COLOR_RESET " %5d..%5d %3d %3d %6d) ",
                cctx->tovl[ j ].a,
                cctx->tovl[ j ].b,
                cctx->tovl[ j ].bbpos, cctx->tovl[ j ].bepos,
                cctx->tovl[ j ].bepos - cctx->tovl[ j ].bbpos,
                cctx->tovl[ j ].qv, cctx->tovl[ j ].apos );

        if ( j != cctx->toff[ tile + 1 ] - 1 && ( j - cctx->toff[ tile ] + 1 ) % 5 == 0 )
            printf( "\n" );
    }

    printf( "\n" );
}

#endif

static char* single_tile_consensus( corrector_context* cctx, int t, int* tiles_used )
{
    cctx->stats_tiles_single++;

    consensus_reset( cctx->cons );

#ifdef DEBUG_MULTI
    msa_reset( cctx->malign );
#endif

    int j;
    for ( j = cctx->toff[ t ]; j < cctx->toff[ t + 1 ]; j++ )
    {
        int bb = cctx->tovl[ j ].bbpos;
        int be = cctx->tovl[ j ].bepos;

        if ( bb > be )
        {
            printf( "bb > be %d %d ... a %d b %d\n", bb, be, cctx->tovl[ j ].a, cctx->tovl[ j ].b );
            fflush( stdout );
        }

        assert( bb <= be );

        if ( cctx->tovl[ j ].apos == 0 && ( be - bb ) >= MIN_TWIDTH )
        {
            consensus_add( cctx->cons, cctx->reads[ cctx->tovl[ j ].read ], bb, be );

#ifdef DEBUG_MULTI
            msa_add( cctx->malign, cctx->reads[ cctx->tovl[ j ].read ],
                     -1, -1, // 0, cctx->malign->alen,
                     bb, be, NULL, 0 );
#endif

            if ( cctx->cons->added >= MAX_COVERAGE )
            {
                break;
            }
        }
    }

#ifdef DEBUG_MULTI
    msa_print( cctx->malign, stdout );
    cctx->malign_indent += cctx->malign->added + 1;
    printf( "      %s\n", consensus_sequence( cctx->cons, 1 ) );
#endif

    *tiles_used = consensus_added( cctx->cons );

    return consensus_sequence( cctx->cons, 0 );
}

#ifdef FIX_BOUNDARY_ERRORS

static int cmp_mtc_data( const void* x, const void* y )
{
    int** a = (int**)x;
    int** b = (int**)y;

    return ( *a )[ 4 ] - ( *b )[ 4 ];
}

static char* multi_tile_consensus( corrector_context* cctx, Overlap* ovls, int novls, int tb, int te )
{
    cctx->stats_tiles_multi++;

    int re_ab = tb * cctx->twidth;
    int len   = DB_READ_LEN( cctx->db, ovls->aread );
    int re_ae = MIN( len, ( te + 1 ) * cctx->twidth );

#ifdef DEBUG
    printf( "multi_tile_consensus tb %3d te %3d re_ab %5d re_ae %5d\n", tb, te, re_ab, re_ae );
#endif

    int i, j, ovl, ab, ae;
    int doff = 5; // ab, ae, bb, be, q

    if ( novls >= cctx->mtc_dmax )
    {
        cctx->mtc_data  = (int*)realloc( cctx->mtc_data, sizeof( int ) * novls * doff );
        cctx->mtc_dsort = (int**)realloc( cctx->mtc_dsort, sizeof( int* ) * novls );
    }

    int* data  = cctx->mtc_data;
    int** sort = cctx->mtc_dsort;

    bzero( data, sizeof( int ) * doff * novls );

    for ( i = 0; i < novls; i++ )
    {
        sort[ i ] = data + i * doff;
    }

    // collect intervals that need to be aligned

    for ( i = tb; i <= te; i++ )
    {
        ab = i * cctx->twidth;
        ae = ( i + 1 ) * cctx->twidth;

        if ( ae > re_ae )
        {
            ae = re_ae;
        }

        for ( j = cctx->toff[ i ]; j < cctx->toff[ i + 1 ]; j++ )
        {
            ovl = cctx->tovl[ j ].ovl;

            if ( ovl == -1 )
            {
                continue;
            }

            data[ ovl * doff + 4 ] += cctx->tovl[ j ].qv;

            if ( data[ ovl * doff + 1 ] == 0 )
            {
                data[ ovl * doff + 0 ] = ab;
                data[ ovl * doff + 2 ] = cctx->tovl[ j ].bbpos;
            }

            data[ ovl * doff + 1 ] = ae;
            data[ ovl * doff + 3 ] = cctx->tovl[ j ].bepos;

            if ( cctx->tovl[ j ].apos < 0 )
            {
                data[ ovl * doff + 1 ] = -cctx->tovl[ j ].apos;
            }
            else if ( cctx->tovl[ j ].apos > 0 )
            {
                data[ ovl * doff + 0 ] = cctx->tovl[ j ].apos;
            }
        }
    }

    qsort( sort, novls, sizeof( int* ), cmp_mtc_data );

    consensus_reset( cctx->cons );

#ifdef DEBUG_MULTI
    msa_reset( cctx->malign );
#endif

    int bb, be;
    for ( i = 0; i < novls; i++ )
    {
        ovl = ( sort[ i ] - data ) / doff;
        ab  = sort[ i ][ 0 ];
        ae  = sort[ i ][ 1 ];
        bb  = sort[ i ][ 2 ];
        be  = sort[ i ][ 3 ];

        // need to span the region fully

        if ( ae == 0 || ae != re_ae || ab != re_ab )
        {
            continue;
        }

        consensus_add( cctx->cons, cctx->reads[ ovl + 1 ], bb, be ); // , 0, ae-ab);

#ifdef DEBUG_MULTI
        msa_add( cctx->malign, cctx->reads[ ovl + 1 ], -1, -1, bb, be, NULL, 0 );
#endif

        if ( cctx->cons->added == MAX_COVERAGE )
        {
            break;
        }
    }

    if ( cctx->cons->added == 0 )
    {
        return NULL;
    }

#ifdef DEBUG_MULTI
    msa_print( cctx->malign, stdout, 0 );

    // msa_print_v(cctx->malign, stdout, cctx->malign_indent);
    cctx->malign_indent += cctx->malign->added + 1;
#endif

    return consensus_sequence( cctx->cons, 0 );
}

#endif // FIX_BOUNDARY_ERRORS

static int load_read( corrector_context* cctx, int rid, int comp )
{
    if ( cctx->nreads == cctx->maxreads )
    {
        int maxreads = cctx->maxreads * 1.2 + 100;
        cctx->reads  = realloc( cctx->reads, sizeof( char* ) * maxreads );

        for ( ; cctx->maxreads < maxreads; cctx->maxreads++ )
        {
            cctx->reads[ cctx->maxreads ] = New_Read_Buffer( cctx->db );
        }
    }

    Load_Read( cctx->db, rid, cctx->reads[ cctx->nreads ], 0 );

    if ( comp )
    {
        Complement_Seq( cctx->reads[ cctx->nreads ], DB_READ_LEN( cctx->db, rid ) );
    }

    cctx->nreads++;

    return cctx->nreads - 1;
}

#ifdef ADJUST_OFFSETS

static int round_up( int n, int f )
{
    return ( n + f - 1 ) - ( ( n - 1 ) % f );
}

static void adjust_offsets( corrector_context* cctx, Overlap* ovl, int novl )
{
    Alignment aln;
    Path path;

    int twidth = cctx->twidth;

    aln.path = &path;
    aln.aseq = cctx->reads[ 0 ];
    aln.alen = ovl->alen;

    int ntmax   = 1000;
    int* ntrace = (int*)malloc( sizeof( int ) * ntmax );
    int ntlen;

    int i;
    for ( i = 0; i < novl; i++ )
    {
#ifdef DEBUG_OFFSETS
        printf( "OVL %5d x %5d @ %5d..%5d x %5d..%5d\n", ovl[ i ].alen, ovl[ i ].blen, ovl[ i ].path.abpos, ovl[ i ].path.aepos, ovl[ i ].path.bbpos, ovl[ i ].path.bepos );
#endif

        aln.bseq = cctx->reads[ i + 1 ];
        aln.blen = ovl[ i ].blen;

        path = ovl[ i ].path;

        Compute_Trace_MID( &aln, cctx->align_work_data, twidth );

        {
            int a = ovl[ i ].path.abpos;
            int b = ovl[ i ].path.bbpos;
            int p, t;
            int diffs   = 0;
            int matches = 0;

            int ntcur = 0;
            ntlen     = ( round_down( ovl[ i ].path.aepos - 1, twidth ) - round_up( ovl[ i ].path.abpos + 1, twidth ) + twidth ) / twidth * 2 + 2;

            if ( ntlen > ntmax )
            {
                ntmax  = 1.2 * ntmax + ntlen;
                ntrace = (int*)malloc( sizeof( int ) * ntmax );
            }

            // printf("pts  ");

            int bprev = aln.path->bbpos;

            for ( t = 0; t < aln.path->tlen; t++ )
            {
                if ( ( p = ( (int*)( aln.path->trace ) )[ t ] ) < 0 )
                {
                    p = -p - 1;
                    while ( a < p )
                    {
                        if ( aln.aseq[ a ] != aln.bseq[ b ] )
                            diffs++;
                        else
                            matches++;

                        a += 1;
                        b += 1;

                        if ( a % twidth == 0 )
                        {
                            ntrace[ ntcur++ ] = diffs;
                            ntrace[ ntcur++ ] = b - bprev;
                            bprev             = b;

                            // printf(" %4dx%4d %3d %3d", a, b, diffs, matches);
                            diffs = matches = 0;
                        }
                    }

                    diffs++;
                    b += 1;
                }
                else
                {
                    p--;

                    while ( b < p )
                    {
                        if ( aln.aseq[ a ] != aln.bseq[ b ] )
                            diffs++;
                        else
                            matches++;

                        a += 1;
                        b += 1;

                        if ( a % twidth == 0 )
                        {
                            ntrace[ ntcur++ ] = diffs;
                            ntrace[ ntcur++ ] = b - bprev;
                            bprev             = b;

                            // printf(" %4dx%4d %3d %3d", a, b, diffs, matches);
                            diffs = matches = 0;
                        }
                    }

                    diffs++;
                    a += 1;

                    if ( a % twidth == 0 )
                    {
                        ntrace[ ntcur++ ] = diffs;
                        ntrace[ ntcur++ ] = b - bprev;
                        bprev             = b;

                        // printf(" %4dx%4d %3d %3d", a, b, diffs, matches);
                        diffs = matches = 0;
                    }
                }
            }

            p = aln.path->aepos;
            while ( a < p )
            {
                if ( aln.aseq[ a ] != aln.bseq[ b ] )
                    diffs++;
                else
                    matches++;

                a += 1;
                b += 1;

                if ( a % twidth == 0 && a != ovl[ i ].path.aepos )
                {
                    ntrace[ ntcur++ ] = diffs;
                    ntrace[ ntcur++ ] = b - bprev;
                    bprev             = b;

                    // printf(" %4dx%4d %3d %3d", a, b, diffs, matches);
                    diffs = matches = 0;
                }
            }

            ntrace[ ntcur++ ] = diffs;
            ntrace[ ntcur++ ] = b - bprev;

#ifdef DEBUG_OFFSETS
            if ( ntcur != ntlen )
            {
                a = round_up( ovl[ i ].path.abpos + 1, twidth );
                b = ovl[ i ].path.bbpos;
                printf( "ntcur %d != ntlen %d\n", ntcur, ntlen );
                printf( "%d..%d x %d..%d -> %d pts\n", ovl[ i ].path.abpos, ovl[ i ].path.aepos, ovl[ i ].path.bbpos, ovl[ i ].path.bepos, ntlen );

                printf( "npts  " );
                for ( t = 0; t < ntcur - 1; t += 2 )
                {
                    b += ntrace[ t + 1 ];
                    printf( " %d@%d/%dx%d", ntrace[ t ], ntrace[ t + 1 ], a, b );

                    a += twidth;
                }
                printf( " %d@end", ntrace[ ntcur - 1 ] );

                printf( "\n" );
            }
#endif
        }

        int j;

#ifdef DEBUG_OFFSETS
        printf( "(%3d)", ovl[ i ].path.tlen );
        for ( j = 1; j < ovl[ i ].path.tlen; j += 2 )
        {
            if ( j > ntlen || j > ovl[ i ].path.tlen || ( (uint16*)( ovl[ i ].path.trace ) )[ j ] != ntrace[ j ] )
            {
                printf( ANSI_COLOR_RED " %3d" ANSI_COLOR_RESET, ( (uint16*)( ovl[ i ].path.trace ) )[ j ] );
            }
            else
            {
                printf( " %3d", ( (uint16*)( ovl[ i ].path.trace ) )[ j ] );
            }
        }
        printf( "\n" );

        printf( "(%3d)", ntlen );
        for ( j = 1; j < ntlen; j += 2 )
        {
            printf( " %3d", ntrace[ j ] );
        }
        printf( "\n\n" );
#endif

        assert( ovl[ i ].path.tlen == ntlen );

        for ( j = 0; j < ntlen; j++ )
        {
            ( (uint16*)ovl[ i ].path.trace )[ j ] = ntrace[ j ];
        }
    }
}

#endif

static void write_seq( FILE* file, char* seq )
{
    const int width = 100;
    int len         = strlen( seq );
    int j;

    for ( j = 0; j + width < len; j += width )
    {
        fprintf( file, "%.*s\n", width, seq + j );
    }

    if ( j < len )
    {
        fprintf( file, "%s\n", seq + j );
    }
}

static char* append_consensus( corrector_context* cctx, char* seqcons, int tiles_used )
{
    int ncons = strlen( seqcons );

    if ( ncons + cctx->curcons >= cctx->maxcons )
    {
        cctx->maxcons = cctx->maxcons * 1.2 + ncons + 100;
        cctx->seqcons = (char*)realloc( cctx->seqcons, cctx->maxcons );
    }

    cctx->tiles[ cctx->curtiles + 0 ] = ncons;
    cctx->tiles[ cctx->curtiles + 1 ] = tiles_used;
    cctx->curtiles += 2;

    char* start = cctx->seqcons + cctx->curcons;

    strcpy( start, seqcons );
    cctx->curcons += ncons;

    return start;
}

#ifdef FIX_BOUNDARY_ERRORS

static char* fix_boundary_errors( corrector_context* cctx, Overlap* ovl, int novl )
{
    static char n2a[] = {'a', 'c', 'g', 't'};

    Alignment aln;
    Path path;

    aln.path = &path;

    int i, t, slen;
    int last_tile = cctx->ce_first_tile + cctx->ce_tcur - 2;
    char* seq_multi;

    int boundary;
    int boundary_b, boundary_e;

    int tmax    = 0;
    int tcur    = 0;
    int* trace  = NULL;
    int diffs_a = 0;
    int diffs_b = 0;

    for ( i = 0, t = cctx->ce_first_tile; t < last_tile; t++, i++ )
    {
        seq_multi = multi_tile_consensus( cctx, ovl, novl, t, t + 1 );

#ifdef DEBUG_FIX_CONSENSUS
        printf( "\n\nTILE %d OFF %d", t, cctx->ce_tiles[ i ] );
#endif

        if ( seq_multi == NULL )
        {
#ifdef DEBUG_FIX_CONSENSUS
            printf( " -> no multi\n" );
#endif
            continue;
        }

        slen       = cctx->ce_tiles[ i + 2 ] - cctx->ce_tiles[ i ];
        boundary   = cctx->ce_tiles[ i + 1 ] - cctx->ce_tiles[ i ];
        boundary_b = (int)( boundary * ( 1. - CE_CENTER_DISTANCE ) );
        boundary_e = (int)( boundary * ( 1. + CE_CENTER_DISTANCE ) );

        if ( slen >= cctx->ce_smax )
        {
            cctx->ce_smax        = slen * 1.2;
            cctx->ce_seq_singles = (char*)realloc( cctx->ce_seq_singles, sizeof( char ) * cctx->ce_smax );
        }

        strncpy( cctx->ce_seq_singles, cctx->seqcons + cctx->ce_tiles[ i ], slen );
        cctx->ce_seq_singles[ slen ] = '\0';

        aln.aseq = cctx->ce_seq_singles;
        aln.alen = slen;

        aln.bseq = seq_multi;
        aln.blen = strlen( seq_multi );

        if ( aln.alen == aln.blen )
        {
#ifdef DEBUG_FIX_CONSENSUS
            printf( " -> alen = blen\n" );
#endif

            continue;
        }

#ifdef DEBUG_FIX_CONSENSUS
        printf( "\n%3d %s\n%3d %s\n", aln.alen, aln.aseq, aln.blen, aln.bseq );
#endif

        Number_Read( aln.aseq );
        Number_Read( aln.bseq );

        path.tlen  = 0;
        path.trace = NULL;

        path.abpos = 0;
        path.aepos = aln.alen;

        path.bbpos = 0;
        path.bepos = aln.blen;

        path.diffs = aln.alen + aln.blen;

        Compute_Trace_ALL( &aln, cctx->align_work_data );

        int j;

#ifdef DEBUG_FIX_CONSENSUS
        printf( "%d / boundary %d %d..%d\n", path.diffs, boundary, boundary_b, boundary_e );

        Print_Reference( stdout, &aln, cctx->align_work_data, 0, 100, 0, 0, 5 );

        printf( "\n" );

        printf( "trace" );
        for ( j = 0; j < path.tlen; j++ )
        {
            int tp = ( (int*)path.trace )[ j ];

            if ( abs( tp ) > boundary * ( 1 - CE_CENTER_DISTANCE ) &&
                 abs( tp ) < boundary * ( 1 + CE_CENTER_DISTANCE ) )
            {
                printf( ANSI_COLOR_GREEN " %d" ANSI_COLOR_RESET, tp );
            }
            else
            {
                printf( " %d", tp );
            }
        }
        printf( "\n" );
#endif

        if ( 2 * path.tlen + tcur > tmax )
        {
            tmax  = tmax * 1.2 + path.tlen * 2;
            trace = (int*)realloc( trace, sizeof( int ) * tmax );
        }

        int a = 0;
        int b = 0;

        for ( j = 0; j < path.tlen; j++ )
        {
            int c = ( (int*)path.trace )[ j ];

            if ( c < 0 )
            {
                c = -c - 1;
                while ( a < c )
                {
                    a++;
                    b++;
                }

                if ( c >= boundary_b && c <= boundary_e )
                {
                    trace[ tcur++ ] = -cctx->ce_tiles[ i ] - ( c + 1 );
                    trace[ tcur++ ] = aln.bseq[ b ];
                    diffs_a++;
                }

                b++;
            }
            else
            {
                c = c - 1;

                while ( b < c )
                {
                    a++;
                    b++;
                }

                if ( c >= boundary_b && c <= boundary_e )
                {
                    trace[ tcur++ ] = a + 2 + cctx->ce_tiles[ i ];
                    trace[ tcur++ ] = -1;
                    diffs_b++;
                }

                a++;
            }
        }

#ifdef DEBUG_FIX_CONSENSENSUS
        printf( "trace" );
        for ( j = 0; j < tcur; j += 2 )
        {
            printf( " (%d %d)", trace[ j ], trace[ j + 1 ] );
        }
        printf( "\n\n" );
#endif
    }

    if ( diffs_a + cctx->curcons >= cctx->maxcons )
    {
        cctx->maxcons = cctx->maxcons * 1.2 + diffs_a + 100;
        cctx->seqcons = (char*)realloc( cctx->seqcons, cctx->maxcons );
    }

    int n = cctx->curcons - 1 + diffs_a;
    int p = cctx->curcons - 1;

#ifdef DEBUG_FIX_CONSENSUS
    write_seq( stdout, cctx->seqcons );

    printf( "diffs_a = %d ... diffs_b = %d\n", diffs_a, diffs_b );
#endif

    for ( i = tcur - 2; i >= 0; i -= 2 )
    {
        int c = trace[ i ];

        if ( c > 0 )
        {
            c--;

            while ( c <= p )
            {
                cctx->seqcons[ n ] = cctx->seqcons[ p ];

                n--;
                p--;
            }

            /*
            cctx->seqcons[n] = '-';
            n--;
            */

            p--;
        }
        else
        {
            c = -c;
            c--;

            while ( c <= p )
            {
                cctx->seqcons[ n ] = cctx->seqcons[ p ];

                n--;
                p--;
            }

            cctx->seqcons[ n ] = n2a[ trace[ i + 1 ] ]; // '+';
            n--;
        }
    }

    while ( p >= 0 )
    {
        cctx->seqcons[ n ] = cctx->seqcons[ p ];
        n--;
        p--;
    }

    n++;

    cctx->curcons += diffs_a;
    cctx->seqcons[ cctx->curcons ] = '\0';

#ifdef DEBUG_FIX_CONSENSUS
    write_seq( stdout, cctx->seqcons + n );
#endif

    free( trace );

    return cctx->seqcons + n;
}

#endif // FIX_BOUNDARY_ERRORS

static int break_read( corrector_context* cctx, Overlap* ovl, int novl )
{
#ifndef FIX_BOUNDARY_ERRORS
    UNUSED( novl );
#endif

    if ( cctx->curcons == 0 )
    {
        return 0;
    }

    // validate consensus using multi-tile alignments

    // cctx->ce_tiles[ cctx->ce_tcur ] = cctx->curcons;
    // cctx->ce_tcur++;

    int len = strlen( cctx->seqcons );

#ifdef DEBUG_FIX_CONSENSUS

    if ( len > cctx->minRLen )
    {
        fprintf( cctx->fileOut, ">%d.%d source=%d\n", ovl->aread, cctx->ncorrected, ovl->aread );
        write_seq( cctx->fileOut, cctx->seqcons );
    }

#endif

    char* seq;

#ifdef FIX_BOUNDARY_ERRORS
    seq = fix_boundary_errors( cctx, ovl, novl );
#else
    seq = cctx->seqcons;
#endif

    len = strlen( seq );

#ifdef DEBUG_MULTI
    cctx->malign_indent = 0;
#endif

    if ( len > cctx->minRLen )
    {
        int ab   = cctx->ce_first_tile * cctx->twidth;
        int alen = cctx->db->reads[ ovl->aread ].rlen;
        int ae   = MIN( alen, ( cctx->ce_first_tile + cctx->ce_tcur - 1 ) * cctx->twidth );

        fprintf( cctx->fileOut, ">%d.%d source=%d,%d,%d correctionq=", ovl->aread, cctx->ncorrected, ovl->aread, ab, ae );

        int i;
        for ( i = 0; i < cctx->curtiles; i += 2 )
        {
            if ( i > 0 )
                fprintf( cctx->fileOut, "," );

            fprintf( cctx->fileOut, "%d,%d", cctx->tiles[ i ], cctx->tiles[ i + 1 ] );
        }

#ifdef TRACK_POSITIONS
        Alignment aln;
        Path path;

        aln.path = &path;

        aln.aseq = cctx->reads[0];
        aln.alen = DB_READ_LEN(cctx->db, ovl->aread);

        aln.bseq = seq;
        aln.blen = strlen( aln.bseq );
        Number_Read( aln.bseq );

        path.tlen  = 0;
        path.trace = NULL;

        path.abpos = 0;
        path.aepos = aln.alen;

        path.bbpos = 0;
        path.bepos = aln.blen;

        path.diffs = aln.alen + aln.blen;

        Compute_Trace_ALL( &aln, cctx->align_work_data );

        if ( path.tlen > 0 )
        {
            fprintf(cctx->fileOut, " postrace=");
            for (i = 0; i < path.tlen; i++)
            {
                if (i > 0 )
                {
                    fprintf(cctx->fileOut, ",");
                }
                fprintf(cctx->fileOut, "%d", ((int*)(path.trace))[i]);
            }
        }

        // Print_Alignment(stdout, &aln, cctx->align_work_data, 0, 100, 0, 100, 0);
        // trace_to_posmap(path.trace, path.tlen, aln.alen);

        Lower_Read( aln.bseq);
#endif

        fprintf( cctx->fileOut, "\n" );

        write_seq( cctx->fileOut, seq );

        cctx->ncorrected++;
    }

    cctx->seqcons[ 0 ] = '\0';
    cctx->curcons      = 0;

    cctx->curtiles = 0;

    cctx->ce_tcur = 0;

    return 1;
}

static int cmp_ovl_length( const void* a, const void* b )
{
    Overlap* x = (Overlap*)a;
    Overlap* y = (Overlap*)b;

    int len_x = x->path.aepos - x->path.abpos;
    int len_y = y->path.aepos - y->path.abpos;

    return len_y - len_x;
}

static void correct_overlaps( corrector_context* cctx, Overlap* pOvls, int nOvls )
{
    int a = pOvls->aread;

    // if ( a != 48506 ) return ;

#ifdef VERBOSE
    printf( "READ %8d (%5d) CONSENSUS [ SINGLE %5d MULTI %5d ]\n",
            a, cctx->db->reads[ pOvls->aread ].rlen, cctx->stats_tiles_single, cctx->stats_tiles_multi );
#endif

    // verify that the PT points are valid

    int i;
    for ( i = 0; i < nOvls; i++ )
    {
        if ( !has_valid_pt_points( pOvls + i ) )
        {
            printf( "ERROR: bad pt points ovl %d bread %d\n", i, pOvls[ i ].bread );
            return;
        }
    }

    // init pos tracking
    int alen = DB_READ_LEN( cctx->db, a );

    // alloc & init
    int ntiles = ( alen + cctx->twidth - 1 ) / cctx->twidth;

    if ( ntiles + 1 >= cctx->ntoff )
    {
        cctx->ntoff = cctx->ntoff * 1.2 + ntiles + 1;
        cctx->toff  = (int*)realloc( cctx->toff, sizeof( int ) * cctx->ntoff );
    }

    bzero( cctx->toff, sizeof( int ) * cctx->ntoff );

    int t;

    for ( i = 0; i < nOvls; i++ )
    {
        for ( t = round_down( pOvls[ i ].path.abpos, cctx->twidth ) / cctx->twidth;
              t <= round_down( pOvls[ i ].path.aepos - 1, cctx->twidth ) / cctx->twidth;
              t++ )
        {
            if ( cctx->toff[ t + 1 ] < MAX_TILES )
            {
                cctx->toff[ t + 1 ]++;
            }
        }
    }

#ifdef USE_A_TILES
    for ( i = 0; i < ntiles; i++ )
    {
        cctx->toff[ i + 1 ] += 1;
    }
#endif

    for ( i = 1; i <= ntiles; i++ )
    {
        cctx->toff[ i ] += cctx->toff[ i - 1 ];
    }

    if ( cctx->toff[ ntiles ] >= cctx->ntovl )
    {
        cctx->ntovl = cctx->ntovl * 1.2 + cctx->toff[ ntiles ] + 1;
        cctx->tovl  = (tile_overlap*)realloc( cctx->tovl, sizeof( tile_overlap ) * cctx->ntovl );
    }

    // fill in tiles

    int bb, be, tile;
    unsigned short* trace;
    int p = 0;

    int* curtiles = malloc( sizeof( int ) * ntiles );
    bzero( curtiles, sizeof( int ) * ntiles );

// A tiles

#ifdef USE_A_TILES

    track_anno oqa = cctx->qtrack_offset[ a ] / sizeof( track_data );

    int readidx = load_read( cctx, a, 0 );

    for ( i = 0; i < ntiles; i++ )
    {
        // p = cctx->toff[ i ]++;
        // curtiles[ i ]++;

        p = cctx->toff[i] + curtiles[i];
        curtiles[i] += 1;

        cctx->tovl[ p ].read  = readidx;
        cctx->tovl[ p ].a     = a;
        cctx->tovl[ p ].b     = -1;
        cctx->tovl[ p ].apos  = 0;
        cctx->tovl[ p ].bbpos = cctx->twidth * i;
        cctx->tovl[ p ].bepos = cctx->twidth * ( i + 1 );

        cctx->tovl[ p ].qv = cctx->qtrack_data[ oqa + i ];
    }

    cctx->tovl[ p ].bepos = alen;

#endif // USE_A_TILES

    // B tiles

    for ( i = 0; i < nOvls; i++ )
    {
        int bread = pOvls[ i ].bread;

        tile    = pOvls[ i ].path.abpos / cctx->twidth;
        trace   = pOvls[ i ].path.trace;
        be      = pOvls[ i ].path.bbpos + trace[ 1 ];
        readidx = -1;

        if ( curtiles[ tile ] < MAX_TILES )
        {
            // p = cctx->toff[ tile ]++;
            // curtiles[ tile ]++;

            p = cctx->toff[tile] + curtiles[tile];
            curtiles[tile] += 1;

            readidx = load_read( cctx, bread, pOvls[ i ].flags & OVL_COMP );

            cctx->tovl[ p ].read  = readidx;
            cctx->tovl[ p ].a     = a;
            cctx->tovl[ p ].b     = bread;
            cctx->tovl[ p ].bbpos = pOvls[ i ].path.bbpos;
            cctx->tovl[ p ].bepos = be;

            cctx->tovl[ p ].qv = trace[ 0 ];

            if ( pOvls[ i ].path.abpos % cctx->twidth )
            {
                cctx->tovl[ p ].apos = pOvls[ i ].path.abpos;
            }
            else
            {
                cctx->tovl[ p ].apos = 0;
            }
        }

        for ( t = 2; t < pOvls[ i ].path.tlen; t += 2 )
        {
            tile++;
            bb = be;

            if ( t == pOvls[ i ].path.tlen - 1 )
            {
                be = pOvls[ i ].path.bepos + 1;
            }
            else
            {
                be += trace[ t + 1 ];
            }

            if ( curtiles[ tile ] < MAX_TILES )
            {
                // curtiles[ tile ]++;
                // p = cctx->toff[ tile ]++;

                p = cctx->toff[tile] + curtiles[tile];
                curtiles[tile] += 1;

                if ( readidx == -1 )
                {
                    readidx = load_read( cctx, bread, pOvls[ i ].flags & OVL_COMP );
                }

                cctx->tovl[ p ].read  = readidx;
                cctx->tovl[ p ].a     = a;
                cctx->tovl[ p ].b     = bread;
                cctx->tovl[ p ].bbpos = bb;
                cctx->tovl[ p ].bepos = be;
                cctx->tovl[ p ].apos  = 0;

                cctx->tovl[ p ].qv = trace[ t ];
            }
            else
            {
                p = -1;
            }
        }

        if ( p != -1 && pOvls[ i ].path.aepos < alen && ( pOvls[ i ].path.aepos % cctx->twidth ) )
        {
            cctx->tovl[ p ].apos = -pOvls[ i ].path.aepos;
        }
    }

    free( curtiles );

    /*
    for ( i = ntiles; i >= 1; i-- )
    {
        cctx->toff[ i ] = cctx->toff[ i - 1 ];
    }
    cctx->toff[ 0 ] = 0;
    */

    cctx->curcons = 0;

    for ( i = 0; i < ntiles; i++ )
    {
        int tovl_b = cctx->toff[ i ];
        int tovl_e = cctx->toff[ i + 1 ];

#ifdef USE_A_TILES
        qsort( cctx->tovl + tovl_b + 1, tovl_e - ( tovl_b + 1 ), sizeof( tile_overlap ), cmp_tovl_qv );
#else
        qsort( cctx->tovl + tovl_b, tovl_e - tovl_b, sizeof( tile_overlap ), cmp_tovl_qv );
#endif
    }

    // correct tiles

    cctx->ce_tcur = 0;

    /*
    int trim_b, trim_e;
    get_trim( cctx->db, cctx->trimtrack, a, &trim_b, &trim_e );

    trim_b = trim_b / cctx->twidth;
    trim_e = ( trim_e + cctx->twidth - 1 ) / cctx->twidth;

    assert( trim_e <= ntiles );
    */

    for ( i = 0; i < ntiles; i++ )
    // for ( i = trim_b ; i < trim_e ; i++ )
    {

#ifdef DEBUG
        print_tiles( cctx, i );
#endif

        int tiles_used;
        char* seqcons = single_tile_consensus( cctx, i, &tiles_used );

        // printf("%3d %3d %s\n", i, tiles_used, seqcons);

        /*
#ifdef USE_A_TILES
        if ( tiles_used < 2 )
#else
        if ( tiles_used < 1 )
#endif
        {
            break_read( cctx, pOvls, nOvls );
        }
        else
*/
        {
            if ( cctx->ce_tcur == 0 )
            {
                cctx->ce_first_tile = i;
            }

            cctx->ce_tiles[ cctx->ce_tcur ] = cctx->curcons;
            cctx->ce_tcur++;

            append_consensus( cctx, seqcons, tiles_used );
        }
    }

    if ( cctx->curcons > 0 )
    {
        break_read( cctx, pOvls, nOvls );
    }

    cctx->nreads = 0;
}

static void* corrector_thread( void* arg )
{
    corrector_arg* carg = (corrector_arg*)arg;
    FILE* fileOvls      = carg->fileOvls;
    corrector_context cctx;

    cctx.cons = consensus_init();

#ifdef DEBUG_MULTI
    cctx.malign        = msa_init();
    cctx.malign_indent = 0;
#endif

    cctx.minRLen = carg->minRLen;

    cctx.ntoff = cctx.ntovl = 0;
    cctx.nreads = cctx.maxreads = 0;
    cctx.toff                   = NULL;
    cctx.tovl                   = NULL;
    cctx.reads                  = NULL;
    cctx.fileOut                = carg->fileOut;
    cctx.fastaHeader            = carg->fastaHeader;
    cctx.db                     = &( carg->db );
    cctx.seqcons                = NULL;
    cctx.maxcons                = 0;
    cctx.tiles                  = malloc( sizeof( int ) * 2 * ( carg->db.maxlen / carg->twidth + 1 ) );
    cctx.curtiles               = 0;
    cctx.twidth                 = carg->twidth;
    cctx.qtrack_offset          = carg->qtrack->anno;
    cctx.qtrack_data            = carg->qtrack->data;
    // cctx.trimtrack              = carg->trimtrack;
    cctx.track                  = malloc( sizeof( int ) * carg->db.maxlen );

    cctx.stats_tiles_single = 0;
    cctx.stats_tiles_multi  = 0;

    cctx.thread = carg->thread;

    cctx.mtc_dmax  = 0;
    cctx.mtc_data  = NULL;
    cctx.mtc_dsort = NULL;

    cctx.ce_smax        = 0;
    cctx.ce_seq_singles = NULL;

    cctx.ce_tcur  = 0;
    cctx.ce_tiles = (int*)malloc( sizeof( int ) * ( cctx.db->maxlen / cctx.twidth + 1 ) );

    cctx.align_work_data = New_Work_Data();

    ovl_trace* trace = NULL;
    int tmax, tcur;
    tcur = tmax = 0;

    size_t tbytes = TBYTES( cctx.twidth );

    Overlap* pOvls = NULL;
    int omax       = 500;
    pOvls          = (Overlap*)malloc( sizeof( Overlap ) * omax );

    fseek( fileOvls, carg->start, SEEK_SET );

    while ( !Read_Overlap( fileOvls, pOvls ) && pOvls->flags & OVL_DISCARD )
    {
        fseek( fileOvls, tbytes * pOvls->path.tlen, SEEK_CUR );
    }

    int a, n;
    n = 0;

    cctx.ncorrected = 0;

    while ( ftell( fileOvls ) < carg->end )
    {
        pOvls[ 0 ] = pOvls[ n ];
        a          = pOvls->aread;

        if ( pOvls->path.tlen > tmax )
        {
            tmax  = tmax * 1.2 + pOvls->path.tlen;
            trace = (ovl_trace*)realloc( trace, sizeof( ovl_trace ) * tmax );
        }

        tcur              = 0;
        pOvls->path.trace = trace;

        read_unpacked_trace( fileOvls, pOvls, tbytes );

        tcur += pOvls->path.tlen;

        n = 1;

        while ( 1 )
        {
            if ( Read_Overlap( fileOvls, pOvls + n ) || pOvls[ n ].aread != a )
            {
                break;
            }

            if ( pOvls[ n ].flags & OVL_DISCARD )
            {
                fseek( fileOvls, tbytes * pOvls[ n ].path.tlen, SEEK_CUR );
                continue;
            }

            if ( tcur + pOvls[ n ].path.tlen >= tmax )
            {
                tmax  = tmax * 1.2 + pOvls[ n ].path.tlen;
                trace = (ovl_trace*)realloc( trace, sizeof( ovl_trace ) * tmax );

                tcur = 0;
                int k;
                for ( k = 0; k < n; k++ )
                {
                    pOvls[ k ].path.trace = trace + tcur;
                    tcur += pOvls[ k ].path.tlen;
                }
            }

            pOvls[ n ].path.trace = trace + tcur;
            tcur += pOvls[ n ].path.tlen;

            read_unpacked_trace( fileOvls, pOvls + n, tbytes );

            n += 1;
            if ( n >= omax )
            {
                omax  = 1.2 * n + 10;
                pOvls = (Overlap*)realloc( pOvls, sizeof( Overlap ) * omax );
            }
        }

        qsort( pOvls, n, sizeof( Overlap ), cmp_ovl_length );

#ifdef ADJUST_OFFSETS
        adjust_offsets( &cctx, pOvls, n );
#endif

        correct_overlaps( &cctx, pOvls, n );
    }

    if ( cctx.toff != NULL )
    {
        free( cctx.toff );
    }

    if ( cctx.tovl != NULL )
    {
        free( cctx.tovl );
    }

    int i;
    for ( i = 0; i < cctx.maxreads; i++ )
    {
        free( cctx.reads[ i ] - 1 );
    }

    free( cctx.reads );
    free( cctx.mtc_data );
    free( cctx.mtc_dsort );

    free( cctx.ce_tiles );
    free( cctx.ce_seq_singles );

    Free_Work_Data( cctx.align_work_data );

    consensus_free( cctx.cons );

#ifdef DEBUG_MULTI
    msa_free( cctx.malign );
#endif

#ifdef WRITE_STATISTICS
    fprintf( carg->fileStats, "BREAK_COV       %5d\n", cctx.stats_break_cov );
    fprintf( carg->fileStats, "BREAK_LQ        %5d\n", cctx.stats_break_lq );
    fprintf( carg->fileStats, "BREAK_UNKNOWN   %5d\n", cctx.stats_break_unknown );
#endif

    return NULL;
}

static void usage()
{
    printf( "[-v] [-jx <int>] [-q <track>] <db> <in.las> <out.fasta>\n" );
    printf( "options: -v ... verbose\n" );
    printf( "         -j ... number of threads\n" );
    printf( "         -x ... minimum sequence length (%d)\n", DEF_ARG_X );
    printf( "         -q ... q track (%s)\n", DEF_ARG_Q );
}

int main( int argc, char* argv[] )
{
    FILE* fileOvls;
    HITS_DB db;
    int64 novl;
    int twidth;

    int verbose   = 0;
    int nThreads  = 1;
    int minLength = DEF_ARG_X;

    char* qTrackName    = DEF_ARG_Q;

    // process arguments

    int c;

    opterr = 0;

    while ( ( c = getopt( argc, argv, "vj:x:q:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'v':
                verbose++;
                break;

            case 'j':
                nThreads = atoi( optarg );
                break;

            case 'x':
                minLength = atoi( optarg );
                break;

            case 'q':
                qTrackName = optarg;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind != 3 )
    {
        usage();
        exit( 1 );
    }

    char* pcPathReadsIn  = argv[ optind++ ];
    char* pcPathOverlaps = argv[ optind++ ];
    char* pcBaseOut      = argv[ optind++ ];

    if ( ( fileOvls = fopen( pcPathOverlaps, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathOverlaps );
        exit( 1 );
    }

    if ( fread( &novl, sizeof( novl ), 1, fileOvls ) != 1 )
    {
        fprintf( stderr, "failed to read %s header\n", pcPathOverlaps );
        exit( 1 );
    }

    if ( fread( &twidth, sizeof( twidth ), 1, fileOvls ) != 1 )
    {
        fprintf( stderr, "failed to read %s header\n", pcPathOverlaps );
        exit( 1 );
    }

    // init

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathReadsIn );
        exit( 1 );
    }

    HITS_TRACK* qtrack = track_load( &db, qTrackName );

    if ( qtrack == NULL )
    {
        fprintf( stderr, "could not load quality track %s\n", qTrackName );
        exit( 1 );
    }

    if ( minLength < 0 )
    {
        fprintf( stderr, "negative read lengths are not supported\n" );
        exit( 1 );
    }

    off_t* offsets = partition_overlaps( fileOvls, nThreads );

    fclose( fileOvls );

    int i;
    pthread_t* threads   = (pthread_t*)malloc( sizeof( pthread_t ) * nThreads );
    corrector_arg* cargs = (corrector_arg*)malloc( sizeof( corrector_arg ) * nThreads );

    char* pcOut = (char*)malloc( strlen( pcBaseOut ) + 20 );

    for ( i = 0; i < nThreads; i++ )
    {
        cargs[ i ].qtrack    = qtrack;

        cargs[ i ].thread  = i;
        cargs[ i ].start   = offsets[ i ];
        cargs[ i ].end     = offsets[ i + 1 ];
        cargs[ i ].twidth  = twidth;
        cargs[ i ].minRLen = minLength;

        cargs[ i ].fileOvls = fopen( pcPathOverlaps, "r" );

        sprintf( pcOut, "%s.%02d.fasta", pcBaseOut, i );
        cargs[ i ].fileOut = fopen( pcOut, "w" );

#ifdef WRITE_STATISTICS
        sprintf( pcOut, "%s.%02d.stats", pcBaseOut, i );
        cargs[ i ].fileStats = fopen( pcOut, "w" );
#endif

        memcpy( &( cargs[ i ].db ), &db, sizeof( HITS_DB ) );

        cargs[ i ].db.bases    = NULL;
        cargs[ i ].fastaHeader = pcBaseOut;
    }

    free( pcOut );

    for ( i = 0; i < nThreads; i++ )
    {
        // if (cargs[i].start < cargs[i].end)
        pthread_create( threads + i, NULL, corrector_thread, cargs + i );
    }

    for ( i = 0; i < nThreads; i++ )
    {
        pthread_join( threads[ i ], NULL );
    }

    for ( i = 0; i < nThreads; i++ )
    {
        fclose( cargs[ i ].fileOvls );
        fclose( cargs[ i ].fileOut );
        fclose( cargs[ i ].db.bases );

#ifdef WRITE_STATISTICS
        fclose( cargs[ i ].fileStats );
#endif
    }

    free( cargs );
    free( threads );

    Close_DB( &db );

    return 0;
}
