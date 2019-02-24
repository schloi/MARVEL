/*******************************************************************************************
 *
 *  filters overlaps by various criteria
 *
 *  Author :  MARVEL Team
 *
 *******************************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

#include "lib.ext/fgetln.h"
#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/read_loader.h"
#include "lib/tracks.h"
#include "lib/trim.h"
#include "lib/utils.h"

#include "dalign/align.h"
#include "db/DB.h"

// command line defaults

#define DEF_ARG_S -1
#define DEF_ARG_T TRACK_TRIM
#define DEF_ARG_TT 0
#define DEF_ARG_R TRACK_REPEATS

#define DEF_ARG_NN_LEN 2000
#define DEF_ARG_NN_RATE 80
#define DEF_ARG_NN_FUZZ 0

// constants

#define BIN_SIZE 100
#define MIN_LR 500

// read flags, used for rules processing

#define READ_NONE 0x0
#define READ_DISCARD ( 0x1 << 0 )
#define READ_STRICT ( 0x1 << 1 )

// debug toggles

#define VERBOSE
#undef VERBOSE_STITCH
#undef DEBUG_REPEAT_EXTENSION

// macros

#ifdef VERBOSE_STITCH
#define OVL_STRAND( ovl ) ( ( ( ovl )->flags & OVL_COMP ) ? 'c' : 'n' )
#endif


typedef struct
{
    // stats counters
    int nFilteredDiffs;
    int nFilteredDiffsSegments;
    int nFilteredUnalignedBases;
    int nFilteredLength;
    int nFilteredRepeat;
    int nFilteredReadLength;
    int nRepeatOvlsKept;
    int nFilteredLocalEnd;

    // settings
    int nStitched;
    float fMaxDiffs;
    int nMaxUnalignedBases, nMinAlnLength;
    int nMinNonRepeatBases, nMinReadLength;
    int nVerbose;
    int stitch;
    int stitch_aggressively;
    int rm_cov;        // repeat modules, coverage
    int rm_aggressive; // -M
    int do_trim;
    int contained;     // -c

    int hrd;      // -N
    int hrd_len;  // -N x,-,-
    int hrd_rate; // -N -,x,-
    int hrd_fuzz; // -N -,-,x

    // hrd

    Overlap*** hrd_groups;
    int* hrd_ngroups;
    int* hrd_maxgroups;
    int hrd_allocated;
    uint16_t* hrd_mapgroup;

    // repeat modules - merged repeats
    int rm_merge;
    int rm_mode;

    track_data* rm_repeat;
    unsigned int rm_maxrepeat;

    uint64_t* rm_bins;
    int rm_maxbins;

    // repeat modules - result track

    track_anno* rm_anno;
    track_data* rm_data;
    track_anno rm_ndata;
    track_anno rm_maxdata;

    // local ends
    int* le_lbins;
    int* le_rbins;
    int le_maxbins;

    HITS_DB* db;
    HITS_TRACK* trackRepeat;
    HITS_TRACK* trackTrim;
    HITS_TRACK* trackRepeatStrict;

    int* r2bin;
    int max_r2bin;

    int useRLoader;
    TRIM* trim;
    Read_Loader* rl;

    ovl_header_twidth twidth;

} FilterContext;


typedef struct
{
    int* exclude_reads;
    int exclude_reads_n;

    int* exclude_edges;
    int exclude_edges_n;

    int* include_edges;
    int include_edges_n;

    int* strict_reads;
    int strict_reads_n;
} FilterRules;


extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_int( const void* a, const void* b )
{
    return ( *(int*)a ) - ( *(int*)b );
}

static int cmp_2int( const void* a, const void* b )
{
    int cmp = cmp_int( a, b );

    if ( cmp )
    {
        return cmp;
    }

    return cmp_int( ( (int*)a ) + 1, ( (int*)b ) + 1 );
}

static int cmp_ovl_q_desc( const void* a, const void* b )
{
    Overlap* o1 = *(Overlap**)a;
    Overlap* o2 = *(Overlap**)b;

    float q1 = 1.0 * o1->path.diffs / ( o1->path.aepos - o1->path.abpos );
    float q2 = 1.0 * o2->path.diffs / ( o2->path.aepos - o2->path.abpos );

    if ( q1 < q2 )
    {
        return 1;
    }
    else if ( q2 > q1 )
    {
        return -1;
    }

    return 0;
}

static void fread_rules( FILE* fin, FilterRules* rules )
{
    char* line;
    size_t len;
    int* singles      = NULL;
    int singles_cur   = 0;
    int singles_max   = 0;
    int* multi_inc    = NULL;
    int multi_inc_cur = 0;
    int multi_inc_max = 0;
    int* multi_ex     = NULL;
    int multi_ex_cur  = 0;
    int multi_ex_max  = 0;
    int* strict       = NULL;
    int strict_cur    = 0;
    int strict_max    = 0;

    while ( ( line = fgetln( fin, &len ) ) != NULL )
    {
        char* end = line + len;

        if ( len <= 1 || line[ 0 ] == '#' )
        {
            continue;
        }

        char mode = *line;

        if ( mode != '-' && mode != '+' && mode != 's' )
        {
            printf( "malformed line: %*s\n", (int)( len - 1 ), line );
            continue;
        }

        int single = 1;
        char* sep;
        for ( sep = line + 1; sep != end; sep++ )
        {
            if ( *sep == '-' )
            {
                single = 0;
                break;
            }
        }

        if ( single )
        {
            if ( mode == '-' )
            {
                if ( singles_max == singles_cur )
                {
                    singles_max = singles_max * 1.2 + 100;
                    singles     = realloc( singles, sizeof( int ) * singles_max );
                }

                singles[ singles_cur ] = strtoimax( line + 1, NULL, 10 );
                singles_cur += 1;
            }
            else if ( mode == 's' )
            {
                if ( strict_max == strict_cur )
                {
                    strict_max = strict_max * 1.2 + 100;
                    strict     = realloc( strict, sizeof( int ) * strict_max );
                }

                strict[ strict_cur ] = strtoimax( line + 1, NULL, 10 );
                strict_cur += 1;
            }
        }
        else
        {
            *sep = '\0';

            if ( mode == '-' )
            {
                if ( multi_ex_cur + 4 >= multi_ex_max )
                {
                    multi_ex_max = multi_ex_max * 1.2 + 100;
                    multi_ex     = realloc( multi_ex, sizeof( int ) * multi_ex_max );
                }

                multi_ex[ multi_ex_cur ]     = strtoimax( line + 1, NULL, 10 );
                multi_ex[ multi_ex_cur + 1 ] = strtoimax( sep + 1, NULL, 10 );
                multi_ex[ multi_ex_cur + 2 ] = multi_ex[ multi_ex_cur + 1 ];
                multi_ex[ multi_ex_cur + 3 ] = multi_ex[ multi_ex_cur ];

                multi_ex_cur += 4;
            }
            else
            {
                if ( multi_inc_cur + 4 >= multi_inc_max )
                {
                    multi_inc_max = multi_inc_max * 1.2 + 100;
                    multi_inc     = realloc( multi_inc, sizeof( int ) * multi_inc_max );
                }

                multi_inc[ multi_inc_cur ]     = strtoimax( line + 1, NULL, 10 );
                multi_inc[ multi_inc_cur + 1 ] = strtoimax( sep + 1, NULL, 10 );
                multi_inc[ multi_inc_cur + 2 ] = multi_inc[ multi_inc_cur + 1 ];
                multi_inc[ multi_inc_cur + 3 ] = multi_inc[ multi_inc_cur ];

                multi_inc_cur += 4;
            }
        }
    }

    qsort( singles, singles_cur, sizeof( int ), cmp_int );
    qsort( strict, strict_cur, sizeof( int ), cmp_int );
    qsort( multi_inc, multi_inc_cur / 2, sizeof( int ) * 2, cmp_2int );
    qsort( multi_ex, multi_ex_cur / 2, sizeof( int ) * 2, cmp_2int );

    rules->exclude_reads   = singles;
    rules->exclude_reads_n = singles_cur;

    rules->strict_reads   = strict;
    rules->strict_reads_n = strict_cur;

    rules->exclude_edges   = multi_ex;
    rules->exclude_edges_n = multi_ex_cur;

    rules->include_edges   = multi_inc;
    rules->include_edges_n = multi_inc_cur;

    int i;
    printf( "exclude reads: " );
    for ( i = 0; i < rules->exclude_reads_n; i++ )
        printf( "%d ", rules->exclude_reads[ i ] );
    printf( "\nstrict reads: " );
    for ( i = 0; i < rules->strict_reads_n; i++ )
        printf( "%d ", rules->strict_reads[ i ] );
    printf( "\ninclude edges: " );
    for ( i = 0; i < rules->include_edges_n; i += 2 )
        printf( "%d-%d ", rules->include_edges[ i ], rules->include_edges[ i + 1 ] );
    printf( "\nexclude edges: " );
    for ( i = 0; i < rules->exclude_edges_n; i += 2 )
        printf( "%d-%d ", rules->exclude_edges[ i ], rules->exclude_edges[ i + 1 ] );
    printf( "\n" );
}

static int loader_handler( void* _ctx, Overlap* ovl, int novl )
{
    FilterContext* ctx = (FilterContext*)_ctx;
    Read_Loader* rl    = ctx->rl;

    int i;
    for ( i = 0; i < novl; i++ )
    {
        int b = ovl[ i ].bread;

        int trim_b_left, trim_b_right;
        get_trim( ctx->db, ctx->trackTrim, b, &trim_b_left, &trim_b_right );

        if ( ovl[ i ].flags & OVL_COMP )
        {
            int tmp      = trim_b_left;
            int blen     = DB_READ_LEN( ctx->db, ovl[ i ].bread );
            trim_b_left  = blen - trim_b_right;
            trim_b_right = blen - tmp;
        }

        if ( trim_b_left >= trim_b_right )
        {
            continue;
        }

        int bbt = MAX( trim_b_left, ovl[ i ].path.bbpos );
        int bet = MIN( trim_b_right, ovl[ i ].path.bepos );

        if ( bbt >= bet )
        {
            continue;
        }

        if ( bbt == ovl[ i ].path.bbpos && bet == ovl[ i ].path.bepos )
        {
            continue;
        }

        bbt = MAX( trim_b_left, ovl[ i ].path.bbpos );
        bet = MIN( trim_b_right, ovl[ i ].path.bepos );

        if ( bbt < bet && ( bbt != ovl[ i ].path.bbpos || bet != ovl[ i ].path.bepos ) )
        {
            rl_add( rl, ovl[ i ].aread );
            rl_add( rl, ovl[ i ].bread );

            continue;
        }

        int bepos = ovl[ i ].path.bepos;

        if ( bepos > bet )
        {
            rl_add( rl, ovl[ i ].aread );
            rl_add( rl, ovl[ i ].bread );
        }
    }

    return 1;
}

static int stitch( Overlap* ovls, int n, int fuzz, int aggressive )
{
    int stitched = 0;

    if ( n < 2 )
    {
        return stitched;
    }

    int i, k, b;
    int ab2, ae1, ae2;
    int bb2, be1, be2;

    const int ignore_mask = OVL_CONT | OVL_STITCH | OVL_GAP | OVL_TRIM;

    for ( i = 0; i < n; i++ )
    {
        Overlap* ovli = ovls + i;

        if ( ovli->flags & ignore_mask )
        {
            continue;
        }

        b = ovli->bread;

        ae1 = ovli->path.aepos;
        be1 = ovli->path.bepos;

        int found = 1;

        while ( found )
        {
            found      = 0;
            int maxk   = 0;
            int maxlen = 0;

            for ( k = i + 1; k < n && ovls[ k ].bread <= b; k++ )
            {
                Overlap* ovlk = ovls + k;

                if ( ovlk->flags & ignore_mask || ( ovli->flags & OVL_COMP ) != ( ovlk->flags & OVL_COMP ) )
                {
                    continue;
                }

                ab2 = ovlk->path.abpos;
                ae2 = ovlk->path.aepos;

                bb2 = ovlk->path.bbpos;
                be2 = ovlk->path.bepos;

                int deltaa = abs( ae1 - ab2 );
                int deltab = abs( be1 - bb2 );

                if ( deltaa < fuzz && deltab < fuzz && ( aggressive || abs( deltaa - deltab ) < 40 ) )
                {
                    if ( ae2 - ab2 > maxlen )
                    {
                        found = 1;

                        maxk   = k;
                        maxlen = ae2 - ab2;
                    }
                }
            }

            if ( found )
            {
                Overlap* ovlk = ovls + maxk;

                ab2 = ovlk->path.abpos;
                ae2 = ovlk->path.aepos;

                bb2 = ovlk->path.bbpos;
                be2 = ovlk->path.bepos;

#ifdef VERBOSE_STITCH
                int ab1 = ovli->path.abpos;
                int bb1 = ovli->path.bbpos;

                printf("STITCH %8d @ %5d..%5d -> %8d @ %5d..%5d %c\n"
                    "                  %5d..%5d -> %8d @ %5d..%5d %c\n",
                    ovli->aread,
                    ab1, ae1, ovli->bread, bb1, be1, OVL_STRAND(ovli),
                    ab2, ae2, ovlk->bread, bb2, be2, OVL_STRAND(ovlk)));
#endif

                ovli->path.aepos = ae2;
                ovli->path.bepos = be2;
                ovli->path.diffs += ovlk->path.diffs;
                ovli->path.tlen = 0;

                ae1 = ae2;
                be1 = be2;

                assert( ovli->bread == ovlk->bread );

                ovli->flags &= ~( OVL_DISCARD | OVL_LOCAL ); // force a re-evaluation of the OVL_LOCAL flags

                ovlk->flags |= OVL_DISCARD | OVL_STITCH;

                stitched += 1;

#ifdef VERBOSE_STITCH
                printf( "    -> %8d @ %5d..%5d -> %8d @ %5d..%5d %c  delta a %3d b %3d\n",
                        ovli->aread,
                        ovli->path.abpos, ovli->path.aepos,
                        ovli->bread,
                        ovli->path.abpos, ovli->path.aepos,
                        OVL_STRAND( ovli ),
                        deltaa, deltab );
#endif
            }
        }
    }

    return stitched;
}

static int find_repeat_modules( FilterContext* ctx, Overlap* ovls, int novl )
{
    int a = ovls->aread;
    int trim_ab, trim_ae;
    const int exclude_mask = OVL_LOCAL | OVL_TRIM | OVL_CONT | OVL_STITCH | OVL_GAP | OVL_DIFF;
    const int allowed_mask = OVL_DISCARD | OVL_REPEAT | OVL_COMP; // | OVL_OLEN | OVL_RLEN;

    if ( ctx->trackTrim )
    {
        get_trim( ctx->db, ctx->trackTrim, a, &trim_ab, &trim_ae );
    }
    else
    {
        trim_ab = 0;
        trim_ae = DB_READ_LEN( ctx->db, a );
    }

    if ( trim_ab >= trim_ae )
    {
        return 1;
    }

    int i;
    uint32_t left  = 0;
    uint32_t right = 0;

    int left_potential  = 0;
    int right_potential = 0;

    // if there are non-repeat overlaps, entering left and leaving right, then no module overlaps are needed

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;
        int abpos    = ovl->path.abpos;
        int aepos    = ovl->path.aepos;
        int flags    = ovl->flags;

        // contained

        if ( abpos == trim_ab && aepos == trim_ae )
        {
            left  = 1;
            right = 1;
            break;
        }

        // potential exits, only allowed to have discard, repeat or comp flag

        if ( ( flags & ( ~allowed_mask ) ) == 0 )
        {
            if ( abpos == trim_ab )
            {
                left_potential += 1;
            }

            if ( aepos == trim_ae )
            {
                right_potential += 1;
            }
        }

        // exits left / right

        if ( flags & OVL_DISCARD )
        {
            continue;
        }

        if ( abpos == trim_ab )
        {
            left += 1;
        }

        if ( aepos == trim_ae )
        {
            right += 1;
        }
    }

    if ( ( left > 0 && right > 0 ) ) // || ( left == 0 && right == 0 ) )
    {
        return 1;
    }

    track_anno* ranno;
    track_data* rdata;

    if ( ctx->trackRepeatStrict )
    {
        ranno = (track_anno*)ctx->trackRepeatStrict->anno;
        rdata = (track_data*)ctx->trackRepeatStrict->data;
    }
    else
    {
        ranno = (track_anno*)ctx->trackRepeat->anno;
        rdata = (track_data*)ctx->trackRepeat->data;
    }

    track_anno ob = ranno[ a ];
    track_anno oe = ranno[ a + 1 ];

    // no repeats, nothing to do

    if ( ob >= oe )
    {
        return 1;
    }

    if ( oe - ob > ctx->rm_maxrepeat ) // bytes
    {
        ctx->rm_maxrepeat = ( oe - ob ) + 128;
        ctx->rm_repeat    = malloc( ctx->rm_maxrepeat );
    }

    // merge repeats close to each other

    int nrepeat = 0;

    ob /= sizeof( track_data );
    oe /= sizeof( track_data );

    while ( ob < oe )
    {
        int b = MAX( trim_ab, rdata[ ob ] );
        int e = MIN( trim_ae, rdata[ ob + 1 ] );
        ob += 2;

        if ( b >= e )
        {
            continue;
        }

        if ( nrepeat > 0 && b - ctx->rm_repeat[ nrepeat - 1 ] < ctx->rm_merge )
        {
            // ctx->stats_merged++;
            ctx->rm_repeat[ nrepeat - 1 ] = e;
        }
        else
        {
            ctx->rm_repeat[ nrepeat++ ] = b;
            ctx->rm_repeat[ nrepeat++ ] = e;
        }
    }

    // for each segment count number of reads anchored with at least MIN_LR

    bzero( ctx->rm_bins, sizeof( uint64_t ) * ctx->rm_maxbins );

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;

        if ( ovl->flags & ( OVL_STITCH | OVL_TRIM ) )
        {
            continue;
        }

        int b = ( ovl->path.abpos + MIN_LR ) / BIN_SIZE;
        int e = ( ovl->path.aepos - MIN_LR ) / BIN_SIZE;

        // spanning local alignments indicate non-reliable points inside the repeat

        int incr = 1;

        if ( ctx->rm_aggressive == 0 && ( ovl->flags & OVL_LOCAL ) )
        {
            incr = ctx->rm_cov;
        }

        while ( b < e )
        {
            ctx->rm_bins[ b ] += incr;
            b++;
        }
    }

    track_anno prev_offset = ctx->rm_ndata;

    for ( i = 0; i < nrepeat; i += 2 )
    {
        int rb = ctx->rm_repeat[ i ];
        int re = ctx->rm_repeat[ i + 1 ];

        // skip repeats if there are valid spanning overlaps

        int skip = 0;
        int j;

        for ( j = 0; j < novl; j++ )
        {
            Overlap* ovl = ovls + j;

            if ( ( ovl->flags & OVL_DISCARD ) )
            {
                continue;
            }

            if ( ovl->path.abpos + MIN_LR <= rb && ovl->path.aepos - MIN_LR >= re )
            {
                skip = 1;
                break;
            }
        }

        if ( skip )
        {
            continue;
        }

        // gaps inside the repeat with < expected coverage are potential repeat modules

        int b = MAX( trim_ab + 1000, rb + MIN_LR ) / BIN_SIZE;
        int e = MIN( trim_ae - 1000, re - MIN_LR ) / BIN_SIZE;

        int beg = -1;

        while ( b < e )
        {
            if ( ctx->rm_bins[ b ] > 0 && ctx->rm_bins[ b ] < (uint64_t)ctx->rm_cov )
            {
                //printf("READ %7d POINT @ %5d..%5d %3d %2llu\n", a, b * BIN_SIZE - 50, b * BIN_SIZE + 50, b, ctx->rm_bins[b]);

                if ( beg == -1 )
                {
                    beg = b;
                }
            }
            else
            {
                if ( beg != -1 )
                {
                    if ( b - beg > 7 )
                    {
                        //printf("MOD  %7d POINT @ %5d %5d %5d..%5d\n", a, (b + beg) / 2, (b + beg) * BIN_SIZE / 2, beg, b);

                        if ( ctx->rm_ndata + 2 > ctx->rm_maxdata )
                        {
                            ctx->rm_maxdata = 1.2 * ctx->rm_ndata + 100;
                            ctx->rm_data    = realloc( ctx->rm_data, ctx->rm_maxdata * sizeof( track_data ) );
                        }

                        int intb = ( b + beg ) / 2 * BIN_SIZE - 50;
                        int inte = intb + 100;

                        ctx->rm_anno[ a ] += 2 * sizeof( track_data );

                        ctx->rm_data[ ctx->rm_ndata++ ] = intb;
                        ctx->rm_data[ ctx->rm_ndata++ ] = inte;
                    }

                    beg = -1;
                }
            }

            b++;
        }
    }

    // restore discarded overlaps spanning the repeat module junction

    uint32_t enabled = 0;

    while ( prev_offset < ctx->rm_ndata )
    {
        int b = ctx->rm_data[ prev_offset++ ];
        int e = ctx->rm_data[ prev_offset++ ];

        // ignore module that would result in excessive coverage at the junction

        //printf("MODULE @ %d..%d\n", b, e);
        int cov = 0;

        for ( i = 0; i < novl; i++ )
        {
            Overlap* ovl = ovls + i;

            if ( ovl->path.abpos + 100 < b && ovl->path.aepos - 100 > e )
            {
                if ( !( ovl->flags & OVL_DISCARD ) ||
                     ( ( ovl->flags & OVL_REPEAT ) && !( ovl->flags & exclude_mask ) ) )
                {
                    cov++;
                }
            }
        }

        if ( cov > ctx->rm_cov || cov < ctx->rm_cov / 2 )
        {
            continue;
        }

        for ( i = 0; i < novl; i++ )
        {
            Overlap* ovl = ovls + i;

            if ( !( ovl->flags & OVL_REPEAT ) || ( ovl->flags & exclude_mask ) )
            {
                continue;
            }

            // MARTIN ignore overlaps that do not have an overhang
            if ( ovl->path.abpos > trim_ab && ovl->path.aepos < trim_ae )
            {
                continue;
            }

            if ( ovl->path.abpos + MIN_LR < b && ovl->path.aepos - MIN_LR > e )
            {
                ovl->flags &= ~OVL_DISCARD;
                ovl->flags |= OVL_MODULE;

                ctx->nRepeatOvlsKept++;
                enabled += 1;
            }
        }
    }

    if ( enabled )
    {
        return 1;
    }

    printf( "%d | left = %d right %d lp %d rp %d\n", a, left, right, left_potential, right_potential );

    /*

    // try to relax the min non-repeat bases condition

    ob = ranno[a] / sizeof(track_data);
    oe = ranno[a + 1] / sizeof(track_data);

    int dist_left = INT_MAX;
    int dist_right = INT_MAX;

    while ( ob < oe )
    {
        int rb = MAX(trim_ab, rdata[ob]);
        int re = MIN(trim_ae, rdata[ob + 1]);
        ob += 2;

        if (rb >= re)
        {
            continue;
        }

        dist_left = MIN( rb - trim_ab, dist_left );
        dist_right = MIN( trim_ae - re, dist_right );
    }

    printf("%d | min dist left %d right %d\n", a, dist_left, dist_right);

    if ( (left == 0 && left_potential <= ctx->rm_cov && dist_left > 100 && dist_left != INT_MAX) ||
            (right == 0 && right_potential <= ctx->rm_cov && dist_right > 100 && dist_right != INT_MAX) )
    {
        for ( i = 0 ; i < novl ; i++ )
        {
            Overlap* ovl = ovls + i;

            if ( (left == 0 && ovl->path.abpos == trim_ab && !(ovl->flags & (~allowed_mask)) ) ||
                (right == 0 && ovl->path.aepos == trim_ae && !(ovl->flags & (~allowed_mask)) ) )
            {
                ovl->flags &= ~OVL_DISCARD;
                ovl->flags |= OVL_MODULE;
                ctx->nRepeatOvlsKept++;
                enabled += 1;
            }
        }
    }

    printf("%d | enabled %d\n", a, enabled);

    if (enabled)
    {
        return 1;
    }
    */

    //
    // TODO --- coverage ... don't count multiple overlaps with the same b ????
    //

    if ( ctx->rm_mode < 2 )
    {
        return 1;
    }

    // look at b reads and see if they would lead is to a unique region

    if ( novl > ctx->max_r2bin )
    {
        ctx->max_r2bin = novl * 1.2 + 128;
        ctx->r2bin     = realloc( ctx->r2bin, sizeof( int ) * ctx->max_r2bin );
    }

    int potential = 0;

    bzero( ctx->r2bin, sizeof( int ) * ctx->max_r2bin );
    bzero( ctx->rm_bins, sizeof( uint64_t ) * ctx->rm_maxbins );

    int binsize = MAX( BIN_SIZE, 1000 );

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;
        int trim_bb, trim_be;
        int b = ovl->bread;

        if ( ( ovl->flags & exclude_mask ) ||
             ( left == 0 && ovl->path.abpos != trim_ab ) ||
             ( right == 0 && ovl->path.aepos != trim_ae ) )
        {
            continue;
        }

        get_trim( ctx->db, ctx->trackTrim, b, &trim_bb, &trim_be );

        printf( "%7d | btrim  %5d..%5d\n", b, trim_bb, trim_be );

        int bb, be;

        if ( ovl->flags & OVL_COMP )
        {
            int blen = DB_READ_LEN( ctx->db, b );
            bb       = blen - ovl->path.bepos;
            be       = blen - ovl->path.bbpos;
        }
        else
        {
            bb = ovl->path.bbpos;
            be = ovl->path.bepos;
        }

        ob = ranno[ b ] / sizeof( track_data );
        oe = ranno[ b + 1 ] / sizeof( track_data );
        int rb, re;

        int nrb       = 0;
        int prev_end  = trim_bb;
        int rlen_in_b = 0;

        while ( ob < oe )
        {
            rb = MAX( trim_bb, rdata[ ob ] );
            re = MIN( trim_be, rdata[ ob + 1 ] );
            ob += 2;

            printf( "%d | %7d | b repeat %5d..%5d\n", a, b, rb, re );

            if ( rb >= re )
            {
                continue;
            }

            if ( left == 0 )
            {
                if ( ovl->flags & OVL_COMP )
                {
                    if ( rb < be && be < re )
                    {
                        rlen_in_b = re - be;
                    }
                }
                else
                {
                    if ( rb < bb && bb < re )
                    {
                        rlen_in_b = bb - rb;
                    }
                }
            }
            else
            {
                if ( ovl->flags & OVL_COMP )
                {
                    if ( rb < bb && bb < re )
                    {
                        rlen_in_b = bb - rb;
                    }
                }
                else
                {
                    if ( rb < be && be < re )
                    {
                        rlen_in_b = re - be;
                    }
                }
            }

            nrb += rb - prev_end;
            prev_end = re;
        }

        nrb += trim_be - prev_end;

        if ( nrb < ctx->nMinNonRepeatBases || rlen_in_b == 0 )
        {
            continue;
        }

        printf( "%d -> %7d | leftover rlen %5d nrb %5d", a, b, rlen_in_b, nrb );

        if ( !( ovl->flags & ( ~allowed_mask ) ) )
        {
            ovl->flags |= OVL_TEMP;
            potential += 1;

            printf( "  YES" );

            int bin = rlen_in_b / binsize;

            ctx->rm_bins[ bin ] += 1;
            ctx->r2bin[ i ] = bin;
        }

        printf( "\n" );
    }

    if ( potential > 0 )
    {
        for ( i = 0; i < novl; i++ )
        {
            Overlap* ovl = ovls + i;

            if ( ovl->flags & OVL_TEMP )
            {
                int bin = ctx->r2bin[ i ];
                ovl->flags &= ~OVL_TEMP;

                if ( ctx->rm_bins[ bin ] > 2 && ctx->rm_bins[ bin ] < (uint32_t)ctx->rm_cov )
                {
                    ovl->flags &= ~OVL_DISCARD;
                    ovl->flags |= OVL_OPTIONAL;
                    ctx->nRepeatOvlsKept++;
                    enabled += 1;
                }
            }
        }

        if ( enabled )
        {
            return 1;
        }
    }

    if ( ctx->rm_mode < 3 )
    {
        return 1;
    }

    int prevb     = -1;
    int distinctb = 1;

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;
        int b        = ovl->bread;

        if ( ( ovl->flags & exclude_mask ) ||
             ( left == 0 && ovl->path.abpos != trim_ab ) ||
             ( right == 0 && ovl->path.aepos != trim_ae ) )
        {
            continue;
        }

        if ( prevb != b )
        {
            distinctb += 1;
            prevb = b;
        }
    }

    if ( distinctb < ctx->rm_cov )
    {
        prevb       = -1;
        int maxbidx = -1;
        int maxblen = -1;

        for ( i = 0; i < novl; i++ )
        {
            Overlap* ovl = ovls + i;
            int b        = ovl->bread;
            int len      = ovl->path.aepos - ovl->path.abpos;

            if ( ( ovl->flags & exclude_mask ) ||
                 ( left == 0 && ovl->path.abpos != trim_ab ) ||
                 ( right == 0 && ovl->path.aepos != trim_ae ) )
            {
                continue;
            }

            if ( prevb != b )
            {
                if ( maxbidx != -1 )
                {
                    ovls[ maxbidx ].flags &= ~OVL_DISCARD;
                    ovls[ maxbidx ].flags |= OVL_OPTIONAL;
                    ctx->nRepeatOvlsKept++;
                    enabled += 1;
                }

                prevb   = b;
                maxbidx = i;
                maxblen = len;
            }
            else
            {
                if ( len > maxblen )
                {
                    maxbidx = i;
                    maxblen = len;
                }
            }
        }

        if ( maxbidx != -1 )
        {
            ovls[ maxbidx ].flags &= ~OVL_DISCARD;
            ovls[ maxbidx ].flags |= OVL_OPTIONAL;
            ctx->nRepeatOvlsKept++;
            enabled += 1;
        }
    }

    return 1;
}

static int filter( FilterContext* ctx, Overlap* ovl )
{
    int nLen         = ovl->path.aepos - ovl->path.abpos;
    int nLenB        = ovl->path.bepos - ovl->path.bbpos;
    int ret          = 0;
    HITS_READ* reads = ctx->db->reads;
    int bread        = ovl->bread;

    if ( nLenB < nLen )
    {
        nLen = nLenB;
    }

    int trim_ab, trim_ae, trim_bb, trim_be;
    int trim_alen, trim_blen;

    int ovlALen = DB_READ_LEN( ctx->db, ovl->aread );
    int ovlBLen = DB_READ_LEN( ctx->db, bread );

    if ( ctx->trackTrim )
    {
        get_trim( ctx->db, ctx->trackTrim, ovl->aread, &trim_ab, &trim_ae );
        trim_alen = trim_ae - trim_ab;

        get_trim( ctx->db, ctx->trackTrim, bread, &trim_bb, &trim_be );
        trim_blen = trim_be - trim_bb;

        if ( ovl->flags & OVL_COMP )
        {
            int t   = trim_bb;
            trim_bb = ovlBLen - trim_be;
            trim_be = ovlBLen - t;
        }
    }
    else
    {
        trim_ab = 0;
        trim_ae = ovlALen;

        trim_bb = 0;
        trim_be = ovlBLen;

        trim_alen = ovlALen;
        trim_blen = ovlBLen;
    }

    if ( ctx->nMinReadLength != -1 && ( trim_alen < ctx->nMinReadLength || trim_blen < ctx->nMinReadLength ) )
    {
        ctx->nFilteredReadLength++;
        ret |= OVL_DISCARD | OVL_RLEN;
    }

    if ( ctx->nMinAlnLength != -1 && nLen < ctx->nMinAlnLength )
    {
        if ( ctx->nVerbose )
        {
            printf( "overlap %d -> %d: drop due to length %d\n", ovl->aread, bread, nLen );
        }

        ctx->nFilteredLength++;
        ret |= OVL_DISCARD | OVL_OLEN;
    }

    if ( ctx->nMaxUnalignedBases != -1 )
    {
        if ( ( ( ovl->path.abpos - trim_ab ) > ctx->nMaxUnalignedBases &&
               ( ovl->path.bbpos - trim_bb ) > ctx->nMaxUnalignedBases ) ||
             ( ( trim_ae - ovl->path.aepos ) > ctx->nMaxUnalignedBases &&
               ( trim_be - ovl->path.bepos ) > ctx->nMaxUnalignedBases ) )
        {
            if ( ctx->nVerbose )
            {
                printf( "overlap %d -> %d: drop due to unaligned overhang [%d, %d -> trim %d, %d], [%d, %d -> trim %d, %d]\n", ovl->aread, ovl->bread, ovl->path.abpos, ovl->path.aepos, trim_ab, trim_ae, ovl->path.bbpos, ovl->path.bepos, trim_bb, trim_be );
            }

            ctx->nFilteredUnalignedBases++;

            ret |= OVL_DISCARD | OVL_LOCAL;
        }
    }

    if ( ctx->fMaxDiffs > 0 )
    {
        if ( 1.0 * ovl->path.diffs / nLen > ctx->fMaxDiffs )
        {
            if ( ctx->nVerbose )
            {
                printf( "overlap %d -> %d: drop due to diffs %d length %d\n", ovl->aread, bread, ovl->path.diffs, nLen );
            }

            ctx->nFilteredDiffs++;

            ret |= OVL_DISCARD | OVL_DIFF;
        }
    }

    if ( ctx->nMinNonRepeatBases != -1 )
    {
        int b, e, rb, re, ovllen, repeat, repeat_read;

        track_anno* repeats_anno;
        track_data* repeats_data;

        if ( ctx->trackRepeatStrict && ( ( reads[ ovl->aread ].flags & READ_STRICT ) || ( reads[ bread ].flags & READ_STRICT ) ) )
        {
            repeats_anno = ctx->trackRepeatStrict->anno;
            repeats_data = ctx->trackRepeatStrict->data;
        }
        else
        {
            repeats_anno = ctx->trackRepeat->anno;
            repeats_data = ctx->trackRepeat->data;
        }

        b      = repeats_anno[ ovl->aread ] / sizeof( track_data );
        e      = repeats_anno[ ovl->aread + 1 ] / sizeof( track_data );
        ovllen = ovl->path.aepos - ovl->path.abpos;
        repeat = repeat_read = 0;

        while ( b < e )
        {
            rb = repeats_data[ b ];
            re = repeats_data[ b + 1 ];

            repeat_read += ( re - rb );
            repeat += intersect( ovl->path.abpos, ovl->path.aepos, rb, re );

            b += 2;
        }

        if ( repeat > 0 && ovllen - repeat < ctx->nMinNonRepeatBases )
        {
            if ( ctx->nVerbose )
            {
                printf( "overlap %d -> %d: drop due to repeat in a\n", ovl->aread, bread );
            }

            ctx->nFilteredRepeat++;
            ret |= OVL_DISCARD | OVL_REPEAT;
        }

        b      = repeats_anno[ bread ] / sizeof( track_data );
        e      = repeats_anno[ bread + 1 ] / sizeof( track_data );
        ovllen = ovl->path.bepos - ovl->path.bbpos;

        int bbpos, bepos;

        if ( ovl->flags & OVL_COMP )
        {
            bbpos = ovlBLen - ovl->path.bepos;
            bepos = ovlBLen - ovl->path.bbpos;
        }
        else
        {
            bbpos = ovl->path.bbpos;
            bepos = ovl->path.bepos;
        }

        repeat = repeat_read = 0;

        while ( b < e )
        {
            rb = repeats_data[ b ];
            re = repeats_data[ b + 1 ];

            repeat += intersect( bbpos, bepos, rb, re );
            repeat_read += ( re - rb );

            b += 2;
        }

        if ( repeat > 0 && ovllen - repeat < ctx->nMinNonRepeatBases )
        {
            if ( ctx->nVerbose )
            {
                printf( "overlap %d -> %d: drop due to repeat in b\n", ovl->aread, bread );
            }

            ctx->nFilteredRepeat++;

            ret |= OVL_DISCARD | OVL_REPEAT;
        }
    }

    // check tracepoints
    if ( !( ovl->flags & OVL_DISCARD ) )
    {
        ovl_trace* trace = (ovl_trace*)ovl->path.trace;

        int bpos = ovl->path.bbpos;

        int j;

        for ( j = 0; j < ovl->path.tlen; j += 2 )
        {
            bpos += trace[ j + 1 ];
        }

        if ( bpos != ovl->path.bepos )
        {
            ret |= OVL_DISCARD;

            if ( ctx->nVerbose )
            {
                printf( "overlap (%d x %d): pass-through points inconsistent be = %d (expected %d)\n", ovl->aread, ovl->bread, bpos, ovl->path.bepos );
            }
        }
    }

    return ret;
}

static void filter_pre( PassContext* pctx, FilterContext* fctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS filtering\n" ANSI_COLOR_RESET );
#endif

    fctx->twidth = pctx->twidth;

    // trim

    if ( fctx->do_trim )
    {
        fctx->trim = trim_init( fctx->db, pctx->twidth, fctx->trackTrim, fctx->rl );
    }

    // repeat modules

    fctx->rm_anno = (track_anno*)malloc( sizeof( track_anno ) * ( DB_NREADS( fctx->db ) + 1 ) );
    bzero( fctx->rm_anno, sizeof( track_anno ) * ( DB_NREADS( fctx->db ) + 1 ) );

    fctx->rm_ndata   = 0;
    fctx->rm_maxdata = 100;
    fctx->rm_data    = (track_data*)malloc( sizeof( track_data ) * fctx->rm_maxdata );

    fctx->rm_maxbins = ( DB_READ_MAXLEN( fctx->db ) + BIN_SIZE ) / BIN_SIZE;
    fctx->rm_bins    = malloc( sizeof( uint64_t ) * fctx->rm_maxbins );

    fctx->le_maxbins = ( DB_READ_MAXLEN( fctx->db ) + BIN_SIZE ) / BIN_SIZE;
    fctx->le_lbins   = malloc( sizeof( int ) * fctx->le_maxbins );
    fctx->le_rbins   = malloc( sizeof( int ) * fctx->le_maxbins );
}

static void filter_post( FilterContext* ctx )
{
#ifdef VERBOSE
    if ( ctx->trim )
    {
        printf( "trimmed %'lld of %'lld overlaps\n", ctx->trim->nTrimmedOvls, ctx->trim->nOvls );
        printf( "trimmed %'lld of %'lld bases\n", ctx->trim->nTrimmedBases, ctx->trim->nOvlBases );
    }

    if ( ctx->nFilteredReadLength > 0 )
    {
        printf( "min read length of %d discarded %d\n", ctx->nMinReadLength, ctx->nFilteredReadLength );
    }

    if ( ctx->nFilteredRepeat > 0 )
    {
        printf( "min non-repeat bases of %d discarded %d\n", ctx->nMinNonRepeatBases, ctx->nFilteredRepeat );
    }

    if ( ctx->nRepeatOvlsKept > 0 )
    {
        printf( "  kept %d repeat overlaps (mode %d)\n", ctx->nRepeatOvlsKept, ctx->rm_mode );
    }

    if ( ctx->nFilteredDiffs > 0 )
    {
        printf( "diff threshold of %.1f discarded %d\n", ctx->fMaxDiffs * 100., ctx->nFilteredDiffs );
    }

    if ( ctx->nFilteredDiffsSegments > 0 )
    {
        printf( "diff threshold of %.1f on segments discarded %d\n", ctx->fMaxDiffs * 100., ctx->nFilteredDiffsSegments );
    }

    if ( ctx->nFilteredLength > 0 )
    {
        printf( "min overlap length of %d discarded %d\n", ctx->nMinAlnLength, ctx->nFilteredLength );
    }

    if ( ctx->nFilteredUnalignedBases > 0 )
    {
        printf( "unaligned bases threshold of %d discarded %d\n", ctx->nMaxUnalignedBases, ctx->nFilteredUnalignedBases );
    }

    if ( ctx->nStitched > 0 )
    {
        printf( "stitched %d at fuzzing of %d\n", ctx->nStitched, ctx->stitch );
    }

    if ( ctx->nFilteredLocalEnd )
    {
        printf( "local ends discarded %d\n", ctx->nFilteredLocalEnd );
    }

#endif

    if ( ctx->trim )
    {
        trim_close( ctx->trim );
    }

    free( ctx->le_lbins );
    free( ctx->le_rbins );

    free( ctx->rm_bins );
    free( ctx->rm_anno );
    free( ctx->rm_data );
}


static int filter_handler( void* _ctx, Overlap* ovl, int novl )
{
    FilterContext* ctx = (FilterContext*)_ctx;
    int j;
    int aread  = ovl->aread;
    int alen = DB_READ_LEN(ctx->db, aread);
    HITS_DB* db = ctx->db;

    /*
    if ( ovl->aread > 100 )
    {
        return 0;
    }
    */

    if ( ctx->trim )
    {
        for ( j = 0; j < novl; j++ )
        {
            trim_overlap( ctx->trim, ovl + j );
        }
    }

    if ( ctx->stitch >= 0 )
    {
        int k;
        j = k = 0;

        while ( j < novl )
        {
            while ( k < novl - 1 && ovl[ j ].bread == ovl[ k + 1 ].bread )
            {
                k++;
            }

            ctx->nStitched += stitch( ovl + j, k - j + 1, ctx->stitch, ctx->stitch_aggressively );

            j = k + 1;
        }
    }

    if ( ctx->contained )
    {
        int contained = 0;
        for ( j = 0; j < novl; j++)
        {
            Overlap* o = ovl + j;

            if ( o->path.abpos == 0 && o->path.aepos == alen )
            {
                contained = 1;
                break;
            }
            else if ( o->path.bbpos == 0 && o->path.bepos == DB_READ_LEN(db, o->bread) )
            {
                o->flags |= OVL_DISCARD;
            }
        }

        if (contained)
        {
            for ( j = 0; j < novl; j++)
            {
                Overlap* o = ovl + j;
                o->flags |= OVL_DISCARD;
            }

            return 1;
        }
    }

    if ( ctx->hrd )
    {
        track_anno* repeats_anno = ctx->trackRepeat->anno;
        track_data* repeats_data = ctx->trackRepeat->data;

        int b = repeats_anno[ ovl->aread ] / sizeof( track_data );
        int e = repeats_anno[ ovl->aread + 1 ] / sizeof( track_data );

        Overlap*** groups = ctx->hrd_groups;
        int* ngroups = ctx->hrd_ngroups;
        int* maxgroups = ctx->hrd_maxgroups;
        uint16_t* mapgroup = ctx->hrd_mapgroup;
        int allocated = ctx->hrd_allocated;

        if ( allocated < ( e - b ) / 2 + 1 )
        {
            int prev = allocated;
            allocated = ctx->hrd_allocated = ( e - b ) / 2 + 1 + 100;
            groups = ctx->hrd_groups = realloc(groups, allocated * sizeof(Overlap**) );
            ngroups = ctx->hrd_ngroups = realloc(ngroups, allocated * sizeof(int) );
            maxgroups = ctx->hrd_maxgroups = realloc(maxgroups, allocated * sizeof(int) );

            bzero( groups + prev, (allocated - prev) * sizeof(Overlap**) );
            bzero( ngroups + prev, (allocated - prev) * sizeof(int) );
            bzero( maxgroups + prev, (allocated - prev) * sizeof(int) );
        }

        bzero(mapgroup, DB_READ_LEN(ctx->db, aread) * sizeof(uint16_t) );

        int curgroup = 1;

        while ( b < e )
        {
            int rb = repeats_data[ b ];
            int re = repeats_data[ b + 1 ] + 1;

            while ( rb < re )
            {
                mapgroup[ rb ] = curgroup;
                rb += 1;
            }

            ngroups[curgroup] = 0;

            curgroup += 1;
            b += 2;
        }

        uint16_t ngroup = mapgroup[ ctx->hrd_fuzz ];

        if ( ngroup != 0 && mapgroup[ DB_READ_LEN( ctx->db, aread ) - 1 - ctx->hrd_fuzz ] == ngroup )
        {
            for ( j = 0; j < novl; j++ )
            {
                Overlap* o = ovl + j;
                o->flags |= OVL_DISCARD | OVL_REPEAT;
            }
        }
        else
        {
            for ( j = 0; j < novl; j++ )
            {
                Overlap* o = ovl + j;
                ngroup     = mapgroup[ o->path.abpos ];

                if ( ngroup == 0 || mapgroup[ o->path.aepos ] != ngroup )
                {
                    continue;
                }

                if ( o->path.aepos - o->path.abpos < ctx->hrd_len )
                {
                    o->flags |= OVL_DISCARD | OVL_REPEAT;
                    continue;
                }

                if ( ngroups[ ngroup ] + 1 >= maxgroups[ ngroup ] )
                {
                    maxgroups[ ngroup ] = 1.2 * maxgroups[ ngroup ] + 1000;
                    groups[ ngroup ]    = realloc( groups[ ngroup ], maxgroups[ ngroup ] * sizeof( Overlap** ) );
                }

                groups[ ngroup ][ ngroups[ ngroup ] ] = o;
                ngroups[ ngroup ] += 1;
            }

            for ( j = 1; j < curgroup; j++ )
            {
                if ( ngroups[ j ] < 1000 )
                {
                    continue;
                }

                qsort( groups[ j ], ngroups[ j ], sizeof( Overlap* ), cmp_ovl_q_desc );

                // printf("%d group %d -> %d\n", aread, j, ngroups[j]);

                int k;
                int discard = ngroups[ j ] * (ctx->hrd_rate / 100.0);
                for ( k = 0; k < discard; k++ )
                {
                    Overlap* o = groups[ j ][ k ];
                    o->flags |= OVL_DISCARD | OVL_REPEAT;

                    // int q = 100.0 * o->path.diffs / (o->path.aepos - o->path.abpos);
                    // printf("%d..%d %d %d\n", groups[j][k]->path.abpos, groups[j][k]->path.aepos, o->path.diffs, q);
                }
            }

        }

    }

    // set filter flags

    for ( j = 0; j < novl; j++ )
    {
        ovl[ j ].flags |= filter( ctx, ovl + j );
    }

    // find repeat modules and rescue overlaps

    if ( ctx->rm_cov != -1 )
    {
        find_repeat_modules( ctx, ovl, novl );
    }

    // filter by read flags
    HITS_READ* reads = ctx->db->reads;

    if ( reads[ aread ].flags & READ_DISCARD )
    {
        for ( j = 0; j < novl; j++ )
        {
            ovl[ j ].flags |= OVL_DISCARD;
        }
    }
    else
    {
        for ( j = 0; j < novl; j++ )
        {
            int bread = ovl[ j ].bread;

            if ( reads[ bread ].flags & READ_DISCARD )
            {
                ovl[ j ].flags |= OVL_DISCARD;
            }
        }
    }

    return 1;
}

static void usage( FILE* fout, const char* app )
{
    fprintf( fout, "usage: %s [-cLpTv] [-dlmMnosSu n] [-rRt track] [-x file] [-N n,n,n] database input.las output.las\n\n", app );

    fprintf( fout, "Filters the input las file by various critera\n\n" );

    fprintf( fout, "options: -v  verbose output\n" );
    fprintf( fout, "         -c  drop contained reads\n");
    fprintf( fout, "         -d n  max divergence allowed [0,100]\n" );
    fprintf( fout, "         -l n  minimum read length\n" );
    fprintf( fout, "         -L  two-pass processing with read caching\n\n" );
    fprintf( fout, "         -n n  minimum number of non-repeat-annotated bases in an alignment\n" );
    fprintf( fout, "         -o n  minimum alignment length\n" );
    fprintf( fout, "         -p purge discarded alignments\n" );
    fprintf( fout, "         -r track  name of the track containing the repeat annotation (%s)\n", DEF_ARG_R );
    fprintf( fout, "         -s n  conservatively stitch split alignments with distance < n\n" );
    fprintf( fout, "         -S n  stitch irrespective of discrepencies in the distance between A and B read\n" );
    fprintf( fout, "         -t track  name of the track containing the trim annotation (%s)\n", DEF_ARG_T );
    fprintf( fout, "         -T  apply the trim annotation and update the alignments\n" );
    fprintf( fout, "         -u  maximum number of unaligned (leftover) bases in the alignment\n" );

    fprintf( fout, "experimental options:\n" );
    fprintf( fout, "         -N n,n,n  heuristic repeat-induced alignment droppping. TODO: explain\n" );

    fprintf( fout, "         -m n  resolve repeat modules, pass coverage as argument.\n" );
    fprintf( fout, "         -M n  -m + more aggressive module detection\n" );
    fprintf( fout, "         -x file  exclude read ids found in file\n" );
    fprintf( fout, "         -R track  strict repeat track name\n\n" );

    fprintf( fout, "Up to three m's can be used (-mmm n) with each additional m increasing the aggressiveness of the repeat resolution.\n" );
}

static int opt_repeat_count( int argc, char** argv, char opt )
{
    int i;
    int count = 0;
    for ( i = 1; i < argc; i++ )
    {
        char* arg = argv[ i ];

        if ( *arg == '-' )
        {
            arg += 1;

            while ( *arg == opt )
            {
                count += 1;
                arg += 1;
            }

            if ( count )
            {
                argv[ i ][ 2 ] = '\0';
                break;
            }
        }
    }

    return count;
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    FilterContext fctx;
    PassContext* pctx;
    FILE* fileOvlIn;
    FILE* fileOvlOut;
    char* app = argv[ 0 ];

    bzero( &fctx, sizeof( FilterContext ) );

    fctx.db = &db;

    // args

    char* pathRules            = NULL;
    char* pcTrackRepeats       = DEF_ARG_R;
    char* pcTrackRepeatsStrict = NULL;
    char* arg_trimTrack        = DEF_ARG_T;
    int arg_purge              = 0;

    fctx.trackRepeatStrict   = NULL;
    fctx.fMaxDiffs           = -1;
    fctx.nMaxUnalignedBases  = -1;
    fctx.nMinAlnLength       = -1;
    fctx.nMinNonRepeatBases  = -1;
    fctx.nMinReadLength      = -1;
    fctx.nVerbose            = 0;
    fctx.stitch              = DEF_ARG_S;
    fctx.rm_cov              = -1;
    fctx.rm_aggressive       = 0;
    fctx.useRLoader          = 0;
    fctx.do_trim             = DEF_ARG_TT;
    fctx.rm_mode             = 0;
    fctx.stitch_aggressively = 0;
    fctx.contained = 0;
    fctx.hrd                = 0; // heuristic repeat dropping
    fctx.hrd_len            = DEF_ARG_NN_LEN;
    fctx.hrd_rate           = DEF_ARG_NN_RATE;
    fctx.hrd_fuzz           = DEF_ARG_NN_FUZZ;

    fctx.rm_merge = 50;

    int c;

    fctx.rm_mode = opt_repeat_count( argc, argv, 'm' );
    if ( fctx.rm_mode == 0 )
    {
        fctx.rm_mode = opt_repeat_count( argc, argv, 'M' );
    }

    opterr = 0;
    while ( ( c = getopt( argc, argv, "cLpTvd:l:m:M:n:N:o:r:R:s:S:t:u:x:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'c':
                fctx.contained = 1;
                break;

            case 'N':
                fctx.hrd = 1;

                char* dup = strdup(optarg);
                char* token;
                int count = 0;
                while ( ( token = strsep(&dup, ",") ) )
                {
                    size_t toklen = strlen(token);

                    switch (count)
                    {
                        case 0:
                            if (toklen == 1 && token[0] == '-')
                            {
                                fctx.hrd_len = DEF_ARG_NN_LEN;
                            }
                            else
                            {
                                fctx.hrd_len = atoi(token);
                            }
                            break;

                        case 1:
                            if (toklen == 1 && token[0] == '-')
                            {
                                fctx.hrd_rate = DEF_ARG_NN_RATE;
                            }
                            else
                            {
                                fctx.hrd_rate = atoi(token);
                            }
                            break;

                        case 2:
                            if (toklen == 1 && token[0] == '-')
                            {
                                fctx.hrd_fuzz = DEF_ARG_NN_FUZZ;
                            }
                            else
                            {
                                fctx.hrd_fuzz = atoi(token);
                            }
                            break;
                    }

                    count += 1;
                }

                free(dup);

                if ( count != 3 )
                {
                    usage(stdout, app);
                    exit(1);
                }

                break;

            case 'T':
                fctx.do_trim = 1;
                break;

            case 'x':
                pathRules = optarg;
                break;

            case 'L':
                fctx.useRLoader = 1;
                break;

            case 'M':
                fctx.rm_aggressive = 1;

                // fall through

            case 'm':
                fctx.rm_cov = atoi( optarg );
                break;

            case 'S':
                fctx.stitch_aggressively = 1;

                // fall through

            case 's':
                fctx.stitch = atoi( optarg );
                break;

            case 'v':
                fctx.nVerbose = 1;
                break;

            case 'p':
                arg_purge = 1;
                break;

            case 'd':
                fctx.fMaxDiffs = atof( optarg ) / 100.0;
                break;

            case 'o':
                fctx.nMinAlnLength = atoi( optarg );
                break;

            case 'l':
                fctx.nMinReadLength = atoi( optarg );
                break;

            case 'u':
                fctx.nMaxUnalignedBases = atoi( optarg );
                break;

            case 'n':
                fctx.nMinNonRepeatBases = atoi( optarg );
                fctx.rm_merge           = fctx.nMinNonRepeatBases;
                break;

            case 'r':
                pcTrackRepeats = optarg;
                break;

            case 'R':
                pcTrackRepeatsStrict = optarg;
                break;

            case 't':
                arg_trimTrack = optarg;
                break;

            default:
                fprintf( stderr, "unknown option %c\n", c );
                usage( stdout, app );
                exit( 1 );
        }
    }

    if ( argc - optind != 3 )
    {
        usage( stdout, app );
        exit( 1 );
    }

    char* pcPathReadsIn     = argv[ optind++ ];
    char* pcPathOverlapsIn  = argv[ optind++ ];
    char* pcPathOverlapsOut = argv[ optind++ ];

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathOverlapsIn );
        exit( 1 );
    }

    if ( ( fileOvlOut = fopen( pcPathOverlapsOut, "w" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathOverlapsOut );
        exit( 1 );
    }

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "could not open %s\n", pcPathReadsIn );
        exit( 1 );
    }

    int i;
    for ( i = 0; i < DB_NREADS( &db ); i++ )
    {
        db.reads[ i ].flags = READ_NONE;
    }

    if ( pcTrackRepeatsStrict )
    {
        fctx.trackRepeatStrict = track_load( &db, pcTrackRepeatsStrict );
        if ( !fctx.trackRepeatStrict )
        {
            fprintf( stderr, "could not load track %s\n", pcTrackRepeatsStrict );
            exit( 1 );
        }
    }

    if ( fctx.nMinNonRepeatBases != -1 || fctx.hrd )
    {
        fctx.trackRepeat = track_load( &db, pcTrackRepeats );

        if ( !fctx.trackRepeat )
        {
            fprintf( stderr, "could not load track %s\n", pcTrackRepeats );
            exit( 1 );
        }
    }

    fctx.trackTrim = track_load( &db, arg_trimTrack );

    if ( !fctx.trackTrim )
    {
        fprintf( stderr, "could not load track %s\n", arg_trimTrack );
        // exit( 1 );
    }

    if ( pathRules )
    {
        FILE* fileIn = fopen( pathRules, "r" );

        if ( fileIn == NULL )
        {
            fprintf( stderr, "could not open %s\n", pathRules );
            exit( 1 );
        }

        FilterRules rules;

        fread_rules( fileIn, &rules );

        printf( "excluding %d reads\n", rules.exclude_reads_n );

        for ( i = 0; i < rules.exclude_reads_n; i++ )
        {
            db.reads[ rules.exclude_reads[ i ] ].flags |= READ_DISCARD;
        }

        printf( "strict repeats for %d reads\n", rules.strict_reads_n );

        for ( i = 0; i < rules.strict_reads_n; i++ )
        {
            db.reads[ rules.strict_reads[ i ] ].flags |= READ_STRICT;
        }

        fclose( fileIn );
    }

    if ( fctx.hrd )
    {
#ifdef VERBOSE
        printf("HRD drop < %dbp - downsample by %d%% - fuzzing %d\n", fctx.hrd_len, fctx.hrd_rate, fctx.hrd_fuzz);
#endif

        fctx.hrd_mapgroup = calloc( DB_READ_MAXLEN( fctx.db ), sizeof( uint16_t ) );
    }

    // passes

    if ( fctx.useRLoader )
    {
        fctx.rl = rl_init( &db, 1 );

        pctx = pass_init( fileOvlIn, NULL );

        pctx->data       = &fctx;
        pctx->split_b    = 1;
        pctx->load_trace = 0;

        pass( pctx, loader_handler );
        rl_load_added( fctx.rl );
        pass_free( pctx );
    }

    pctx = pass_init( fileOvlIn, fileOvlOut );

    pctx->split_b         = 0;
    pctx->load_trace      = 1;
    pctx->unpack_trace    = 1;
    pctx->data            = &fctx;
    pctx->write_overlaps  = 1;
    pctx->purge_discarded = arg_purge;

    filter_pre( pctx, &fctx );
    pass( pctx, filter_handler );
    filter_post( &fctx );

    pass_free( pctx );

    // cleanup

    if ( fctx.useRLoader )
    {
        rl_free( fctx.rl );
    }

    Close_DB( &db );

    if ( fctx.hrd )
    {
        int i = 1;
        while ( fctx.hrd_groups[i] )
        {
            free(fctx.hrd_groups[i]);
            i += 1;
        }

        free(fctx.hrd_groups);
        free(fctx.hrd_ngroups);
        free(fctx.hrd_maxgroups);
        free(fctx.hrd_mapgroup);
    }

    fclose( fileOvlOut );
    fclose( fileOvlIn );

    return 0;
}
