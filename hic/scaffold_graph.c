
#define _GNU_SOURCE

#include "cycle.h"
#include "graphviz.h"
#include "hashmap.h"
#include "qsort_r.h"

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <search.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

// macros

#define SUM4( a ) ( ( a )[ 0 ] + ( a )[ 1 ] + ( a )[ 2 ] + ( a )[ 3 ] )

// command line defaults

#define DEF_ARG_C 1
#define DEF_ARG_L 10
#define DEF_ARG_M 1
#define DEF_ARG_S 20000
#define DEF_ARG_W 100

// toggles

#define SHOW_LINKS_READING_PROGRESS
#undef WRITE_INTERMEDIARY_STATE
#undef SCORING_DEBUG_OUTPUT

// scaffold graph nodes and edges

#define EDGE_LL 0
#define EDGE_LR 1
#define EDGE_RL 2
#define EDGE_RR 3

// constants

#define CUT_SITE_MAX_DIST 0

typedef struct ScafGraphNode ScafGraphNode;
typedef struct ScafGraphEdge ScafGraphEdge;
typedef struct ScaffoldContext ScaffoldContext;

struct ScafGraphEdge
{
    uint64_t source;        // from node
    uint64_t target;        // to node

    uint32_t links[ 4 ];    // number of LL LR RL RR links

    double score;           // score of the edge
};

struct ScafGraphNode
{
    char* seqname;          // node/contig name
    uint64_t len;           // sequence based length
    uint64_t efflen;        // effective lengths based on presence of cut sites
    uint64_t midpoint;      // recomputed midpoint based on effective lengths and mapability
    uint64_t selflinks;
    uint64_t flags;         // see NODE_ defines below

    uint64_t len_cuts;
    uint64_t efflen_cuts;

    int64_t best_l;
    int64_t best_r;

    int64_t path_beg;
    int64_t path_end;

    int64_t path_prev;
    int64_t path_next;
};

#define NODE_VISITED ( 0x1 << 0 )
#define NODE_CONTAINED ( 0x1 << 1 )
#define NODE_PATH ( 0x1 << 2 )
#define NODE_ACTIVE ( 0x1 << 3 )

#define NODE_TEMP ( 0x1 << 4 ) // first unused bit

// maintains the state of the app

struct ScaffoldContext
{
    // command line arguments

    uint16_t mapq;      // MAPQ threshold
    uint16_t minlinks;  // minimum number of links between contigs
    uint16_t minseqlen; // minimum sequence length
    char* seqnameprefix;
    uint16_t minclusters;
    char* path_output_prefix;

    uint16_t normalize_cuts;
    uint16_t normalize_len;
    uint16_t normalize_midpoint;

    uint16_t rescaffold_wnd_size;

    // sequences

    Hashmap mapseqname;
    uint64_t maxlen; // length of longest sequence

    uint64_t* mapping;

    // cutters

    uint64_t maxcutters;
    uint64_t ncutters;
    char** cutters;

    // cut points

    uint64_t* cut_points;
    uint64_t* cut_offsets;

    // graph nodes

    ScafGraphNode* sgnodes;
    uint64_t nsgnodes;
    uint64_t maxsgnodes;

    uint64_t* offsets; // index of a node's edges
    uint32_t maxoffsets;

    // graph edges

    ScafGraphEdge* sgedges;
    uint64_t maxsgedges;
    uint64_t nsgedges;
    uint64_t* sgedges_sorted;

    // links on which the edges are based on

    uint64_t* links; // sequence links array of (seqid_1, seqid_2, pos_1, pos_2)
    uint64_t maxlinks;
    uint64_t nlinks;

    // guide scaffold

    int64_t* guide;
    uint64_t nguide;
    uint64_t maxguide;

    // dump matrix

    uint16_t* matrix_self;
    uint32_t* vector_nself;
    uint64_t maxmatrix;

    uint64_t* offsets_links; // index of a sequence's links
    uint64_t maxoffsets_links;
};

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

#include <execinfo.h>

void printStackTrace()
{
    void* returnAddresses[ 500 ];
    int depth = backtrace( returnAddresses, sizeof returnAddresses / sizeof *returnAddresses );
    printf( "stack depth = %d\n", depth );

    char** symbols = backtrace_symbols( returnAddresses, depth );

    int i;
    for ( i = 0; i < depth; ++i )
    {
        printf( "%s\n", symbols[ i ] );
    }

    free( symbols );
}

/*
        fgetln is only available on bsd derived systems.
        getline on the other hand is only in gnu libc derived systems.

        if fgetln is missing emulate it using getline
*/

#if !defined( fgetln )

static char* fgetln_( FILE* stream, size_t* len )
{
    static char* linep    = NULL;
    static size_t linecap = 0;
    ssize_t length        = getline( &linep, &linecap, stream );

    if ( length == -1 )
    {
        free( linep );
        linep   = NULL;
        linecap = 0;
        length  = 0;
    }

    if ( len )
    {
        *len = length;
    }

    return linep;
}

#define fgetln fgetln_

#endif

// maps the sequence "name" to its id using the hash map

inline static int32_t name_to_id( ScaffoldContext* ctx, char* name )
{
    void* res = hashmap_get( &( ctx->mapseqname ), name, name + strlen( name ) );

    if ( res == NULL )
    {
        return -1;
    }

#ifdef DEBUG
    // clang complains about illegal memory alignment
    // in debug mode when casting the res void* to int32_t*
    int32_t val;
    memcpy( &val, res, sizeof( val ) );

    return val;
#else
    return *( (int32_t*)res );
#endif
}

inline static uint32_t cutter_to_id( ScaffoldContext* ctx, char* name )
{
    assert( name != NULL );

    uint32_t i;
    for ( i = 0; i < ctx->ncutters; i++ )
    {
        if ( strcmp( ctx->cutters[ i ], name ) == 0 )
        {
            return i;
        }
    }

    if ( ctx->ncutters + 1 >= ctx->maxcutters )
    {
        ctx->maxcutters = ctx->maxcutters * 1.2 + 10;
        ctx->cutters    = realloc( ctx->cutters, sizeof( char* ) * ctx->maxcutters );
    }

    ctx->cutters[ ctx->ncutters ] = strdup( name );
    ctx->ncutters += 1;

    return ctx->ncutters - 1;
}

/*
static void dump_links( ScaffoldContext* ctx, char* path, uint32_t binsize )
{
    uint64_t* links        = ctx->links;
    uint64_t nlinks        = ctx->nlinks;
    uint64_t maxlen        = ctx->maxlen;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    uint32_t bins     = ( maxlen + binsize - 1 ) / binsize;
    uint64_t szmatrix = bins * bins;

    uint8_t* matrix = malloc( szmatrix );

    FILE* fout = fopen( path, "w" );

    if ( !fout )
    {
        return;
    }

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1 = links[i];
        uint64_t id2 = links[i+1];
        uint64_t pos1 = links[i+2];
        uint64_t pos2 = links[i+3];

        fprintf(fout, "%" PRIu64 " %s %" PRIu64 " %s %" PRIu64 " %" PRIu64 "\n", id1, sgnodes[id1].seqname, id2, sgnodes[id2].seqname, pos1, pos2);
    }

    fclose( fout );

    free( matrix );
}
*/

// comparison function for uint64_t[4]

static int cmp_uint64_4( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    int i;
    for ( i = 0; i < 4; i++ )
    {
        if ( a[ i ] < b[ i ] )
        {
            return -1;
        }
        else if ( a[ i ] > b[ i ] )
        {
            return 1;
        }
    }

    return 0;
}

static int cmp_uint64_3( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    int i;
    for ( i = 0; i < 3; i++ )
    {
        if ( a[ i ] < b[ i ] )
        {
            return -1;
        }
        else if ( a[ i ] > b[ i ] )
        {
            return 1;
        }
    }

    return 0;
}

static int cmp_uint64( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    if ( *a < *b )
    {
        return -1;
    }
    else if ( *a > *b )
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int cmp_uint64_4_pos1( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    if ( a[ 0 ] < b[ 0 ] )
    {
        return -1;
    }
    else if ( a[ 0 ] > b[ 0 ] )
    {
        return 1;
    }

    if ( a[ 2 ] < b[ 2 ] )
    {
        return -1;
    }
    else if ( a[ 2 ] > b[ 2 ] )
    {
        return 1;
    }

    return 0;
}

static int cmp_uint64_4_id2( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    if ( a[ 1 ] < b[ 1 ] )
    {
        return -1;
    }
    else if ( a[ 1 ] > b[ 1 ] )
    {
        return 1;
    }

    return 0;
}

static void reverse_array( uint64_t* arr, uint64_t n )
{
    uint64_t i = 0;
    n -= 1;

    for ( i = 0; i < n; i++, n-- )
    {
        uint64_t t = arr[ i ];
        arr[ i ]   = arr[ n ];
        arr[ n ]   = t;
    }
}

static void links_sort( ScaffoldContext* ctx, uint64_t offset )
{
    uint64_t* links = ctx->links + offset;
    uint64_t nlinks = ctx->nlinks - offset;

    printf( "  sorting %" PRIu64 " links\n", nlinks / 4 );
    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_uint64_4 );
}

static void links_sort_id2( ScaffoldContext* ctx, uint64_t offset )
{
    uint64_t* links = ctx->links + offset;
    uint64_t nlinks = ctx->nlinks - offset;

    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_uint64_4_id2 );
}

/*
static void dump_matrix( ScaffoldContext* ctx, uint32_t seqid, FILE* fileout, uint32_t binsize )
{
    uint64_t* offsets      = ctx->offsets_links;
    uint64_t* links        = ctx->links;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    uint32_t* vector_nself = ctx->vector_nself;
    uint16_t* matrix_self   = ctx->matrix_self;

    uint32_t bins     = ( sgnodes[ seqid ].len + binsize - 1 ) / binsize;
    uint64_t szmatrix = bins * bins;

    printf( "writing matrix for %s (%d bins)\n", sgnodes[ seqid ].seqname, bins );

    if ( szmatrix > ctx->maxmatrix )
    {
        ctx->maxmatrix   = szmatrix * 1.2 + 1000;
        ctx->matrix_self = matrix_self = realloc( matrix_self, sizeof(uint16_t) * ctx->maxmatrix );
    }

    uint64_t obeg = offsets[ seqid ];
    uint64_t oend = offsets[ seqid + 1 ];

    if ( obeg >= oend )
    {
        return;
    }

    bzero( matrix_self, sizeof(uint16_t) * bins * bins );
    bzero( vector_nself, sizeof( uint32_t ) * ctx->maxlen );

    while ( obeg < oend )
    {
        assert( links[ obeg ] == seqid );

        uint64_t id1  = links[ obeg ];
        uint64_t id2  = links[ obeg + 1 ];
        uint64_t pos1 = links[ obeg + 2 ];
        uint64_t pos2 = links[ obeg + 3 ];

        assert( id1 == seqid );

        if ( pos1 > sgnodes[ id1 ].len )
        {
            obeg += 4;
            continue;
        }

        if ( pos2 > sgnodes[ id2 ].len )
        {
            obeg += 4;
            continue;
        }

        uint64_t bin1 = pos1 / binsize;
        uint64_t bin2 = pos2 / binsize;

        uint64_t el1 = bin1 * bins + bin2;
        uint64_t el2 = bin2 * bins + bin1;

        if ( id1 == id2 )
        {
            if ( matrix_self[ el1 ] < UINT16_MAX )
            {
                matrix_self[ el1 ] += 1;
            }

            if ( matrix_self[ el2 ] < UINT16_MAX )
            {
                matrix_self[ el2 ] += 1;
            }
        }
        else
        {
            if ( seqid == id1 )
            {
                vector_nself[ bin1 ] += 1;
            }
            else if ( seqid == id2 )
            {
                vector_nself[ bin2 ] += 1;
            }
        }

        obeg += 4;
    }

    uint32_t j;
    for ( j = 0; j < bins; j++ )
    {
        uint32_t nself = vector_nself[ j ];

        uint32_t k;
        for ( k = 0; k < bins; k++ )
        {
            if ( matrix_self[ j * bins + k ] > 0 || nself > 0 )
            {
                fprintf( fileout, "%d %s %d %d %u %d\n",
                         seqid, sgnodes[ seqid ].seqname, j, k,
                         matrix_self[ j * bins + k ],
                         nself );
                nself = 0;
            }
        }
    }
}
*/

static uint64_t* create_cut_site_mapping( ScaffoldContext* ctx, uint64_t seqid, uint64_t cutid, uint64_t maxdist )
{
    uint64_t* mapping = ctx->mapping;
    uint64_t seqlen   = ctx->sgnodes[ seqid ].len;
    uint64_t ncutters = ctx->ncutters;

    bzero( mapping, sizeof( uint64_t ) * seqlen );

    uint64_t beg = ctx->cut_offsets[ seqid * ncutters + cutid ];
    uint64_t end = ctx->cut_offsets[ seqid * ncutters + cutid + 1 ];

    uint64_t mfrom;
    uint64_t mto = 0;

    while ( beg < end )
    {
        mfrom       = mto;
        uint64_t pt = ctx->cut_points[ beg ];

        if ( beg + 1 < end )
        {
            assert( pt < ctx->cut_points[ beg + 1 ] );

            mto = ( ctx->cut_points[ beg + 1 ] + pt ) / 2;
        }
        else
        {
            mto = ctx->sgnodes[ seqid ].len;
        }

        uint64_t _mfrom = mfrom;
        uint64_t _mto   = mto;

        if ( maxdist > 0 )
        {
            if ( pt - mfrom > maxdist )
            {
                _mfrom = pt - maxdist;
            }

            if ( mto - pt > maxdist )
            {
                _mto = pt + maxdist;
            }
        }

        while ( _mfrom < _mto )
        {
            mapping[ _mfrom ] = pt;
            _mfrom += 1;
        }

        beg += 1;
    }

    return mapping;
}

static void remap_links( ScaffoldContext* ctx, uint64_t cutid, uint64_t offset )
{
    assert( ctx->nlinks );
    assert( ctx->links );

    ScafGraphNode* sgnodes = ctx->sgnodes;
    uint64_t* links   = ctx->links;
    uint64_t nlinks   = ctx->nlinks;
    uint64_t id1_cur  = links[ 0 ];
    uint64_t* mapping = create_cut_site_mapping( ctx, id1_cur, cutid, CUT_SITE_MAX_DIST );

    printf( "mapping positions %" PRIu64 "..%" PRIu64 " to cut sites\n", offset / 4, nlinks / 4 );

    links_sort( ctx, offset );

    uint64_t i;
    for ( i = offset; i < nlinks; i += 4 )
    {
        uint64_t id1  = links[ i ];
        uint64_t pos1 = links[ i + 2 ];

        if ( id1 != id1_cur )
        {
            mapping = create_cut_site_mapping( ctx, id1, cutid, CUT_SITE_MAX_DIST );
            id1_cur = id1;
        }

        uint64_t mpos1 = mapping[ pos1 ];

        assert( mpos1 < sgnodes[id1].len );

        if ( mpos1 == 0 )
        {
            links[ i + 2 ] = 0;
            links[ i + 3 ] = 0;
        }
        else
        {
            links[ i + 2 ] = mpos1;
        }
    }

    links_sort_id2( ctx, offset );
    uint64_t id2_cur = links[ 1 ];

    mapping = create_cut_site_mapping( ctx, id2_cur, cutid, CUT_SITE_MAX_DIST );

    for ( i = offset; i < nlinks; i += 4 )
    {
        uint64_t id2  = links[ i + 1 ];
        uint64_t pos2 = links[ i + 3 ];

        if ( id2 != id2_cur )
        {
            mapping = create_cut_site_mapping( ctx, id2, cutid, CUT_SITE_MAX_DIST );
            id2_cur = id2;
        }

        uint64_t mpos2 = mapping[ pos2 ];

        assert(mpos2 < sgnodes[id2].len );

        if ( mpos2 == 0 )
        {
            links[ i + 2 ] = 0;
            links[ i + 3 ] = 0;
        }
        else
        {
            links[ i + 3 ] = mpos2;
        }
    }

    links_sort( ctx, offset );
}

static int64_t node_l_reverse( ScaffoldContext* ctx, uint64_t n )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges = ctx->sgedges;
    ScafGraphNode* source  = sgnodes + n;

    if ( source->best_l == -1 )
    {
        return -1;
    }

    ScafGraphEdge* edge   = sgedges + source->best_l;
    ScafGraphNode* target = sgnodes + edge->target;

    if ( target->best_l != -1 && sgedges[ target->best_l ].target == n )
    {
        return target->best_l;
    }

    if ( target->best_r != -1 && sgedges[ target->best_r ].target == n )
    {
        return target->best_r;
    }

    return -1;
}

static int64_t node_r_reverse( ScaffoldContext* ctx, uint64_t n )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges = ctx->sgedges;
    ScafGraphNode* source  = sgnodes + n;

    if ( source->best_r == -1 )
    {
        return -1;
    }

    ScafGraphEdge* edge   = sgedges + source->best_r;
    ScafGraphNode* target = sgnodes + edge->target;

    if ( target->best_l != -1 && sgedges[ target->best_l ].target == n )
    {
        return target->best_l;
    }

    if ( target->best_r != -1 && sgedges[ target->best_r ].target == n )
    {
        return target->best_r;
    }

    return -1;
}

static ScafGraphEdge* edge_new( ScaffoldContext* ctx, uint64_t src, uint64_t trgt, uint32_t* links, double score )
{
    uint64_t nsgedges        = ctx->nsgedges;
    uint64_t maxsgedges      = ctx->maxsgedges;
    ScafGraphEdge* sgedges   = ctx->sgedges;
    uint64_t* sgedges_sorted = ctx->sgedges_sorted;

    if ( nsgedges + 1 >= maxsgedges )
    {
        uint64_t old = maxsgedges;
        maxsgedges = ctx->maxsgedges = maxsgedges * 1.2 + 1000;
        sgedges = ctx->sgedges = realloc( sgedges, sizeof( ScafGraphEdge ) * maxsgedges );
        sgedges_sorted = ctx->sgedges_sorted = realloc( sgedges_sorted, maxsgedges * sizeof( uint64_t ) );

        bzero( sgedges + old, sizeof( ScafGraphEdge ) * ( maxsgedges - old ) );
    }

    // printf("edge_new> %" PRIu64 " -> %" PRIu64 " ... %s -> %s\n", src, trgt, ctx->sgnodes[src].seqname, ctx->sgnodes[trgt].seqname);

    ScafGraphEdge* edge = sgedges + nsgedges;

    edge->source = src;
    edge->target = trgt;
    edge->score  = score;

    if ( links )
    {
        memcpy( edge->links, links, 4 * sizeof( uint32_t ) );
    }

    ctx->nsgedges += 1;

    return edge;
}

#ifdef QSORT_R_DATA_FIRST
static int cmp_r_scafgraphedge( void* data, const void* x, const void* y )
#else
static int cmp_r_scafgraphedge( const void* x, const void* y, void* data )
#endif
{
    ScafGraphEdge* sgedges = (ScafGraphEdge*)data;

    ScafGraphEdge* a = sgedges + ( *(uint64_t*)x );
    ScafGraphEdge* b = sgedges + ( *(uint64_t*)y );

    if ( a->source < b->source )
        return -1;
    else if ( a->source > b->source )
        return 1;

    if ( a->target < b->target )
        return -1;
    else if ( a->target > b->target )
        return 1;

    return 0;
}

/*
static int cmp_scafgraphedge( const void* x, const void* y )
{
    ScafGraphEdge* a = (ScafGraphEdge*)x;
    ScafGraphEdge* b = (ScafGraphEdge*)y;

    if ( a->source < b->source )
        return -1;
    else if ( a->source > b->source )
        return 1;

    if ( a->target < b->target )
        return -1;
    else if ( a->target > b->target )
        return 1;

    return 0;
}
*/

static void edge_offsets_update( ScaffoldContext* ctx )
{
    uint64_t nsgnodes   = ctx->nsgnodes;
    uint64_t* offsets   = ctx->offsets;
    uint64_t maxoffsets = ctx->maxoffsets;
    // ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges   = ctx->sgedges;
    uint64_t nsgedges        = ctx->nsgedges;
    uint64_t* sgedges_sorted = ctx->sgedges_sorted;

    uint64_t i;

    printf( "sorting and computing offsets for %" PRIu64 " edges\n", nsgedges );

    if ( nsgnodes >= maxoffsets )
    {
        maxoffsets = ctx->maxoffsets = maxoffsets * 1.2 + 100;
        offsets = ctx->offsets = realloc( offsets, maxoffsets * sizeof( uint64_t ) );
    }

    bzero( offsets, nsgnodes * sizeof( uint64_t ) );

    for ( i = 0; i < nsgedges; i++ )
    {
        sgedges_sorted[ i ] = i;
        offsets[ sgedges[ i ].source ] += 1;
    }

    QSORT_R( sgedges_sorted, nsgedges, sizeof( uint64_t ), sgedges, cmp_r_scafgraphedge );

    // qsort( sgedges, nsgedges, sizeof( ScafGraphEdge ), cmp_scafgraphedge );

    uint64_t off = 0;
    uint64_t coff;

    for ( i = 0; i <= nsgnodes; i++ )
    {
        coff         = offsets[ i ];
        offsets[ i ] = off;
        off += coff;
    }
}

static ScafGraphNode* node_new( ScaffoldContext* ctx, char* seqname, uint64_t seqlen, uint64_t efflen )
{
    if ( ctx->nsgnodes + 1 >= ctx->maxsgnodes )
    {
        size_t prev = ctx->maxsgnodes;

        ctx->maxsgnodes = 1.2 * ctx->maxsgnodes + 100;
        ctx->sgnodes    = realloc( ctx->sgnodes, ctx->maxsgnodes * sizeof( ScafGraphNode ) );

        bzero( ctx->sgnodes + prev, ( ctx->maxsgnodes - prev ) * sizeof( ScafGraphNode ) );
    }

    if ( ctx->nsgnodes + 2 >= ctx->maxoffsets )
    {
        ctx->maxoffsets = 1.2 * ctx->maxoffsets + 100;
        ctx->offsets    = realloc( ctx->offsets, ctx->maxoffsets * sizeof( uint64_t ) );
    }

    ScafGraphNode* node = ctx->sgnodes + ctx->nsgnodes;
    node->seqname       = seqname;
    node->len           = seqlen;
    node->efflen        = efflen;
    node->best_l        = -1;
    node->best_r        = -1;

    if ( seqlen > ctx->maxlen )
    {
        ctx->maxlen = seqlen;
    }

    ctx->offsets[ ctx->nsgnodes + 1 ] = ctx->offsets[ ctx->nsgnodes ];

    ctx->nsgnodes += 1;

    return node;
}

static int process_cut_sites( ScaffoldContext* ctx, const char* path_sites )
{
    printf( "processing cut sites\n" );

    FILE* fin = fopen( path_sites, "r" );

    if ( fin == NULL )
    {
        return 0;
    }

    uint64_t maxpos = 1000;
    uint64_t* pos   = malloc( sizeof( uint64_t ) * maxpos );
    uint64_t npos   = 0;

    char* line;
    while ( ( line = fgetln( fin, NULL ) ) )
    {
        char* token;
        char* next    = line;
        int col       = 0;
        int64_t seqid = -1;
        int64_t cutid = -1;

        while ( ( token = strsep( &next, " \t" ) ) != NULL )
        {
            if ( col == 0 )
            {
                seqid = name_to_id( ctx, token );
                if ( seqid == -1 )
                {
                    fprintf( stderr, "could not resolve sequence %s\n", token );
                }
            }
            else if ( col == 1 )
            {
                cutid = cutter_to_id( ctx, token );
            }
            else
            {
                if ( npos + 3 >= maxpos )
                {
                    maxpos = 1.2 * maxpos + 100;
                    pos    = realloc( pos, sizeof( uint64_t ) * maxpos );
                }

                pos[ npos ]     = seqid;
                pos[ npos + 1 ] = cutid;
                pos[ npos + 2 ] = strtol( token, NULL, 10 );
                npos += 3;
            }

            col += 1;
        }
    }

    qsort( pos, npos / 3, sizeof( uint64_t ) * 3, cmp_uint64_3 );

    uint64_t nsgnodes    = ctx->nsgnodes;
    uint64_t ncutters    = ctx->ncutters;
    uint64_t* cut_points = ctx->cut_points = calloc( ( npos / 3 ), sizeof( uint64_t ) );
    uint64_t* cut_offsets = ctx->cut_offsets = calloc( ( nsgnodes + 1 ) * ncutters, sizeof( uint64_t ) );
    uint64_t ncpoint                         = 0;

    uint64_t i;
    for ( i = 0; i < npos; i += 3 )
    {
        uint64_t seqid        = pos[ i ];
        uint64_t cutid        = pos[ i + 1 ];
        cut_points[ ncpoint ] = pos[ i + 2 ];
        ncpoint += 1;

        cut_offsets[ seqid * ncutters + cutid ] += 1;
    }

    printf( "  %" PRIu64 " sites\n", npos / 3 );

    free( pos );

    uint64_t seq;
    uint64_t off = 0;

    for ( seq = 0; seq <= nsgnodes; seq += 1 )
    {
        uint64_t cut;
        uint64_t off_prev = off;
        for ( cut = 0; cut < ncutters; cut += 1 )
        {
            uint64_t coff                       = cut_offsets[ seq * ncutters + cut ];
            cut_offsets[ seq * ncutters + cut ] = off;
            off += coff;
        }

        ctx->sgnodes[ seq ].len_cuts = off - off_prev;
    }

    return 1;
}

static int process_fasta( ScaffoldContext* ctx, const char* path_fasta )
{
    printf( "processing fasta\n" );

    // look for <path_fasta>.fai

    FILE* fin;
    char* line;
    char* seqname   = NULL;
    uint64_t seqlen = 0;
    char path_lengths[ PATH_MAX ];
    sprintf( path_lengths, "%s.fai", path_fasta );

    if ( ( fin = fopen( path_lengths, "r" ) ) )
    {
        while ( ( line = fgetln( fin, NULL ) ) )
        {
            // <name>\t<length>\t...\n

            char* num = strchr( line, '\t' );
            *num      = '\0';

            seqlen = strtoull( num + 1, NULL, 10 );

            node_new( ctx, strdup( line ), seqlen, seqlen );

            if ( seqlen > ctx->maxlen )
            {
                ctx->maxlen = seqlen;
            }
        }
    }
    else
    {
        // process fasta file

        if ( !( fin = fopen( path_fasta, "r" ) ) )
        {
            fprintf( stderr, "failed to open %s\n", path_fasta );
            return 0;
        }

        while ( ( line = fgetln( fin, NULL ) ) )
        {
            int lenline = strlen( line );

            if ( line[ lenline - 1 ] != '\n' )
            {
                fprintf( stderr, "line to long in %s\n", path_fasta );
                fclose( fin );
                return 0;
            }

            line[ lenline - 1 ] = '\0';

            if ( line[ 0 ] == '>' )
            {
                if ( seqlen > ctx->minseqlen )
                {
                    node_new( ctx, seqname, seqlen, seqlen );

                    if ( seqlen > ctx->maxlen )
                    {
                        ctx->maxlen = seqlen;
                    }
                }

                seqname = strdup( line + 1 );
                seqlen  = 0;
            }
            else
            {
                seqlen += lenline - 1;
            }
        }

        fclose( fin );

        if ( seqlen )
        {
            node_new( ctx, strdup( line ), seqlen, seqlen );

            if ( seqlen > ctx->maxlen )
            {
                ctx->maxlen = seqlen;
            }
        }
    }

    uint64_t nsgnodes      = ctx->nsgnodes;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    hashmap_open( &( ctx->mapseqname ), nsgnodes );

    uint64_t i;
    for ( i = 0; i < nsgnodes; i++ )
    {
        sgnodes[ i ].midpoint = sgnodes[ i ].len / 2;
        hashmap_put( &( ctx->mapseqname ),
                     sgnodes[ i ].seqname,
                     sgnodes[ i ].seqname + strlen( sgnodes[ i ].seqname ),
                     &i,
                     (void*)( &i ) + sizeof( uint64_t ) );
    }

    printf( "  %" PRIu64 " nodes\n", nsgnodes );

    return 1;
}

inline static void link_new( ScaffoldContext* ctx,
                             uint64_t id1, uint64_t id2, uint64_t pos1, uint64_t pos2,
                             uint8_t symmetric )
{
    uint64_t* links   = ctx->links;
    uint64_t maxlinks = ctx->maxlinks;
    uint64_t nlinks   = ctx->nlinks;

    /*
    if ( pos2 >= ctx->sgnodes[id2].len )
    {
        printStackTrace();
    }
    */

    assert( pos1 < ctx->sgnodes[ id1 ].len );
    assert( pos2 < ctx->sgnodes[ id2 ].len );

    if ( nlinks + 8 >= maxlinks )
    {
        uint64_t old = maxlinks;
        maxlinks     = maxlinks * 1.5 + 1000;

        links = realloc( links, sizeof( uint64_t ) * maxlinks );

        if ( links == NULL )
        {
            fprintf( stderr, "failed to allocate %" PRIu64 " bytes for links\n", sizeof( uint64_t ) * maxlinks );
        }

        bzero( links + old, ( maxlinks - old ) * sizeof( uint64_t ) );
    }

    links[ nlinks ]     = id1;
    links[ nlinks + 1 ] = id2;
    links[ nlinks + 2 ] = pos1;
    links[ nlinks + 3 ] = pos2;

    nlinks += 4;

    if ( symmetric )
    {
        links[ nlinks ]     = id2;
        links[ nlinks + 1 ] = id1;
        links[ nlinks + 2 ] = pos2;
        links[ nlinks + 3 ] = pos1;

        nlinks += 4;
    }

    ctx->links    = links;
    ctx->maxlinks = maxlinks;
    ctx->nlinks   = nlinks;
}

static int process_guide(ScaffoldContext* ctx, char* path, int ignore_missing)
{
    printf("processing guide scaffold %s\n", path);

    int64_t* guide = ctx->guide;
    uint64_t nguide = ctx->nguide;
    uint64_t maxguide = ctx->maxguide;

    FILE* fin;

    if ( !( fin = fopen( path, "r" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path );
        return 0;
    }

    int err = 0;
    char* line;
    uint64_t nline = 0;
    while ( ( line = fgetln( fin, NULL ) ) )
    {
        nline += 1;

        int lenline = strlen( line );

        if (lenline < 2 || line[0] == '#' || line[0] == '>')
        {
            continue;
        }

        if ( line[ lenline - 1 ] == '\n' )
        {
            lenline -= 1;
            line[ lenline ] = '\0';
        }
        else if ( !feof(fin) )
        {
            fprintf( stderr, "line %" PRIu64 " exceeds maximum length\n", nline );
            err = 1;
            break;
        }

        char orientation = line[0];
        char* seq = NULL;

        if ( isspace(line[1]) )
        {
            if ( orientation != '+' && orientation != '-' )
            {
                printf("bad orientation in line %" PRIu64 "\n", nline);
                err = 1;
                break;
            }

            int i = 1;

            while ( i < lenline && isspace(line[i]) )
            {
                i++;
            }

            seq = line + i;
        }
        else
        {
            orientation = '+';
            seq = line;
        }

        if ( !isalnum(seq[0]) )
        {
            printf("malformed sequence name in line %" PRIu64 "\n", nline);

            if ( !ignore_missing )
            {
                err = 1;
                break;
            }
        }

        // printf("%c %s\n", orientation, seq);

        if ( nguide + 1 >= maxguide )
        {
            maxguide = maxguide * 1.2 + 1000;
            guide = realloc(guide, sizeof(int64_t) * maxguide);
        }

        int32_t seqid = name_to_id(ctx, seq);

        if ( seqid == -1 )
        {
            fprintf( stderr, "unknown sequence %s in line %" PRIu64 "\n", seq, nline);
            err = 1;
            break;
        }

        if (orientation == '-')
        {
            guide[nguide] = (-1) * seqid;
        }
        else
        {
            guide[nguide] = seqid;
        }

        nguide += 1;
    }

    ctx->guide = guide;
    ctx->maxguide = maxguide;
    ctx->nguide = nguide;

    printf("  read scaffold of %" PRIu64 " sequences\n", nguide);

    fclose(fin);

    return ( err == 0 );
}

static int process_links( ScaffoldContext* ctx, const char* path_links, const char* cutter, int sort )
{
    printf( "processing links in %s using cutter %s\n", path_links, cutter );

    uint16_t mapq          = ctx->mapq;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    FILE* fin;

    if ( !( fin = fopen( path_links, "r" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_links );
        return 0;
    }

#ifdef SHOW_LINKS_READING_PROGRESS
    fseek( fin, 0, SEEK_END );
    off_t finlen = ftello( fin );
    rewind( fin );

    off_t finchunk     = finlen / 100;
    off_t finnext      = finchunk;
    uint32_t finchunkn = 1;
#endif

    char* line;
    uint64_t nline = 0;
    while ( ( line = fgetln( fin, NULL ) ) )
    {
#ifdef SHOW_LINKS_READING_PROGRESS
        if ( ftello( fin ) > finnext )
        {
            if ( finchunkn % 10 == 0 )
                printf( "%d", finchunkn );
            else if ( finchunkn % 2 == 0 )
                printf( "." );
            finnext += finchunk;
            finchunkn += 1;

            fflush( stdout );
        }
#endif

        nline += 1;

        int lenline = strlen( line );

        if ( line[ lenline - 1 ] != '\n' )
        {
            fprintf( stderr, "line to long in %s\n", path_links );
            fclose( fin );
            return 0;
        }

        line[ lenline - 1 ] = '\0';

        char* token;
        char* next = line;
        int col    = 0;
        int skip   = 0;
        int id1, id2, mapqtemp;
        uint64_t pos1, pos2;

        while ( skip == 0 && ( token = strsep( &next, " \t" ) ) != NULL )
        {
            switch ( col )
            {
                case 0:
                    id1 = name_to_id( ctx, token );
                    if ( id1 == -1 )
                    {
                        // printf( "unknown sequence %s\n", token );
                        skip = 1;
                    }
                    break;

                case 1:
                    pos1 = strtol( token, NULL, 10 );
                    break;

                case 2:
                    mapqtemp = strtol( token, NULL, 10 );
                    if ( mapqtemp < mapq )
                        skip = 1;
                    break;

                case 3:
                    if ( token[ 0 ] == '\0' )
                    {
                        id2 = id1;
                    }
                    else
                    {
                        id2 = name_to_id( ctx, token );

                        if ( id2 == -1 )
                        {
                            printf( "unknown sequence %s\n", token );
                            skip = 1;
                        }
                    }

                    break;

                case 4:
                    pos2 = strtol( token, NULL, 10 );
                    break;

                case 5:
                    mapqtemp = strtol( token, NULL, 10 );
                    if ( mapqtemp < mapq )
                        skip = 1;
                    else
                        skip = 2; // stop processing token/columns

                    break;
            }

            col += 1;
        }

        if ( skip == 1 )
        {
            continue;
        }

        if ( id1 == id2 )
        {
            sgnodes[ id1 ].selflinks += 1;
        }

        if ( pos1 >= sgnodes[ id1 ].len )
        {
            if ( pos1 - 100 < sgnodes[ id1 ].len )
            {
                pos1 -= 100;
            }
            else
            {
                printf( "line %" PRIu64 " mapping position %" PRIu64 " beyond contig length %" PRIu64 "\n",
                        nline, pos1, sgnodes[ id1 ].len );
                continue;
            }
        }

        if ( pos2 >= sgnodes[ id2 ].len )
        {
            if ( pos2 - 100 < sgnodes[ id2 ].len )
            {
                pos2 -= 100;
            }
            else
            {
                printf( "line %" PRIu64 " mapping position %" PRIu64 " beyond contig length %" PRIu64 "\n",
                        nline, pos2, sgnodes[ id2 ].len );
                continue;
            }
        }

        link_new( ctx, id1, id2, pos1, pos2, 1 );
    }

#ifdef SHOW_LINKS_READING_PROGRESS
    printf( " (%" PRIu64 ")\n", ctx->nlinks / 4 );
#endif

    fclose( fin );

    if ( sort )
    {
        links_sort( ctx, 0 );
    }

    return 1;
}

/*
static void print_links( const char* path, uint64_t* links, uint64_t nlinks )
{
    FILE* fout = fopen( path, "w" );

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        fprintf( fout, "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n", links[ i ], links[ i + 1 ], links[ i + 2 ], links[ i + 3 ] );
    }

    fclose( fout );
}
*/

static uint64_t compute_effective_length_node( uint64_t seqlen,
                                               uint64_t* bins,
                                               uint64_t binsize, uint64_t empty_threshold )
{
    uint64_t efflen      = seqlen;
    uint64_t seqbins     = ( seqlen + binsize - 1 ) / binsize;
    uint64_t lastbinsize = seqlen % binsize;

    if ( lastbinsize != 0 )
    {
        bins[ seqbins - 1 ] = ( (double)( bins[ seqbins - 1 ] ) / lastbinsize ) * binsize;
    }

    uint64_t j;
    for ( j = 0; j < seqbins; j++ )
    {
        if ( bins[ j ] <= empty_threshold )
        {
            if ( j == seqbins - 1 )
            {
                efflen -= lastbinsize;
            }
            else
            {
                efflen -= binsize;
            }
        }
    }

    return efflen;
}

static int intersect_unsorted( uint64_t* a1, uint64_t n1, uint64_t* a2, uint64_t n2 )
{
    qsort( a1, n1, sizeof( uint64_t ), cmp_uint64 );
    qsort( a2, n2, sizeof( uint64_t ), cmp_uint64 );

    uint64_t i     = 0;
    uint64_t j     = 0;
    uint64_t szint = 0;

    while ( i < n1 && j < n2 )
    {
        if ( a1[ i ] < a2[ j ] )
        {
            i += 1;
        }
        else if ( a1[ i ] > a2[ j ] )
        {
            j += 1;
        }
        else
        {
            szint += 1;

            i += 1;
            j += 1;
        }
    }

    return szint;
}

static void compute_effective_length_cuts( ScaffoldContext* ctx )
{
    uint64_t nlinks        = ctx->nlinks;
    uint64_t* links        = ctx->links;
    ScafGraphNode* sgnodes = ctx->sgnodes;
    uint64_t* cut_offsets  = ctx->cut_offsets;
    uint64_t ncutters      = ctx->ncutters;

    uint64_t maxpos = 1000;
    uint64_t* pos   = malloc( sizeof( uint64_t ) * maxpos );

    uint64_t maxcuts = 1000;
    uint64_t* cuts   = malloc( sizeof( uint64_t ) * maxcuts );

    uint64_t i;
    for ( i = 0; i < nlinks; )
    {
        uint64_t curid = links[ i ];
        // uint64_t beg   = i;
        uint64_t npos = 0;

        for ( ; i < nlinks && links[ i ] == curid; i += 4 )
        {
            if ( npos + 1 == maxpos )
            {
                maxpos = maxpos * 1.2 + 1000;
                pos    = realloc( pos, sizeof( uint64_t ) * maxpos );
            }

            pos[ npos ] = links[ i + 2 ];
            npos += 1;
        }

        uint64_t beg_cuts = cut_offsets[ curid * ncutters ];
        uint64_t end_cuts = cut_offsets[ ( curid + 1 ) * ncutters ];
        uint64_t ncuts    = end_cuts - beg_cuts;

        assert( ncuts == sgnodes[ curid ].len_cuts );

        if ( ncuts > maxcuts )
        {
            maxcuts = ncuts + 1000;
            cuts    = realloc( cuts, sizeof( uint64_t ) * maxcuts );
        }

        memcpy( cuts, ctx->cut_points + beg_cuts, sizeof( uint64_t ) * ncuts );

        sgnodes[ curid ].efflen_cuts = intersect_unsorted( pos, npos, cuts, ncuts );

        if ( sgnodes[ curid ].len_cuts )
            printf( "%s %" PRIu64 " l %" PRIu64 " el %" PRIu64 " lc %" PRIu64 " elc %" PRIu64 "\n", sgnodes[ curid ].seqname, curid,
                    sgnodes[ curid ].len, sgnodes[ curid ].efflen,
                    sgnodes[ curid ].len_cuts, sgnodes[ curid ].efflen_cuts );
    }

    free( cuts );
    free( pos );
}

static void compute_effective_length( ScaffoldContext* ctx, uint64_t binsize, uint64_t empty_threshold )
{
    uint64_t nlinks        = ctx->nlinks;
    uint64_t* links        = ctx->links;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    uint64_t maxbins = ctx->maxlen / binsize + 1;
    uint64_t* bins   = calloc( maxbins, sizeof( uint64_t ) );
    uint64_t added   = 0;

    uint64_t efflen;

    uint64_t curid = links[ 0 ];
    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1 = links[ i ];
        uint64_t id2 = links[ i + 1 ];

        if ( id1 != curid )
        {
            if ( added )
            {
                efflen = compute_effective_length_node( sgnodes[ curid ].len, bins, binsize, empty_threshold );

                if ( efflen == 0 || sgnodes[ curid ].selflinks == 0 )
                {
                    // printf( "warning: %s falling back to actual length for effective length\n", sgnodes[ curid ].seqname );
                }
                else
                {
                    sgnodes[ curid ].efflen = efflen;
                }

                added = 0;
                bzero( bins, maxbins * sizeof( uint64_t ) );
            }

            curid = id1;
        }

        uint64_t pos1 = links[ i + 2 ];
        bins[ pos1 / binsize ] += 1;
        added += 1;

        if ( id1 == id2 )
        {
            uint64_t pos2 = links[ i + 3 ];
            bins[ pos2 / binsize ] += 1;
        }
    }

    if ( added )
    {
        efflen = compute_effective_length_node( sgnodes[ curid ].len, bins, binsize, empty_threshold );

        if ( efflen == 0 )
        {
            printf( "warning: %s falling back to actual length for effective length\n", sgnodes[ curid ].seqname );
        }
        else
        {
            sgnodes[ curid ].efflen = efflen;
        }
    }

    free( bins );
}

static void purge_contained( ScaffoldContext* ctx )
{
    uint64_t nlinks  = ctx->nlinks;
    uint64_t* links  = ctx->links;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    printf( "purging contained nodes' links\n" );

    uint64_t i;
    uint64_t drop = 0;

    for ( i = 0; i < nlinks; i+=4 )
    {
        uint64_t id1 = links[ i ];
        uint64_t id2 = links[ i + 1 ];

        if ( (sgnodes[id1].flags & NODE_CONTAINED) || (sgnodes[id2].flags & NODE_CONTAINED) )
        {
            drop += 4;
            memset( links + i, 0, sizeof( uint64_t ) * 4 );
        }
    }

    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_uint64_4 );

    nlinks -= drop;

    printf( "  discarded %" PRIu64 " links\n", drop / 4 );
    printf( "  %" PRIu64 " links\n", nlinks / 4 );

    memmove( links, links + drop, nlinks * sizeof( uint64_t ) );

    ctx->nlinks = nlinks;
}

static void filter_links( ScaffoldContext* ctx, uint8_t drop_self )
{
    uint32_t discard = 0;
    uint32_t dupe    = 0;
    uint32_t self    = 0;
    uint64_t nlinks  = ctx->nlinks;
    uint64_t* links  = ctx->links;

    printf( "filtering links\n" );

    uint64_t beg = 0;
    uint64_t end = 0;
    while ( end < nlinks )
    {
        beg = end;
        end += 4;

        uint64_t id1 = links[ beg ];
        uint64_t id2 = links[ beg + 1 ];

        uint64_t drop = 0;
        while ( end < nlinks && links[ end ] == id1 && links[ end + 1 ] == id2 )
        {
            if ( links[ end + 2 ] == links[ end - 4 + 2 ] && links[ end + 3 ] == links[ end - 4 + 3 ] )
            {
                drop += 4;
                memset( links + end - 4, 0, sizeof( uint64_t ) * 4 );
            }

            end += 4;
        }

        if ( id1 == id2 && drop_self )
        {
            discard += end - beg;
            self += ( end - beg ) / 4;
            memset( links + beg, 0, sizeof( uint64_t ) * ( end - beg ) );
        }
        else
        {
            discard += drop;

            dupe += drop / 4;
        }
    }

    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_uint64_4 );

    nlinks -= discard;

    printf( "  %d dupe links\n", dupe );
    printf( "  %d self hits\n", self );

    printf( "  discarded %d links\n", discard / 4 );
    printf( "  %" PRIu64 " links\n", nlinks / 4 );

    memmove( links, links + discard, nlinks * sizeof( uint64_t ) );

    ctx->nlinks = nlinks;
}

static void compute_midpoint( ScaffoldContext* ctx, uint64_t off_links )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    uint64_t* links        = ctx->links;
    uint64_t nlinks        = ctx->nlinks;

    if ( off_links )
    {
        assert( links[ off_links - 4 ] != links[ off_links ] );
        assert( links[ off_links ] == links[ nlinks - 4 ] );
    }

    uint64_t i;
    for ( i = off_links; i < nlinks; )
    {
        uint64_t previd  = links[ i ];
        uint64_t previdx = i;

        while ( i < nlinks && links[ i ] == previd )
        {
            i += 4;
        }

        uint64_t mididx = 4 * ( ( i - previdx ) / 8 );

        qsort( links + previdx, ( i - previdx ) / 4, sizeof( uint64_t ) * 4, cmp_uint64_4_pos1 );
        uint64_t mid = links[ previdx + mididx + 2 ];
        qsort( links + previdx, ( i - previdx ) / 4, sizeof( uint64_t ) * 4, cmp_uint64_4 );

        // printf("previd = %" PRIu64 " nsgnodes = %" PRIu64 "\n", previd, ctx->nsgnodes);

        sgnodes[ previd ].midpoint = mid;

        if ( off_links != 0 )
        {
            printf( "midpoint %s %" PRIu64 " @ %" PRIu64 " %" PRIu64 "\n", sgnodes[ previd ].seqname, previd, mid, sgnodes[previd].len );
        }
    }
}

inline static int edge_type( ScafGraphNode* sgnodes, uint64_t* link )
{
    uint64_t id1  = link[ 0 ];
    uint64_t id2  = link[ 1 ];
    uint64_t pos1 = link[ 2 ];
    uint64_t pos2 = link[ 3 ];

    int ret = 0;

    /*
    printf("%" PRIu64 " %s @ %" PRIu64 " %" PRIu64 " x %" PRIu64 " @ %s %" PRIu64 " %" PRIu64 "\n",
            id1, sgnodes[id1].seqname, pos1, sgnodes[id1].len,
            id2, sgnodes[id2].seqname, pos2, sgnodes[id2].len);
    */

    assert( pos1 < sgnodes[ id1 ].len );
    assert( pos2 < sgnodes[ id2 ].len );

    if ( pos1 > sgnodes[ id1 ].midpoint ) // R_
    {
        ret += 2;
    }

    if ( pos2 > sgnodes[ id2 ].midpoint ) // _R
    {
        ret += 1;
    }

    return ret;
}

static void score_node( ScaffoldContext* ctx, uint64_t nid, uint8_t active_only )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges   = ctx->sgedges;
    uint64_t* offsets        = ctx->offsets;
    uint64_t* sgedges_sorted = ctx->sgedges_sorted;

    ScafGraphNode* source = sgnodes + nid;

    uint64_t beg          = offsets[ nid ];
    uint64_t end          = offsets[ nid + 1 ];

    if ( beg >= end )
    {
        return;
    }

    if ( source->flags & NODE_CONTAINED )
    {
        return;
    }

#ifdef SCORING_DEBUG_OUTPUT
    printf( "SCORING %s ID %" PRIu64 " BL %" PRId64 " BR %" PRId64 "\n", source->seqname, nid, source->best_l, source->best_r );
#endif

    if ( source->best_l != -1 )
    {
        ScafGraphEdge* sbl  = sgedges + source->best_l;
        ScafGraphNode* sblt = sgnodes + sbl->target;

        if ( ( sblt->flags & NODE_CONTAINED ) ||
                ( ( sblt->best_l == -1 || sgedges[ sblt->best_l ].target != nid ) &&
                ( sblt->best_r == -1 || sgedges[ sblt->best_r ].target != nid ) ) )
        {
#ifdef SCORING_DEBUG_OUTPUT
            if ( sblt->best_l != -1 && sblt->best_r != -1 )
                printf( "  clear L -> %" PRIu64 " -> best_l %" PRIu64 " best_r %" PRIu64 "\n", sgedges[ sblt->best_l ].target, sbl->target, sgedges[ sblt->best_r ].target );
#endif

            source->best_l = -1;
        }
    }

    if ( source->best_r != -1 )
    {
        ScafGraphEdge* sbr  = sgedges + source->best_r;
        ScafGraphNode* sbrt = sgnodes + sbr->target;

        if ( ( sbrt->flags & NODE_CONTAINED ) ||
                ( ( sbrt->best_l == -1 || sgedges[ sbrt->best_l ].target != nid ) &&
                ( sbrt->best_r == -1 || sgedges[ sbrt->best_r ].target != nid ) ) )
        {
#ifdef SCORING_DEBUG_OUTPUT
            printf( "  clear R\n" );
#endif
            source->best_r = -1;
        }
    }

    uint64_t lensource = source->efflen;
    // lensource          = source->efflen_cuts;

    for ( ; beg < end; beg++ )
    {
        uint64_t edge_idx = sgedges_sorted[ beg ];

        ScafGraphEdge* edge   = sgedges + edge_idx;
        ScafGraphNode* target = sgnodes + edge->target;
        uint32_t* links       = edge->links;
        uint64_t lentarget    = target->efflen;
        // lentarget             = target->efflen_cuts;

        // double score = (double)SUM4(links) / lensource / lentarget;

        if ( target->flags & NODE_CONTAINED )
        {
            continue;
        }

        if ( active_only && (target->flags & NODE_ACTIVE) == 0 )
        {
            continue;
        }

        /*
        double score_ll = ( (double)links[ EDGE_LL ] ) / ( lensource * lentarget );
        double score_lr = ( (double)links[ EDGE_LR ] ) / ( lensource * lentarget );
        double score_rl = ( (double)links[ EDGE_RL ] ) / ( lensource * lentarget );
        double score_rr = ( (double)links[ EDGE_RR ] ) / ( lensource * lentarget );

        printf( "  --> %s LL %e LR %e RL %e RR %e\n", target->seqname, score_ll, score_lr, score_rl, score_rr );

        double max_l = MAX( score_ll, score_lr );
        double max_r = MAX( score_rl, score_rr );

        if ( max_l > max_r )
        {
            edge->score          = MAX( edge->score, max_l );
            ScafGraphEdge* bestl = sgedges + source->best_l;

            if ( source->best_l == -1 || ( max_l > bestl->score ) )
            {
                printf( "  new L ... %" PRId64 " -> %" PRId64 "\n", source->best_l, edge_idx );
                source->best_l = edge_idx;
            }
        }
        else
        {
            edge->score          = MAX( edge->score, max_r );
            ScafGraphEdge* bestr = sgedges + source->best_r;

            if ( source->best_r == -1 || ( max_r > bestr->score ) )
            {
                printf( "  new R ... %" PRId64 " -> %" PRId64 "\n", source->best_r, edge_idx );
                source->best_r = edge_idx;
            }
        }
        */

        double score_l = ( (double)( links[ EDGE_LL ] + links[ EDGE_LR ] ) ) / ( lensource * lentarget );
        double score_r = ( (double)( links[ EDGE_RL ] + links[ EDGE_RR ] ) ) / ( lensource * lentarget );

        // TODO: XXXX

        /*
        if ( MAX( lensource, lentarget ) > 1 * 1000 * 1000 )
            if ( ctx->minlinks * MAX( lensource, lentarget ) / ( 1.0 * 1000 * 1000 ) > SUM4( links ) )
            {
                score_l = 0.0;
                score_r = 0.0;
            }
        */

#ifdef SCORING_DEBUG_OUTPUT
        printf( "  --> %s %d %d %d %d L %e R %e\n", target->seqname, links[ 0 ], links[ 1 ], links[ 2 ], links[ 3 ], score_l, score_r );
#endif

        if ( score_l > score_r )
        {
            edge->score          = MAX( edge->score, score_l );
            ScafGraphEdge* bestl = sgedges + source->best_l;

            if ( source->best_l == -1 || score_l > bestl->score )
            {
#ifdef SCORING_DEBUG_OUTPUT
                printf( "  new L ... %" PRId64 " -> %" PRId64 "\n", source->best_l, edge_idx );
#endif
                source->best_l = edge_idx;
            }
        }
        else
        {
            edge->score          = MAX( edge->score, score_r );
            ScafGraphEdge* bestr = sgedges + source->best_r;

            if ( source->best_r == -1 || score_r > bestr->score )
            {
#ifdef SCORING_DEBUG_OUTPUT
                printf( "  new R ... %" PRId64 " -> %" PRId64 "\n", source->best_r, edge_idx );
#endif
                source->best_r = edge_idx;
            }
        }
    }
}

static void score_nodes( ScaffoldContext* ctx )
{
    uint64_t nsgnodes        = ctx->nsgnodes;

    printf( "scoring nodes\n" );

    uint64_t i;
    for ( i = 0; i < nsgnodes; i++ )
    {
        score_node(ctx, i, 0);
    }
}

static void write_path_nodes( ScaffoldContext* ctx, FILE* fout, ScafGraphNode* node, int indent,
                              int64_t* path, uint64_t* npath )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges = ctx->sgedges;

    int revcomp = 0;
    if ( node->best_l != -1 && sgedges[ node->best_l ].target == (uint64_t)node->path_next )
    {
        revcomp = 1;
    }

    while ( 1 )
    {
        fprintf( fout, "# %*s%c %s %d %" PRIu64 "\n", indent * 2, "", revcomp ? '-' : '+', node->seqname, indent, node->len );

        if ( node->flags & NODE_PATH )
        {
            uint64_t _npath = *npath;

            write_path_nodes( ctx, fout, sgnodes + node->path_beg, indent + 1, path, npath );

            if ( revcomp )
            {
                uint64_t beg = _npath;
                uint64_t end = *npath - 1;

                while ( beg < end )
                {
                    int64_t t   = path[ beg ];
                    path[ beg ] = ( -1 ) * path[ end ];
                    path[ end ] = ( -1 ) * t;

                    beg += 1;
                    end -= 1;
                }

                if ( beg == end )
                {
                    path[ beg ] = ( -1 ) * path[ beg ];
                }
            }
        }
        else
        {
            path[ *npath ] = node - sgnodes;

            if ( revcomp )
            {
                path[ *npath ] *= -1;
            }

            ( *npath ) += 1;
        }

        if ( node->path_next == -1 )
        {
            break;
        }

        node = sgnodes + node->path_next;

        if ( node->best_r != -1 && sgedges[ node->best_r ].target == (uint64_t)node->path_prev )
        {
            revcomp = 1;
        }
        else
        {
            revcomp = 0;
        }
    }
}

static int write_paths( ScaffoldContext* ctx, char* pathout )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    uint64_t nsgnodes      = ctx->nsgnodes;

    int64_t* path  = malloc( nsgnodes * sizeof( int64_t ) );
    uint64_t npath = 0;

    printf( "writing paths to %s\n", pathout );

    FILE* fout;
    fout = fopen( pathout, "w" );

    if ( !fout )
    {
        fprintf( stderr, "could not open %s\n", pathout );
        return 0;
    }

    uint64_t i;
    for ( i = 0; i < nsgnodes; i++ )
    {
        ScafGraphNode* node = sgnodes + i;

        if ( !( ( node->flags & NODE_PATH ) && !( node->flags & NODE_CONTAINED ) ) )
        {
            continue;
        }

        fprintf( fout, "# + %s 0 %" PRIu64 "\n", node->seqname, node->len );

        write_path_nodes( ctx, fout, sgnodes + node->path_beg, 1, path, &npath );

        fprintf( fout, "> %s\n", node->seqname );
        uint64_t j;
        for ( j = 0; j < npath; j++ )
        {
            int rc = (path[j] < 0);

            if ( rc )
            {
                fprintf( fout, "- %s\n", sgnodes[ ( -1 ) * path[ j ] ].seqname );
            }
            else
            {
                fprintf( fout, "+ %s\n", sgnodes[ path[ j ] ].seqname );
            }
        }
        npath = 0;
    }

    free( path );

    fclose( fout );

    return 1;
}

#ifdef WRITE_INTERMEDIARY_STATE

static int write_edges( ScaffoldContext* ctx, char* path )
{
    ScafGraphNode* sgnodes   = ctx->sgnodes;
    ScafGraphEdge* sgedges   = ctx->sgedges;
    uint64_t nsgnodes        = ctx->nsgnodes;
    uint64_t* offsets        = ctx->offsets;
    uint64_t* sgedges_sorted = ctx->sgedges_sorted;

    printf( "writing edges to %s\n", path );

    FILE* fout;
    fout = fopen( path, "w" );

    if ( !fout )
    {
        fprintf( stderr, "could not open %s\n", path );
        return 0;
    }

    uint64_t i;
    for ( i = 0; i < nsgnodes; i++ )
    {
        ScafGraphNode* node = sgnodes + i;

        int64_t beg = offsets[ i ];
        int64_t end = offsets[ i + 1 ];

        for ( ; beg < end; beg++ )
        {
            uint64_t edge_idx = sgedges_sorted[ beg ];

            ScafGraphEdge* edge = sgedges + edge_idx;
            uint32_t* edgelinks = edge->links;

            fprintf( fout, "%s %s %d %d %d %d %d\n", node->seqname, sgnodes[ edge->target ].seqname,
                     edgelinks[ 0 ], edgelinks[ 1 ], edgelinks[ 2 ], edgelinks[ 3 ], SUM4( edgelinks ) );
        }
    }

    fclose( fout );

    return 1;
}

#endif

#ifdef WRITE_INTERMEDIARY_STATE

static int write_graph( ScaffoldContext* ctx, char* path_output )
{
    ScafGraphEdge* sgedges   = ctx->sgedges;
    uint64_t* offsets        = ctx->offsets;
    ScafGraphNode* sgnodes   = ctx->sgnodes;
    uint64_t nsgnodes        = ctx->nsgnodes;
    uint64_t* sgedges_sorted = ctx->sgedges_sorted;
    // uint64_t ncutters        = ctx->ncutters;

    printf( "writing graph to %s\n", path_output );

    FILE* fout;
    fout = fopen( path_output, "w" );

    if ( !fout )
    {
        fprintf( stderr, "could not open %s\n", path_output );
        return 0;
    }

    fprintf( fout, "digraph mergegraph {\n" );
    fprintf( fout, "ordering=out;\n" );
    fprintf( fout, "graph [fontname=\"helvetica\"];\nnode [fontname=\"helvetica\" style=filled];\nedge [fontname=\"helvetica\"]\n" );

    uint64_t i;
    for ( i = 0; i < nsgnodes; i++ )
    {
        ScafGraphNode* node = sgnodes + i;
        char attr[ 256 ];

        if ( node->flags & NODE_CONTAINED )
        {
            continue;
        }

        int64_t beg = offsets[ i ];
        int64_t end = offsets[ i + 1 ];

        if ( beg >= end )
        {
            continue;
        }

        *attr = '\0';

        if ( node->flags & NODE_PATH )
        {
            strcat( attr, "fillcolor=darkgoldenrod1 " );
        }
        else
        {
            strcat( attr, "fillcolor=white " );
        }

        if ( node->selflinks == 0 )
        {
            // printf( "no self links %" PRIu64 " %s\n", i, node->seqname );
            node->selflinks = 1;
        }

        fprintf( fout, "%" PRIu64 " [label=\"%s %" PRIu64 "\n%" PRIu64 " %" PRIu64 " %" PRIu64 "\" %s comment=\"%s\"];\n",
                 i, graphviz_wrap_string( node->seqname ), i,
                 node->len, node->efflen, node->len_cuts,
                 attr, node->seqname );

        for ( ; beg < end; beg++ )
        {
            uint64_t edge_idx = sgedges_sorted[ beg ];

            ScafGraphEdge* edge = sgedges + edge_idx;
            uint32_t* edgelinks = edge->links;

            *attr = '\0';

            if ( (int64_t)edge_idx == node->best_l && (int64_t)edge_idx == node->best_r )
            {
                strcpy( attr, "color=black" );
            }
            else if ( (int64_t)edge_idx == node->best_l )
            {
                strcpy( attr, "color=blue fontcolor=blue" );
            }
            else if ( (int64_t)edge_idx == node->best_r )
            {
                strcpy( attr, "color=red fontcolor=red" );
            }
            else
            {
                // strcpy( attr, "color=darkolivegreen fontcolor=darkolivegreen" );
                continue;
            }

            fprintf( fout,
                     "%" PRIu64 " -> %" PRIu64 " [label=\"%u\n%u\n%e\" %s];\n",
                     i, edge->target,
                     edgelinks[ EDGE_LL ] + edgelinks[ EDGE_LR ], edgelinks[ EDGE_RL ] + edgelinks[ EDGE_RR ],
                     edge->score,
                     attr );
        }
    }

    fprintf( fout, "}\n" );

    fclose( fout );

    return 1;
}

#endif

static void build_edges( ScaffoldContext* ctx )
{
    uint64_t* links        = ctx->links;
    uint64_t nlinks        = ctx->nlinks;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    uint64_t i;

    // create edges

    uint32_t edge_links[ 4 ];
    memset( edge_links, 0, sizeof( uint32_t ) * 4 );

    uint64_t id1_prev = links[ 0 ];
    uint64_t id2_prev = links[ 1 ];

    for ( i = 4; i < nlinks; i += 4 )
    {
        uint64_t id1 = links[ i ];
        uint64_t id2 = links[ i + 1 ];

        if ( id1 != id1_prev || id2 != id2_prev )
        {
            if ( SUM4( edge_links ) >= ctx->minlinks )
            {
                edge_new( ctx, id1_prev, id2_prev, edge_links, 0.0 );
            }

            memset( edge_links, 0, sizeof( uint32_t ) * 4 );

            id1_prev = id1;
            id2_prev = id2;
        }

        edge_links[ edge_type( sgnodes, links + i ) ] += 1;
    }

    if ( SUM4( edge_links ) >= ctx->minlinks )
    {
        edge_new( ctx, id1_prev, id2_prev, edge_links, 0.0 );
    }

    edge_offsets_update( ctx );

    printf( "  %" PRIu64 " distinct edges\n", ctx->nsgedges );
}

static void nodes_clear_flags( ScaffoldContext* ctx, uint64_t flags )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    uint64_t nsgnodes      = ctx->nsgnodes;
    uint64_t i;

    uint64_t mask = ~flags;

    for ( i = 0; i < nsgnodes; i++ )
    {
        ScafGraphNode* node = sgnodes + i;

        node->flags &= mask;
    }
}

static int follow_rev_path( ScaffoldContext* ctx, ScafGraphEdge* edge,
                            uint64_t** path, uint64_t* maxpath, uint64_t* npath )
{
    int circular           = 0;
    uint64_t* _path        = *path;
    uint64_t _maxpath      = *maxpath;
    uint64_t _npath        = *npath;
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges = ctx->sgedges;

    assert( _npath > 0 );

    uint64_t prev = _path[ _npath - 1 ];
    uint64_t next = edge->target;

    sgnodes[ prev ].flags |= NODE_VISITED;

    printf( "follow %3" PRIu64 " %s\n", _npath - 1, sgnodes[ prev ].seqname );

    while ( 1 )
    {
        if ( _npath >= _maxpath )
        {
            _maxpath = 1.2 * _maxpath + 100;
            _path    = realloc( _path, _maxpath * sizeof( uint64_t ) );
        }

        printf( "    -> %3" PRIu64 " %s\n", _npath, sgnodes[ next ].seqname );

        if ( sgnodes[ next ].flags & NODE_VISITED )
        {
            circular = 1;
            break;
        }

        _path[ _npath ] = next;
        _npath += 1;

        sgnodes[ next ].flags |= NODE_VISITED;

        uint64_t tmp;

        if ( sgnodes[ next ].best_l != -1 && sgedges[ sgnodes[ next ].best_l ].target != prev && node_l_reverse( ctx, next ) != -1 )
        {
            tmp = sgedges[ sgnodes[ next ].best_l ].target;
        }
        else if ( sgnodes[ next ].best_r != -1 && sgedges[ sgnodes[ next ].best_r ].target != prev && node_r_reverse( ctx, next ) != -1 )
        {
            tmp = sgedges[ sgnodes[ next ].best_r ].target;
        }
        else
        {
            break;
        }

        prev = next;
        next = tmp;
    }

    *path    = _path;
    *maxpath = _maxpath;
    *npath   = _npath;

    return circular;
}

static int break_circular_path( ScaffoldContext* ctx, uint64_t* path, uint64_t npath )
{
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges = ctx->sgedges;

    int64_t shift    = -1;
    double score_min = 0;
    uint64_t i;

    for ( i = 0; i < npath; i++ )
    {
        ScafGraphEdge* e_fwd;
        ScafGraphEdge* e_rev;
        ScafGraphNode* n_next;
        ScafGraphNode* n_cur = sgnodes + path[ i ];
        uint64_t nidx_next;

        // next node in path

        if ( i + 1 < npath )
        {
            nidx_next = i + 1;
        }
        else
        {
            nidx_next = 0;
        }

        n_next = ctx->sgnodes + path[ nidx_next ];

        // edge to next node

        if ( sgedges[ n_cur->best_l ].target == path[ nidx_next ] )
        {
            e_fwd = sgedges + n_cur->best_l;
        }
        else if ( sgedges[ n_cur->best_r ].target == path[ nidx_next ] )
        {
            e_fwd = sgedges + n_cur->best_r;
        }
        else
        {
            assert( 0 );
        }

        // return edge from next node

        if ( sgedges[ n_next->best_l ].target == path[ i ] )
        {
            e_rev = sgedges + n_next->best_l;
        }
        else if ( sgedges[ n_next->best_r ].target == path[ i ] )
        {
            e_rev = sgedges + n_next->best_r;
        }
        else
        {
            assert( 0 );
        }

        double score = ( e_fwd->score + e_rev->score ) / 2;

        if ( shift == -1 || score_min > score )
        {
            shift     = i;
            score_min = score;
        }
    }

    printf( "CYCLE @ %" PRIu64 " %e\n", shift, score_min );

    printf( "PRE  " );
    for ( i = 0; i < npath; i++ )
        printf( "%" PRIu64 " ", path[ i ] );
    printf( "\n" );

    array_cycle_left( path, npath, sizeof( uint64_t ), shift + 1 );

    printf( "POST " );
    for ( i = 0; i < npath; i++ )
        printf( "%" PRIu64 " ", path[ i ] );
    printf( "\n" );

    return 1;
}

static uint64_t build_paths( ScaffoldContext* ctx )
{
    uint64_t nsgnodes      = ctx->nsgnodes;
    uint64_t* links        = ctx->links;
    ScafGraphNode* sgnodes = ctx->sgnodes;
    ScafGraphEdge* sgedges = ctx->sgedges;

    uint64_t maxpath = 100;
    uint64_t npath   = 0;
    uint64_t* path   = malloc( maxpath * sizeof( uint64_t ) );

    uint64_t i;
    char pathname[ 128 ];

    int64_t* shift                = calloc( maxpath, sizeof( int64_t ) );
    uint64_t* sid2pid             = calloc( nsgnodes, sizeof( uint64_t ) );
    const uint64_t shift_rev_bit  = ( 1ULL << 63 );
    const uint64_t shift_rev_mask = ~shift_rev_bit;

    uint64_t paths = 0;

    for ( i = 0; i < nsgnodes; i++ )
    {
        ScafGraphNode* node = sgnodes + i;

        if ( node->flags & ( NODE_CONTAINED | NODE_VISITED ) )
        {
            continue;
        }

        int64_t rev_l = node_l_reverse( ctx, i );
        int64_t rev_r = node_r_reverse( ctx, i );
        int circular  = 0;

        if ( rev_l != -1 && rev_r == -1 )
        {
            npath = 1;
            *path = i;

            circular = follow_rev_path( ctx, sgedges + node->best_l, &path, &maxpath, &npath );
        }
        else if ( rev_l == -1 && rev_r != -1 )
        {
            npath = 1;
            *path = i;

            circular = follow_rev_path( ctx, sgedges + node->best_r, &path, &maxpath, &npath );
        }
        else if ( rev_l != -1 && rev_r != -1 )
        {
            npath = 1;
            *path = i;

            circular = follow_rev_path( ctx, sgedges + node->best_l, &path, &maxpath, &npath );

            if ( !circular )
            {
                reverse_array( path, npath );
                follow_rev_path( ctx, sgedges + node->best_r, &path, &maxpath, &npath );
            }
        }
        else
        {
            continue;
        }

        if ( circular )
        {
            break_circular_path( ctx, path, npath );

            printf( "  -- circular\n" );
            //            assert(0);
        }

        uint64_t j;
        uint64_t pathselflinks = 0;
        shift[ 0 ]             = 0;

        if ( sgnodes[ path[ 0 ] ].best_l != -1 &&
             sgedges[ sgnodes[ path[ 0 ] ].best_l ].target == path[ 1 ] )
        {
            // 0 ->L 1
            shift[ 0 ] |= shift_rev_bit;
            printf( "  c %s\n", sgnodes[ path[ 0 ] ].seqname );
        }

        sgnodes[ path[ 0 ] ].path_prev = -1;
        paths += 1;

        uint64_t pathlen         = 0;
        uint64_t effpathlen      = 0;
        uint64_t pathlen_cuts    = 0;
        uint64_t effpathlen_cuts = 0;

        for ( j = 0; j < npath; j++ )
        {
            ScafGraphNode* pnode = sgnodes + path[ j ];

            pathlen += pnode->len;
            pathlen_cuts += pnode->len_cuts;
            effpathlen += pnode->efflen;
            effpathlen_cuts += pnode->efflen_cuts;

            pnode->flags |= NODE_CONTAINED | NODE_TEMP; // node contained and worked on

            sid2pid[ path[ j ] ] = j; // mapping of seq.id / node.id to index in path

            if ( j > 0 )
            {
                shift[ j ] = ( shift_rev_mask & shift[ j - 1 ] ) + sgnodes[ path[ j - 1 ] ].len;

                if ( sgnodes[ path[ j ] ].best_r != -1 &&
                     sgedges[ sgnodes[ path[ j ] ].best_r ].target == path[ j - 1 ] )
                {
                    shift[ j ] |= shift_rev_bit;
                    printf( "  c %s\n", sgnodes[ path[ j ] ].seqname );
                }

                pnode->path_prev = path[ j - 1 ];
            }

            if ( j < npath - 1 )
            {
                pnode->path_next = path[ j + 1 ];
            }

            pathselflinks += sgnodes[ path[ j ] ].selflinks;
        }

        sgnodes[ path[ npath - 1 ] ].path_next = -1;

        sprintf( pathname, "path_%" PRIu64, ctx->nsgnodes );
        ScafGraphNode* pathnode = node_new( ctx, strdup( pathname ), pathlen, effpathlen );
        pathnode->path_beg      = *path;
        pathnode->efflen_cuts   = effpathlen_cuts;
        pathnode->len_cuts      = pathlen_cuts;
        pathnode->path_end      = path[ npath - 1 ];
        pathnode->flags         = NODE_PATH;
        pathnode->selflinks     = pathselflinks;
        pathnode->midpoint      = pathlen / 2;

        sgnodes = ctx->sgnodes;

        printf( "PATH %s %s..%s\n", pathname, sgnodes[ *path ].seqname, sgnodes[ path[ npath - 1 ] ].seqname );

        // create edges

        uint32_t linkcounts[ 4 ] = {0, 0, 0, 0};

        uint64_t nlinks          = ctx->nlinks;
        uint64_t nlinks_new_node = 0;

        for ( j = 0; j < nlinks; )
        {
            uint64_t id1       = links[ j ];
            uint64_t id2       = links[ j + 1 ];
            uint64_t id1_flags = sgnodes[ id1 ].flags;
            uint64_t id2_flags = sgnodes[ id2 ].flags;

            if ( !( id1_flags & NODE_TEMP ) )
            {
                uint64_t entry[ 4 ];

                for ( ; j < nlinks && links[ j ] == id1; j += 4 )
                {
                    id2       = links[ j + 1 ];
                    id2_flags = sgnodes[ id2 ].flags;

                    if ( id2_flags & NODE_TEMP )
                    {
                        entry[ 0 ] = id1;
                        entry[ 1 ] = ctx->nsgnodes - 1;
                        entry[ 2 ] = links[ j + 2 ];

                        int64_t seqshift = shift[ sid2pid[ id2 ] ];
                        entry[ 3 ]       = shift_rev_mask & seqshift;
                        if ( seqshift & shift_rev_bit )
                        {
                            entry[ 3 ] += sgnodes[ id2 ].len - links[ j + 3 ] - 1;
                        }
                        else
                        {
                            entry[ 3 ] += links[ j + 3 ];
                        }

                        linkcounts[ edge_type( sgnodes, entry ) ] += 1;

                        link_new( ctx, entry[ 0 ], entry[ 1 ], entry[ 2 ], entry[ 3 ], 1 );
                        links = ctx->links;
                        nlinks_new_node += 1;
                    }
                }

                if ( SUM4( linkcounts ) > ctx->minlinks )
                {
                    edge_new( ctx, id1, pathnode - sgnodes, linkcounts, 0.0 );

                    uint32_t tmp          = linkcounts[ EDGE_LR ];
                    linkcounts[ EDGE_LR ] = linkcounts[ EDGE_RL ];
                    linkcounts[ EDGE_RL ] = tmp;

                    edge_new( ctx, pathnode - sgnodes, id1, linkcounts, 0.0 );
                }

                bzero( linkcounts, sizeof( uint32_t ) * 4 );
            }
            else if ( ( id1_flags & NODE_TEMP ) && ( id2_flags & NODE_TEMP ) )
            {
                int64_t seqshift      = shift[ sid2pid[ id1 ] ];
                uint64_t pos1_shifted = shift_rev_mask & seqshift;
                if ( seqshift & shift_rev_bit )
                {
                    pos1_shifted += sgnodes[ id1 ].len - links[ j + 2 ] - 1;
                }
                else
                {
                    pos1_shifted += links[ j + 2 ];
                }

                seqshift              = shift[ sid2pid[ id2 ] ];
                uint64_t pos2_shifted = shift_rev_mask & seqshift;
                if ( seqshift & shift_rev_bit )
                {
                    pos2_shifted += sgnodes[ id2 ].len - links[ j + 3 ] - 1;
                }
                else
                {
                    pos2_shifted += links[ j + 3 ];
                }

                link_new( ctx, ctx->nsgnodes - 1, ctx->nsgnodes - 1, pos1_shifted, pos2_shifted, 0 );
                links = ctx->links;
                nlinks_new_node += 1;

                pathnode->selflinks += 1;
                j += 4;
            }
            else
            {
                j += 4;
            }
        }

        sgedges = ctx->sgedges;

        for ( j = 0; j < npath; j++ )
        {
            sgnodes[ path[ j ] ].flags &= ~NODE_TEMP;
        }

        links_sort( ctx, nlinks );
        compute_midpoint( ctx, ctx->nlinks - nlinks_new_node * 4 );
    }

    links_sort( ctx, 0 );

    edge_offsets_update( ctx );

    free( shift );
    free( sid2pid );
    free( path );

    nodes_clear_flags( ctx, NODE_VISITED );

    return paths;
}

static uint32_t scaffold( ScaffoldContext* ctx )
{
    char* path_output_prefix = ctx->path_output_prefix;

    printf( "scaffolding %" PRIu64 " sequences\n", ctx->nsgnodes );

    uint32_t npaths;
    int iteration = 0;

    char path[ PATH_MAX ];

    do
    {
        printf("scaffolding %" PRIu64 " links\n", ctx->nlinks);

        score_nodes( ctx );

#ifdef WRITE_INTERMEDIARY_STATE
        if ( path_output_prefix )
        {
            sprintf( path, "%sgraph_%02d.dot", path_output_prefix, iteration );
            write_graph( ctx, path );

            sprintf( path, "%sgraph_%02d.txt", path_output_prefix, iteration );
            write_edges( ctx, path );
        }
#endif

        npaths = build_paths( ctx );

#ifdef WRITE_INTERMEDIARY_STATE
        sprintf( path, "%spaths_%02d.txt", path_output_prefix, iteration );
        write_paths( ctx, path );
#endif

        iteration += 1;

        purge_contained(ctx);

    } while ( npaths != 0 );

    sprintf( path, "%spaths_final.txt", path_output_prefix );
    write_paths( ctx, path );

    return 1;
}

static void rescaffold( ScaffoldContext* ctx )
{
    printf("rescaffolding %" PRIu64 " sequences with window size %" PRIu16 "\n",
            ctx->nsgnodes, ctx->rescaffold_wnd_size);

    ScafGraphNode* sgnodes = ctx->sgnodes;
    int64_t* guide =  ctx->guide;
    uint64_t nguide = ctx->nguide;
    uint16_t wnd_size = ctx->rescaffold_wnd_size;

#ifdef WRITE_INTERMEDIARY_STATE
    char* path_output_prefix = ctx->path_output_prefix;
#endif

    uint64_t wnd_prev = 0;
    uint64_t i;

    do
    {
        for ( i = wnd_prev ; i < nguide && i < wnd_prev + wnd_size ; i++ )
        {
            sgnodes[ labs(guide[i]) ].flags |= NODE_ACTIVE;
        }

        for ( i = wnd_prev ; i < nguide && i < wnd_prev + wnd_size ; i++ )
        {
            score_node(ctx, labs(guide[i]), 1);
        }

        for ( i = wnd_prev ; i < nguide && i < wnd_prev + wnd_size ; i++ )
        {
            sgnodes[ labs(guide[i]) ].flags &= ~NODE_ACTIVE;
        }

        wnd_prev += wnd_size;

    } while ( i < nguide );

#ifdef WRITE_INTERMEDIARY_STATE
    char path[ PATH_MAX ];
    sprintf( path, "%sgraph_re_0.dot", path_output_prefix );
    write_graph( ctx, path );
#endif

    uint64_t npaths = build_paths(ctx);
    printf("built %" PRIu64 " paths\n", npaths);

#ifdef WRITE_INTERMEDIARY_STATE
    sprintf( path, "%sgraph_re_1.dot", path_output_prefix );
    write_graph( ctx, path );
#endif

    scaffold(ctx);
}

/*
static void efflen_thresholds( ScaffoldContext* ctx, uint64_t binsize )
{
    uint64_t* links  = ctx->links;
    uint64_t nlinks  = ctx->nlinks;
    uint64_t maxbins = ( ctx->maxlen + binsize - 1 ) / binsize;
    uint64_t* bins   = calloc( maxbins, sizeof( uint64_t ) );

    uint64_t maxmetabins = 128;
    uint64_t* metabins   = calloc( maxmetabins, sizeof( uint64_t ) );

    uint64_t i;
    uint64_t id1_prev;

    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1  = links[ i ];
        uint64_t id2  = links[ i + 1 ];
        uint64_t pos1 = links[ i + 2 ];
        uint64_t pos2 = links[ i + 3 ];

        if ( id1 != id1_prev )
        {
            uint64_t j;
            for ( j = 0; j < ctx->sgnodes[ id1_prev ].len / binsize; j++ )
            {
                if ( bins[ j ] < maxmetabins )
                {
                    metabins[ j ] += 1;
                }
            }

            bzero( bins, maxbins * sizeof( uint64_t ) );
            id1_prev = id1;
        }

        bins[ pos1 / binsize ] += 1;

        if ( id1 == id2 )
        {
            bins[ pos2 / binsize ] += 1;
        }
    }

    for ( i = 0; i < maxmetabins; i++ )
    {
        printf( "%" PRIu64 " %" PRIu64 "\n", i, metabins[ i ] );
    }
}
*/

/*
static void update_offsets_links( ScaffoldContext* ctx )
{
    uint64_t nseq       = ctx->nsgnodes;
    uint64_t* links     = ctx->links;
    uint64_t nlinks     = ctx->nlinks;
    uint64_t maxoffsets = ctx->maxoffsets_links;
    uint64_t* offsets   = ctx->offsets_links;

    printf( "computing offsets\n" );

    if ( nseq + 1 > maxoffsets )
    {
        maxoffsets = ( nseq + 1 ) * 1.2 + 1000;
        offsets    = realloc( offsets, sizeof( uint64_t ) * maxoffsets );
    }

    bzero( offsets, sizeof( uint64_t ) * maxoffsets );

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        offsets[ links[ i ] ] += 4;
    }

    uint64_t off = 0;
    uint64_t coff;

    uint64_t j;
    for ( j = 0; j <= nseq; j++ )
    {
        coff         = offsets[ j ];
        offsets[ j ] = off;
        off += coff;
    }

    ctx->maxoffsets_links    = maxoffsets;
    ctx->offsets_links = offsets;
}
*/

static void usage( char* app )
{
    printf( "%s [-clms n] [-g file.in] [-p string] [-o path.prefix] "
            "-L string links.in [... links.in] [-L ...] -f fasta.in -d digest_sites.in\n", app );

    printf( "           -c n ... min amount of clusters\n" );
    printf( "           -l n ... minimum number of links (default %d)\n", DEF_ARG_L );
    printf( "           -m n ... MAPQ cutoff (default %d)\n", DEF_ARG_M );
    printf( "           -s n ... minimum sequence length (default %d)\n", DEF_ARG_S );

    printf( "           -g f ... guide scaffold\n" );
    printf( "           -w n ... guide scaffold window size\n" );
    printf( "           -p s ... scaffold names prefix\n" );

    printf( "           -o f ... output scaffolds to f\n" );
    printf( "           -L s f . digest site and associated links files\n" );

}

int main( int argc, char* argv[] )
{
    ScaffoldContext ctx;

    bzero( &ctx, sizeof( ctx ) );

    // process arguments

    ctx.normalize_cuts = 1;
    ctx.normalize_len = 1;
    ctx.normalize_midpoint = 1;

    ctx.mapq          = DEF_ARG_M;
    ctx.minlinks      = DEF_ARG_L;
    ctx.minseqlen     = DEF_ARG_S;
    ctx.minclusters   = DEF_ARG_C;
    ctx.rescaffold_wnd_size = DEF_ARG_W;
    ctx.seqnameprefix = NULL;
    ctx.path_output_prefix    = NULL;

    char* path_fasta     = NULL;
    char* path_cut_sites = NULL;
    char* path_guide = NULL;

    opterr = 0;

    char** links    = calloc( argc * 2, sizeof( char* ) );
    uint64_t nlinks = 0;

    int c;
    while ( ( c = getopt( argc, argv, "c:d:f:g:l:L:m:o:p:s:w:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'c':
                ctx.minclusters = atoi( optarg );
                break;

            case 'd':
                path_cut_sites = optarg;
                break;

            case 'f':
                path_fasta = optarg;
                break;

            case 'g':
                path_guide = optarg;
                break;

            case 'l':
                ctx.minlinks = atoi( optarg );

                if ( ctx.minlinks < 1 )
                {
                    fprintf( stderr, "error: invalid min number of links between contigs\n" );
                    exit( 1 );
                }
                break;

            case 'L':
                links[ nlinks ] = optarg;
                nlinks += 1;

                while ( argv[ optind ] && argv[ optind ][ 0 ] != '-' )
                {
                    links[ nlinks ] = argv[ optind ];
                    nlinks += 1;

                    optind += 1;
                }

                links[ nlinks ] = NULL;
                nlinks += 1;

                break;

            case 'm':
                ctx.mapq = atoi( optarg );

                if ( ctx.mapq < 1 )
                {
                    fprintf( stderr, "error: invalid mapq cutoff\n" );
                    exit( 1 );
                }

                break;

            case 'p':
                ctx.seqnameprefix = optarg;
                break;

            case 'o':
                ctx.path_output_prefix = optarg;
                break;

            case 's':
                ctx.minseqlen = atoi( optarg );

                if ( ctx.minseqlen < 1 )
                {
                    fprintf( stderr, "error: invalid min sequence length\n" );
                    exit( 1 );
                }

                break;

            case 'w':
                ctx.rescaffold_wnd_size = atoi( optarg );
                break ;

            default:
                fprintf( stderr, "malformed command line arguments\n" );
                usage( argv[ 0 ] );
                exit( 1 );
        }
    }

    if ( !nlinks )
    {
        fprintf( stderr, "no files containing links provided\n" );
        usage( argv[ 0 ] );
        exit( 1 );
    }

    if ( !path_fasta || !path_cut_sites )
    {
        fprintf( stderr, "missing mandatory arguments\n" );
        usage( argv[ 0 ] );
        exit( 1 );
    }

    if ( !process_fasta( &ctx, path_fasta ) )
    {
        exit( 1 );
    }

    if ( path_guide && !process_guide(&ctx, path_guide, 1) )
    {
            return 1;
    }

    ctx.mapping = malloc( sizeof( uint64_t ) * ctx.maxlen );

    if ( !process_cut_sites( &ctx, path_cut_sites ) )
    {
        exit( 1 );
    }

    char* cutter         = links[ 0 ];
    uint64_t nlinks_prev = ctx.nlinks;

    uint64_t i = 1;
    while ( 1 )
    {
        if ( !links[ i ] )
        {
            if ( i + 1 < nlinks )
            {
                if (ctx.normalize_cuts)
                {
                    remap_links( &ctx, cutter_to_id( &ctx, cutter ), nlinks_prev );
                }

                cutter = links[ i + 1 ];
                i += 2;
            }
            else
            {
                break;
            }
        }
        else
        {
            nlinks_prev = ctx.nlinks;
            if ( process_links( &ctx, links[ i ], cutter, 0 ) == 0 )
            {
                exit( 1 );
            }

            i += 1;
        }
    }

    if (ctx.normalize_cuts)
    {
        remap_links( &ctx, cutter_to_id( &ctx, cutter ), nlinks_prev );
    }

    links_sort( &ctx, 0 );

    if (ctx.normalize_len)
    {
        // TODO: adapt binsize to sequence length

        compute_effective_length( &ctx, 10000, 2 );     // TODO: hardcoded
    }

    if ( ctx.normalize_midpoint )
    {
        compute_midpoint( &ctx, 0 );
    }

    filter_links( &ctx, 1 );

    if (ctx.normalize_cuts && ctx.normalize_len)
    {
        compute_effective_length_cuts( &ctx );
    }

    build_edges( &ctx );

    if (path_guide)
    {
        rescaffold( &ctx );
    }
    else
    {
        scaffold( &ctx );
    }

    // calls free() on each key
    hashmap_close( &( ctx.mapseqname ) );

    for ( i = 0; i < ctx.nsgnodes; i++ )
    {
        free( ctx.sgnodes[ i ].seqname );
    }

    free( ctx.sgnodes );
    free( ctx.offsets );
    free( ctx.links );
    free( ctx.sgedges_sorted );
    free( ctx.sgedges );

    for ( i = 0; i < ctx.ncutters; i++ )
    {
        free( ctx.cutters[ i ] );
    }

    free( ctx.cutters );
    free( ctx.cut_offsets );
    free( ctx.cut_points );
    free( ctx.mapping );

    free( ctx.matrix_self );
    free( ctx.vector_nself );

    return 0;
}
