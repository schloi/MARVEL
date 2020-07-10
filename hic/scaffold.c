
#include "graphviz.h"
#include "hashmap.h"

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
#include <unistd.h>

// command line defaults

#define DEF_ARG_M 1
#define DEF_ARG_L 10
#define DEF_ARG_S 20000
#define DEF_ARG_C 1

// sequence status flags

#define SFLAG_CONTAINED ( 0x1 << 0 )
#define SFLAG_FORBIDDEN ( 0x1 << 1 )

// bit masks

/*
#define U64_HIGHEST ( (uint64_t)(1) << 63 )
#define U64_HIGHEST_SET(n)  ( (uint64_t)(n) | U64_HIGHEST )
#define U64_HIGHEST_TEST(n) ( (uint64_t)(n) >> 63 )
#define U64_HIGHEST_CLEAR(n) ( (uint64_t)(n) & ~U64_HIGHEST )
*/

// debug toggles

#undef DEBUG_DUMP_ITERATIONS
#undef DEBUG_DUMP_IDS
#define DEBUG_DUMP_LINKS_BEFORE_SCAFFOLDING

// other toggles

#define THREADED_SCAFFOLDING

// #define MIN(x,y) ((x)<(y)?(x):(y))
// #define MAX(x,y) ((x)>(y)?(x):(y))

// maintains the state of the app

typedef struct
{
    // command line arguments

    uint16_t mapq;      // MAPQ threshold
    uint16_t minlinks;  // minimum number of links between contigs
    uint16_t minseqlen; // minimum sequence length
    char* seqnameprefix;
    uint16_t minclusters;
    uint16_t nthreads;

    char** seqname;

    Hashmap mapseqname;

    FILE* fout_mergetree;

    // sequences and their links

    uint8_t* seqstatus; // sequence status flags
    uint64_t* seqlen;   // sequence lengths
    uint64_t* midpoint;     // midpoint which divides the links in two equally sized sets
    uint64_t* efflen;    // effective sequence length (ie. regions that contain links)

    int32_t* seqsource;
    uint32_t nseq; // number of sequences
    uint32_t maxseq;

    uint64_t* offsets; // index of a sequence's links
    uint32_t maxoffsets;

    uint64_t* links; // sequence links array of (seqid_1, seqid_2, pos_1, pos_2)
    uint64_t maxlinks;
    uint64_t nlinks;
} ScaffoldContext;

typedef struct
{
    uint32_t id1;
    uint32_t id2;
    uint8_t type;
    double score;
    double confidence;
} LinkScore;

typedef struct
{
    uint8_t* seqstatus;
    uint32_t id1;
    uint32_t id2;
    uint64_t* offsets;
    uint64_t* links;
    uint64_t idx_beg;
    uint64_t idx_end;

    uint64_t len1;
    uint64_t len2;
    uint32_t idnew;
    uint32_t type;
} ScaffoldSequenceThread;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

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

static int AlmostEqualRelative( double A, double B )
{
    double maxRelDiff = DBL_EPSILON;

    // Calculate the difference.
    double diff = fabs( A - B );
    A           = fabs( A );
    B           = fabs( B );
    // Find the largest
    double largest = ( B > A ) ? B : A;

    if ( diff <= largest * maxRelDiff )
        return 1;

    return 0;
}

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

// comparison function for uint64_t[4]

static int cmp_links4( const void* x, const void* y )
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

static int cmp_uint64_2( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    int i;
    for ( i = 0; i < 2; i++ )
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
static int cmp_double( const void* x, const void* y )
{
    double a = *( (double*)x );
    double b = *( (double*)y );

    if ( a < b )
    {
        return -1;
    }
    else if ( a > b )
    {
        return 1;
    }

    return 0;
}

static int cmp_link_confidence_desc( const void* x, const void* y )
{
    double a = ( (LinkScore*)x )->confidence;
    double b = ( (LinkScore*)y )->confidence;

    if ( a < b )
    {
        return 1;
    }
    else if ( a > b )
    {
        return -1;
    }

    return 0;
}

static int process_fasta( ScaffoldContext* ctx, const char* path_fasta )
{
    printf( "processing fasta\n" );

    FILE* fin;
    char* line;
    int32_t nseq      = 0;
    char* seqname     = NULL;
    uint64_t seqlen   = 0;
    char** seqnames   = NULL;
    uint64_t* seqlens = NULL;
    int32_t seqmax    = 0;

    // look for <path_fasta>.fai

    char path_lengths[ PATH_MAX ];
    sprintf( path_lengths, "%s.fai", path_fasta );

    if ( ( fin = fopen( path_lengths, "r" ) ) )
    {
        while ( ( line = fgetln( fin, NULL ) ) )
        {
            // <name>\t<length>\t...\n

            char* num = strchr( line, '\t' );
            *num      = '\0';

            seqlen = atoi( num + 1 );

            if ( nseq + 1 >= seqmax )
            {
                seqmax   = seqmax * 2 + 100;
                seqnames = realloc( seqnames, sizeof( char** ) * seqmax );
                seqlens  = realloc( seqlens, sizeof( uint64_t ) * seqmax );
            }

            seqlens[ nseq ]  = seqlen;
            seqnames[ nseq ] = strdup( line );
            nseq += 1;
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
                    if ( nseq + 1 >= seqmax )
                    {
                        seqmax   = seqmax * 2 + 100;
                        seqnames = realloc( seqnames, sizeof( char** ) * seqmax );
                        seqlens  = realloc( seqlens, sizeof( uint64_t ) * seqmax );
                    }

                    seqlens[ nseq ]  = seqlen;
                    seqnames[ nseq ] = seqname;
                    nseq += 1;
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
            seqlens[ nseq ]  = seqlen;
            seqnames[ nseq ] = seqname;
            nseq += 1;
        }
    }

    ctx->nseq   = nseq;
    ctx->maxseq = seqmax;
    ctx->seqname = seqnames;
    ctx->seqlen  = seqlens;
    ctx->midpoint = calloc( seqmax, sizeof(uint64_t) );
    ctx->efflen = calloc( seqmax, sizeof(uint64_t) );
    ctx->seqstatus = calloc( seqmax, sizeof( uint8_t ) );
    ctx->seqsource = calloc( seqmax * 2, sizeof( int32_t ) );

    hashmap_open( &( ctx->mapseqname ), nseq );

    int32_t i;
    for ( i = 0; i < nseq; i++ )
    {
        ctx->midpoint[i] = seqlens[i] / 2;
        hashmap_put( &( ctx->mapseqname ), seqnames[ i ], seqnames[ i ] + strlen( seqnames[ i ] ), &i, (void*)( &i ) + sizeof( int32_t ) );
    }

    printf( "  %d sequences\n", nseq );

    return 1;
}

static void write_nself_links( ScaffoldContext* ctx, const char* pathout )
{
    uint64_t* links    = ctx->links;
    uint64_t nlinks    = ctx->nlinks;
    char** seqname     = ctx->seqname;
    uint8_t* seqstatus = ctx->seqstatus;
    uint64_t* seqlen = ctx->seqlen;
    uint64_t* midpoint = ctx->midpoint;

    FILE* fout = fopen( pathout, "w" );

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1  = links[ i + 0 ];
        uint64_t id2  = links[ i + 1 ];

        if ( id1 == id2 )
        {
            continue;
        }

        uint64_t pos1 = links[ i + 2 ];
        uint64_t pos2 = links[ i + 3 ];

        uint64_t len1 = seqlen[id1];
        uint64_t len2 = seqlen[id2];

        if ( ( seqstatus[ id1 ] & SFLAG_CONTAINED ) || ( seqstatus[ id2 ] & SFLAG_CONTAINED ) )
        {
            continue;
        }

        fprintf( fout, "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%c\n",
                    id1, id2, pos1, pos2 < midpoint[id2] ? 'l' : 'r' );

        fprintf( fout, "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%c\n",
                    id2, id1, pos2, pos1 < midpoint[id1] ? 'l' : 'r' );
    }

    fclose( fout );
}

static void write_links( ScaffoldContext* ctx, const char* pathout )
{
    uint64_t* links    = ctx->links;
    uint64_t nlinks    = ctx->nlinks;
    char** seqname     = ctx->seqname;
    uint8_t* seqstatus = ctx->seqstatus;

    FILE* fout = fopen( pathout, "w" );

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1  = links[ i + 0 ];
        uint64_t id2  = links[ i + 1 ];
        uint64_t pos1 = links[ i + 2 ];
        uint64_t pos2 = links[ i + 3 ];

        if ( ( seqstatus[ id1 ] & SFLAG_CONTAINED ) || ( seqstatus[ id2 ] & SFLAG_CONTAINED ) )
        {
            continue;
        }

        fprintf( fout, "%" PRIu64 " %s %" PRIu64 " %s %" PRIu64 " %" PRIu64 "\n",
                    id1, seqname[id1], id2, seqname[id2], pos1, pos2 );
    }

    fclose( fout );
}

static void write_link_stats( ScaffoldContext* ctx, const char* pathout )
{
    uint64_t* links    = ctx->links;
    uint64_t nlinks    = ctx->nlinks;
    char** seqname     = ctx->seqname;
    uint64_t* seqlen   = ctx->seqlen;
    uint8_t* seqstatus = ctx->seqstatus;
    uint64_t* midpoint = ctx->midpoint;

    FILE* fout = fopen( pathout, "w" );

    uint64_t i;
    uint64_t prev     = 0;
    uint64_t id1_prev = links[ 0 ];
    uint64_t id2_prev = links[ 1 ];
    uint64_t l1, r1, l2, r2;
    l1 = r1 = l2 = r2 = 0;

    for ( i = 4; i < nlinks; i += 4 )
    {
        uint64_t id1  = links[ i + 0 ];
        uint64_t id2  = links[ i + 1 ];
        uint64_t pos1 = links[ i + 2 ];
        uint64_t pos2 = links[ i + 3 ];

        if ( ( seqstatus[ id1 ] & SFLAG_CONTAINED ) || ( seqstatus[ id2 ] & SFLAG_CONTAINED ) )
        {
            continue;
        }

        if ( pos1 < midpoint[id1] )
        {
            l1 += 1;
        }
        else
        {
            r1 += 1;
        }

        if ( pos2 < midpoint[id2] )
        {
            l2 += 1;
        }
        else
        {
            r2 += 1;
        }

        if ( id1 != id1_prev || id2 != id2_prev )
        {
            fprintf( fout, "%" PRIu64 " %s %" PRIu64 " %" PRIu64 " %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                     id1_prev, seqname[ id1_prev ], seqlen[ id1_prev ],
                     id2_prev, seqname[ id2_prev ], seqlen[ id2_prev ],
                     l1, r1, l2, r2 );

            id1_prev = id1;
            id2_prev = id2;
            prev     = i;
            l1 = r1 = l2 = r2 = 0;
        }
    }

    fprintf( fout, "%" PRIu64 " %s %" PRIu64 " %" PRIu64 " %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
             id1_prev, seqname[ id1_prev ], seqlen[ id1_prev ],
             id2_prev, seqname[ id2_prev ], seqlen[ id2_prev ],
             l1, r1, l2, r2 );

    fclose( fout );
}

static int process_links( ScaffoldContext* ctx, const char* path_links )
{
    printf( "processing links\n" );

    FILE* fin;

    if ( !( fin = fopen( path_links, "r" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_links );
        return 0;
    }

    uint32_t nseq     = ctx->nseq;
    uint64_t* links   = NULL;
    uint64_t maxlinks = 0;
    uint64_t nlinks   = 0;
    uint16_t mapq     = ctx->mapq;
    uint64_t nline    = 0;
    uint64_t* seqlen  = ctx->seqlen;

    char* line;

    while ( ( line = fgetln( fin, NULL ) ) )
    {
        nline += 1;

        int lenline = strlen( line );

        if (lenline < 2 )
        {
            continue;
        }

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

        while ( skip == 0 && ( token = strsep( &next, " " ) ) != NULL )
        {
            switch ( col )
            {
                case 0:
                    id1 = name_to_id( ctx, token );
                    if ( id1 == -1 )
                    {
                        printf( "unknown sequence %s\n", token );
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
                        else if ( id1 == id2 )
                        {
                            // skip = 1;
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

        if ( id1 > id2 )
        {
            uint64_t temp;

            temp = id2;
            id2  = id1;
            id1  = temp;

            temp = pos2;
            pos2 = pos1;
            pos1 = temp;
        }

        if ( pos1 >= seqlen[ id1 ] )
        {
            if ( pos1 - 100 < seqlen[ id1 ] )
            {
                pos1 -= 100;
            }
            else
            {
                printf( "line %" PRIu64 " mapping position %" PRIu64 " beyond contig length %" PRIu64 "\n",
                        nline, pos1, seqlen[ id1 ] );
                continue;
            }
        }

        if ( pos2 >= seqlen[ id2 ] )
        {
            if ( pos2 - 100 < seqlen[ id2 ] )
            {
                pos2 -= 100;
            }
            else
            {
                printf( "line %" PRIu64 " mapping position %" PRIu64 " beyond contig length %" PRIu64 "\n",
                        nline, pos2, seqlen[ id2 ] );
                continue;
            }
        }

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

        /*
        links[ nlinks ]     = id2;
        links[ nlinks + 1 ] = id1;
        links[ nlinks + 2 ] = pos2;
        links[ nlinks + 3 ] = pos1;

        nlinks += 4;
        */
    }

    printf( "  sorting %" PRIu64 " links\n", nlinks / 4 );

    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_links4 );

    uint32_t i;
    uint32_t discard  = 0;
    uint32_t cur      = 0;
    uint32_t minlinks = ctx->minlinks;

    if ( minlinks > 1 )
    {
        for ( i = 0; i < nlinks; )
        {
            cur = i + 4;
            while ( cur < nlinks && links[ i ] == links[ cur ] && links[ i + 1 ] == links[ cur + 1 ] )
            {
                cur += 4;
            }

            if ( ( cur - i ) / 4 < minlinks )
            {
                discard += ( cur - i );

                uint32_t j;
                for ( j = i; j < cur; j++ )
                {
                    links[ j ] = 0;
                }
            }

            i = cur;
        }

        qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_links4 );

        printf( "  discarded %d links\n", discard / 4 );

        nlinks -= discard;
        memmove( links, links + discard, nlinks * sizeof( uint64_t ) );
    }

    printf( "  computing offsets\n" );

    uint64_t* offsets = malloc( sizeof( uint64_t ) * ( nseq + 1 ) );
    bzero( offsets, sizeof( uint64_t ) * ( nseq + 1 ) );

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

    fclose( fin );

    ctx->maxoffsets = nseq + 1;
    ctx->offsets    = offsets;
    ctx->maxlinks   = maxlinks;
    ctx->links      = links;
    ctx->nlinks     = nlinks;

    return 1;
}

static void validate_offsets( ScaffoldContext* ctx )
{
    printf( "validate_offsets\n" );

    uint64_t* offsets = ctx->offsets;
    uint32_t nseq     = ctx->nseq;
    uint64_t* links   = ctx->links;

    uint64_t i;
    for ( i = 0; i < nseq; i++ )
    {
        uint64_t beg = offsets[ i ];
        uint64_t end = offsets[ i + 1 ];

        while ( beg < end )
        {
            assert( links[ beg ] == i );
            beg += 4;
        }
    }
}

static void remove_dupe_links( ScaffoldContext* ctx )
{
    // links must be sorted

    uint64_t* offsets  = ctx->offsets;
    uint32_t nseq      = ctx->nseq;
    uint64_t nlinks    = ctx->nlinks;
    uint64_t* links    = ctx->links;

    if ( ctx->nlinks == 0 )
    {
        return ;
    }

    bzero( offsets, sizeof( uint64_t ) * ( nseq + 1 ) );

    uint64_t i;
    uint64_t ncur = 0;

    offsets[ links[0] ] += 4;

    for ( i = 4; i < nlinks; i += 4 )
    {
        if ( memcmp(links + i, links + ncur, sizeof(uint64_t) * 4) != 0 )
        {
            ncur += 4;
            offsets[ links[i] ] += 4;

            if ( ncur != i )
            {
                memcpy( links + ncur, links + i, sizeof( uint64_t ) * 4 );
            }
        }
    }

    ncur += 4;

    printf("removed %" PRIu64 " dupe links\n", (nlinks - ncur) / 4);

    ctx->nlinks = ncur;
    ctx->links = realloc(ctx->links, sizeof(uint64_t) * ncur);

    uint64_t off = 0;
    uint64_t coff;

    for ( i = 0; i <= nseq; i++ )
    {
        coff         = offsets[ i ];
        offsets[ i ] = off;
        off += coff;
    }
}


static void purge_links( ScaffoldContext* ctx )
{
    uint64_t* offsets  = ctx->offsets;
    uint32_t nseq      = ctx->nseq;
    uint64_t nlinks    = ctx->nlinks;
    uint64_t* links    = ctx->links;
    uint8_t* seqstatus = ctx->seqstatus;

    uint32_t ncur = 0;

    bzero( offsets, sizeof( uint64_t ) * ( nseq + 1 ) );

    uint64_t zeroes[] = {0, 0, 0, 0};

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1 = links[ i ];
        uint64_t id2 = links[ i + 1 ];

        if ( memcmp(links + i, zeroes, sizeof(uint64_t) * 4) == 0
            || ( seqstatus[ id1 ] & SFLAG_CONTAINED )
            || ( seqstatus[ id2 ] & SFLAG_CONTAINED ) )
        {
            continue;
        }

        offsets[ id1 ] += 4;

        if ( i != ncur )
        {
            memcpy( links + ncur, links + i, sizeof( uint64_t ) * 4 );
        }

        ncur += 4;
    }

    printf("purged %" PRIu64 " links\n", (ctx->nlinks - ncur) / 4);

    ctx->nlinks = ncur;
    ctx->links = realloc(ctx->links, sizeof(uint64_t) * ncur);

    uint64_t off = 0;
    uint64_t coff;

    uint64_t j;
    for ( j = 0; j <= nseq; j++ )
    {
        coff         = offsets[ j ];
        offsets[ j ] = off;
        off += coff;
    }
}

static void compute_midpoint( ScaffoldContext* ctx )
{
    uint64_t* offsets = ctx->offsets;
    uint64_t* links = ctx->links;
    uint64_t* seqlen = ctx->seqlen;
    uint64_t* midpoint = ctx->midpoint;
    uint64_t* efflen = ctx->efflen;
    uint32_t nseq = ctx->nseq;
    uint64_t i;

    uint64_t maxlen = 0;
    for ( i = 0; i < nseq; i++ )
    {
        if ( maxlen < seqlen[ i ] )
        {
            maxlen = seqlen[ i ];
        }
    }

    uint64_t maxtemp = 1000;
    uint64_t curtemp = 0;
    uint64_t* temp = malloc(sizeof(uint64_t) * maxtemp);

    uint64_t beg = offsets[0];
    uint64_t end = offsets[nseq];
    while ( beg < end )
    {
        uint64_t id1 = links[beg];
        uint64_t id2 = links[beg+1];
        uint64_t pos1 = links[beg+2];
        uint64_t pos2 = links[beg+3];

        if ( curtemp + 4 >= maxtemp )
        {
            maxtemp = maxtemp * 1.2 + 1000;
            temp = realloc(temp, sizeof(uint64_t) * maxtemp);
        }

        temp[curtemp] = id1;
        temp[curtemp+1] = pos1;
        temp[curtemp+2] = id2;
        temp[curtemp+3] = pos2;
        curtemp += 4;

        beg += 4;
    }

    qsort(temp, curtemp / 2, sizeof(uint64_t) * 2, cmp_uint64_2);

    uint64_t previd = temp[0];
    uint64_t previdx = 0;
    uint64_t len = seqlen[previd];
    uint32_t binsize = 10000;
    uint32_t* bins = calloc( (maxlen + binsize - 1) / binsize, sizeof(uint32_t) );

    for ( i = 2; i < curtemp; i+=2)
    {
        // printf("%llu %llu %llu\n", i, temp[i], temp[i+1]);

        if ( temp[i] != previd )
        {
            uint64_t mididx = 2 * ( ( i - previdx ) / 4 );
            uint64_t mid = temp[ previdx + mididx + 1 ];

            midpoint[ previd ] = mid;

            printf("seq %llu len %llu mid %llu", previd, seqlen[previd], mid);

            uint64_t j;
            uint64_t seqefflen = len;
            for (j = 0; j < (len + binsize - 1) / binsize; j++)
            {
                if ( bins[j] < 1 )
                {
                    seqefflen -= binsize;
                }
            }

            efflen[ previd ] = seqefflen;

            printf(" len %llu efflen %llu\n", len, seqefflen);

            previdx = i;
            previd = temp[i];
            len = seqlen[ previd ];
            bzero(bins, sizeof(uint32_t) * ( ( len + binsize - 1) / binsize) );
        }

        bins[ temp[i+1] / binsize ] += 1;
    }

    uint64_t mididx = 2 * ( ( i - previdx ) / 4 );
    uint64_t mid = temp[ previdx + mididx + 1 ];

    midpoint[ previd ] = mid;

    printf("seq %llu len %llu mid %llu", previd, seqlen[previd], mididx);
    uint64_t j;
    uint64_t seqefflen = len;
    for (j = 0; j < (len + binsize - 1) / binsize; j++)
    {
        if ( bins[j] < 2 )
        {
            seqefflen -= binsize;
        }
    }

    efflen[previd] = seqefflen;

    printf(" len %llu efflen %llu\n", len, seqefflen);

    free(temp);
}

static void write_scaffolds_rec( ScaffoldContext* ctx, int32_t sid,
                                 int32_t** scaf, uint32_t* maxscaf, uint32_t* nscaf )
{
    int32_t* _scaf    = *scaf;
    uint32_t _maxscaf = *maxscaf;
    uint32_t _nscaf   = *nscaf;

    uint32_t _sid = abs( sid );

    static int indent = 1;
    printf( "DBG OUTPUT %*s %s\n", indent, " ", ctx->seqname[ _sid ] );
    indent += 1;

    if ( ctx->seqsource[ _sid * 2 ] == ctx->seqsource[ _sid * 2 + 1 ] )
    {
        if ( _nscaf + 1 >= _maxscaf )
        {
            _maxscaf = _maxscaf * 1.2 + 100;
            _scaf    = realloc( _scaf, sizeof( int32_t ) * _maxscaf );
        }

        _scaf[ _nscaf ] = sid;
        _nscaf += 1;
    }
    else
    {
        write_scaffolds_rec( ctx, ctx->seqsource[ _sid * 2 ], &_scaf, &_maxscaf, &_nscaf );

        write_scaffolds_rec( ctx, ctx->seqsource[ _sid * 2 + 1 ], &_scaf, &_maxscaf, &_nscaf );
    }

    if ( sid < 0 )
    {
        uint32_t beg = *nscaf;
        uint32_t end = _nscaf - 1;

        if ( beg < end )
        {
            printf( "FLIP     %*s %s %d %d\n", indent, " ", ctx->seqname[ _sid ], beg, end );

            while ( beg < end )
            {
                int32_t t    = _scaf[ beg ];
                _scaf[ beg ] = ( -1 ) * _scaf[ end ];
                _scaf[ end ] = ( -1 ) * t;

                beg += 1;
                end -= 1;
            }

            if ( beg == end )
            {
                _scaf[ beg ] = ( -1 ) * _scaf[ beg ];
            }
        }
    }

    *scaf    = _scaf;
    *maxscaf = _maxscaf;
    *nscaf   = _nscaf;

    indent -= 1;
}

static uint32_t count_scaffolds( ScaffoldContext* ctx )
{
    uint32_t n = 0;

    uint32_t i;
    for ( i = 0; i < ctx->nseq; i++ )
    {
        if ( ctx->seqstatus[ i ] & SFLAG_CONTAINED )
        {
            continue;
        }

        // source = scaffold
        if ( ctx->seqsource[ i * 2 ] != ctx->seqsource[ i * 2 + 1 ] )
        {
            n += 1;
        }
        // else = singleton
    }

    return n;
}

static void write_scaffolds( ScaffoldContext* ctx, FILE* fout, int singletons )
{
    uint32_t i;

    int32_t* scaf    = NULL;
    uint32_t maxscaf = 0;
    uint32_t nscaf   = 0;

    for ( i = 0; i < ctx->nseq; i++ )
    {
        if ( ctx->seqstatus[ i ] & SFLAG_CONTAINED )
        {
            continue;
        }

        nscaf = 0;

        // no source = singleton
        if ( ctx->seqsource[ i * 2 ] == ctx->seqsource[ i * 2 + 1 ] )
        {
            if ( singletons )
            {
#ifdef DEBUG_DUMP_IDS
                fprintf( fout, ">%d length=%" PRIu64 "\n", i, ctx->seqlen[ i ] );
#else
                fprintf( fout, ">%s length=%" PRIu64 "\n", ctx->seqname[ i ], ctx->seqlen[ i ] );
#endif
            }
        }
        // scaffold
        else
        {
            printf( "DBG OUTPUT %s\n", ctx->seqname[ i ] );

            write_scaffolds_rec( ctx, ctx->seqsource[ i * 2 ], &scaf, &maxscaf, &nscaf );
            write_scaffolds_rec( ctx, ctx->seqsource[ i * 2 + 1 ], &scaf, &maxscaf, &nscaf );

            uint32_t j;
            uint32_t total = 0;

#ifdef DEBUG_DUMP_IDS
            fprintf( fout, ">%d length=%" PRIu64 "\n", i, ctx->seqlen[ i ] );
#else
            fprintf( fout, ">%s length=%" PRIu64 "\n", ctx->seqname[ i ], ctx->seqlen[ i ] );
#endif

            for ( j = 0; j < nscaf; j++ )
            {
#ifdef DEBUG_DUMP_IDS
                fprintf( fout, "%d\n", scaf[ j ] );
#else
                fprintf( fout, "%s%s\n", scaf[ j ] < 0 ? "-" : "", ctx->seqname[ abs( scaf[ j ] ) ] );
#endif

                total += ctx->seqlen[ abs( scaf[ j ] ) ];
            }
        }
    }
}

void* scaffold_sequences_thread( void* data )
{
    ScaffoldSequenceThread* sst = data;

    uint32_t id1       = sst->id1;
    uint32_t id2       = sst->id2;
    uint64_t len1      = sst->len1;
    uint64_t len2      = sst->len2;
    uint32_t idnew     = sst->idnew;
    uint32_t type      = sst->type;
    uint64_t* links    = sst->links;
    uint8_t* seqstatus = sst->seqstatus;

    uint64_t beg = sst->idx_beg;
    uint64_t end = sst->idx_end;

    assert( ( beg % 4 ) == 0 );
    assert( ( end % 4 ) == 0 );

    for ( ; beg < end; beg += 4 )
    {
        uint64_t link_id1 = links[ beg + 0 ];
        uint64_t link_id2 = links[ beg + 1 ];
        uint64_t pos1     = links[ beg + 2 ];
        uint64_t pos2     = links[ beg + 3 ];

        // id1 -> x
        if ( link_id1 == id1 )
        {
            // ignore self-links
            if ( link_id2 != id2 )
            {
                links[ beg ] = idnew;

                if ( type == 0 )
                {
                    links[ beg + 2 ] = len1 - pos1;
                }
                else if ( type == 1 )
                {
                    links[ beg + 2 ] = len1 - pos1;
                }
                else
                {
                    links[ beg + 2 ] = pos1;
                }
            }
        }
        // id2 -> x
        else if ( link_id1 == id2 )
        {
            if ( link_id2 != id1 )
            {
                links[ beg ] = idnew;

                if ( type == 0 )
                {
                    links[ beg + 2 ] = len1 + pos1;
                }
                else if ( type == 1 )
                {
                    links[ beg + 2 ] = len1 + ( len2 - pos1 );
                }
                else if ( type == 2 )
                {
                    links[ beg + 2 ] = len1 + pos1;
                }
                else if ( type == 3 )
                {
                    links[ beg + 2 ] = len1 + ( len2 - pos1 );
                }
            }
        }
        // x -> id1/id2
        else if ( !( seqstatus[ link_id1 ] & SFLAG_CONTAINED ) )
        {
            if ( link_id2 == id1 )
            {
                links[ beg + 1 ] = idnew;

                if ( type == 0 )
                {
                    links[ beg + 3 ] = len1 - pos2;
                }
                else if ( type == 1 )
                {
                    links[ beg + 3 ] = len1 - pos2;
                }
            }
            else if ( link_id2 == id2 )
            {
                links[ beg + 1 ] = idnew;

                if ( type == 0 )
                {
                    links[ beg + 3 ] = len1 + pos2;
                }
                else if ( type == 1 )
                {
                    links[ beg + 3 ] = len1 + ( len2 - pos2 );
                }
                else if ( type == 2 )
                {
                    links[ beg + 3 ] = len1 + pos2;
                }
                else if ( type == 3 )
                {
                    links[ beg + 3 ] = len1 + ( len2 - pos2 );
                }
            }
        }
    }

    return NULL;
}

uint32_t scaffold_sequences( ScaffoldContext* ctx, uint32_t id1, uint32_t id2, uint32_t type )
{
    static uint32_t nmergers = 0;

    uint64_t* offsets = ctx->offsets;
    uint64_t* links   = ctx->links;
    uint64_t* seqlen  = ctx->seqlen;
    uint32_t nseq     = ctx->nseq;
    uint8_t* seqstatus = ctx->seqstatus;
    int32_t* seqsource = ctx->seqsource;
    uint64_t* midpoint = ctx->midpoint;
    uint64_t* efflen = ctx->efflen;

    uint32_t idnew = nseq;
    nseq += 1;

    if ( nseq >= ctx->maxseq )
    {
        ctx->maxseq = ctx->maxseq * 1.2 + 100;
        seqlen = ctx->seqlen = realloc( ctx->seqlen, sizeof( uint64_t ) * ctx->maxseq );
        midpoint = ctx->midpoint = realloc( ctx->midpoint, sizeof( uint64_t ) * ctx->maxseq );
        efflen = ctx->efflen = realloc(ctx->efflen, sizeof(uint64_t) * ctx->maxseq);
        seqstatus = ctx->seqstatus = realloc( ctx->seqstatus, sizeof( uint8_t ) * ctx->maxseq );
        seqsource = ctx->seqsource = realloc( ctx->seqsource, sizeof( int32_t ) * ctx->maxseq * 2 );
        ctx->seqname               = realloc( ctx->seqname, sizeof( char* ) * ctx->maxseq );
    }

    if ( nseq + 1 >= ctx->maxoffsets )
    {
        ctx->maxoffsets = ctx->maxoffsets * 1.2 + 100;
        offsets = ctx->offsets = realloc( ctx->offsets, sizeof( uint64_t ) * ctx->maxoffsets );
    }

#ifdef DEBUG_DUMP_LINKS_BEFORE_SCAFFOLDING
    {
        char tmp[ 1024 ];
        sprintf( tmp, "links.%d.summary", nmergers );
        write_link_stats( ctx, tmp );

        sprintf( tmp, "links.%d.fromto", nmergers);
        write_links(ctx, tmp);

        sprintf( tmp, "links.%d.from", nmergers);
        write_nself_links(ctx, tmp);
    }
#endif

    seqlen[ idnew ] = seqlen[ id1 ] + seqlen[ id2 ];
    midpoint[ idnew ] = seqlen[ idnew ] / 2;
    efflen[idnew] = efflen[id1] + efflen[id2];
    seqstatus[ id1 ] |= SFLAG_CONTAINED;
    seqstatus[ id2 ] |= SFLAG_CONTAINED;
    seqstatus[ idnew ] = 0;

    if ( ctx->seqnameprefix )
    {
        ctx->seqname[ idnew ] = malloc( strlen( ctx->seqnameprefix ) + 32 );
        sprintf( ctx->seqname[ idnew ], "%s_scaffold_%d", ctx->seqnameprefix, idnew );
    }
    else
    {
        ctx->seqname[ idnew ] = malloc( 32 );
        sprintf( ctx->seqname[ idnew ], "scaffold_%d", idnew );
    }

    if ( ctx->fout_mergetree )
    {

        nmergers += 1;

        fprintf( ctx->fout_mergetree, "\"%s\" [label=\"%d\n%d\n%s\"];\n", ctx->seqname[ idnew ], nmergers, type, ctx->seqname[ idnew ] );

        char* label;

        if ( ctx->seqsource[ id1 * 2 ] == ctx->seqsource[ id1 * 2 + 1 ] )
        {
            label = graphviz_wrap_string( ctx->seqname[ id1 ] );
            fprintf( ctx->fout_mergetree, "\"%s\" [label=\"%d\n%s\" color=blue];\n", ctx->seqname[ id1 ], id1, label );
        }

        if ( ctx->seqsource[ id2 * 2 ] == ctx->seqsource[ id2 * 2 + 1 ] )
        {
            label = graphviz_wrap_string( ctx->seqname[ id2 ] );
            fprintf( ctx->fout_mergetree, "\"%s\" [label=\"%d\n%s\" color=blue];\n", ctx->seqname[ id2 ], id2, label );
        }

        char* color;
        if ( type == 0 || type == 1 )
        {
            color = "red";
        }
        else
        {
            color = "black";
        }

        fprintf( ctx->fout_mergetree, "\"%s\" -> \"%s\" [label=A,color=%s];\n", ctx->seqname[ idnew ], ctx->seqname[ id1 ], color );

        if ( type == 1 || type == 3 )
        {
            color = "red";
        }
        else
        {
            color = "black";
        }

        fprintf( ctx->fout_mergetree, "\"%s\" -> \"%s\" [label=B,color=%s];\n", ctx->seqname[ idnew ], ctx->seqname[ id2 ], color );

        // fprintf(ctx->fout_mergetree, "%d %s %s %s\n", type, ctx->seqname[ id1 ], ctx->seqname[ id2 ], ctx->seqname[ idnew ]);
    }

    if ( type == 0 || type == 1 )
    {
        seqsource[ idnew * 2 ] = ( -1 ) * (int32_t)id1;
    }
    else
    {
        seqsource[ idnew * 2 ] = id1;
    }

    if ( type == 1 || type == 3 )
    {
        seqsource[ idnew * 2 + 1 ] = ( -1 ) * (int32_t)id2;
    }
    else
    {
        seqsource[ idnew * 2 + 1 ] = id2;
    }

    ctx->offsets[ idnew + 1 ] = ctx->offsets[ idnew ];
    // uint32_t curlink          = ctx->offsets[ idnew ];

    uint64_t len1 = seqlen[ id1 ];
    uint64_t len2 = seqlen[ id2 ];

#ifdef THREADED_SCAFFOLDING
    pthread_t threads[ ctx->nthreads ];
    ScaffoldSequenceThread threaddata[ ctx->nthreads ];
    uint64_t nlinks = ctx->nlinks;

    uint64_t incr = nlinks / ctx->nthreads + ctx->nthreads;
    incr += 4 - ( incr % 4 );

    uint64_t beg = 0;
    uint64_t end = 0;

    uint32_t thread;
    for ( thread = 0; thread < ctx->nthreads; thread++ )
    {
        beg = end;
        end = beg + incr;

        if ( end > nlinks || thread + 1 == ctx->nthreads )
        {
            end = nlinks;
        }

        threaddata[ thread ].seqstatus = seqstatus;
        threaddata[ thread ].id1       = id1;
        threaddata[ thread ].id2       = id2;
        threaddata[ thread ].offsets   = offsets;
        threaddata[ thread ].links     = links;
        threaddata[ thread ].idx_beg   = beg;
        threaddata[ thread ].idx_end   = end;
        threaddata[ thread ].len1      = len1;
        threaddata[ thread ].len2      = len2;
        threaddata[ thread ].idnew     = idnew;
        threaddata[ thread ].type      = type;

        pthread_create( threads + thread, NULL, scaffold_sequences_thread, threaddata + thread );
    }

    for ( thread = 0; thread < ctx->nthreads; thread++ )
    {
        pthread_join( threads[ thread ], NULL );
    }
#else

    ScaffoldSequenceThread sst;

    sst.seqstatus = seqstatus;
    sst.id1       = id1;
    sst.id2       = id2;
    sst.offsets   = offsets;
    sst.links     = links;
    sst.idx_beg   = 0;
    sst.idx_end   = ctx->nlinks;
    sst.len1      = len1;
    sst.len2      = len2;
    sst.idnew     = idnew;
    sst.type      = type;

    scaffold_sequences_thread( &sst );

#endif

    ctx->nseq = nseq;

    return idnew;
}

static uint32_t scaffold( ScaffoldContext* ctx )
{
    printf( "scaffolding %d sequences\n", ctx->nseq );

    uint64_t* offsets  = ctx->offsets;
    uint64_t* links    = ctx->links;
    uint64_t* seqlen   = ctx->seqlen;
    uint32_t nseq      = ctx->nseq;
    uint32_t minlinks  = ctx->minlinks;
    uint8_t* seqstatus = ctx->seqstatus;
    uint64_t* midpoint = ctx->midpoint;
    uint64_t* efflen = ctx->efflen;

    uint64_t i;
    uint32_t maxscores = nseq * 4;
    double* scores     = malloc( sizeof( double ) * maxscores );
    bzero( scores, sizeof( double ) * maxscores );
    uint64_t id1;

    uint32_t maxlscores = nseq;
    uint32_t curlscores = 0;
    LinkScore* lscores  = malloc( sizeof( LinkScore ) * maxlscores );

    printf( "  LR scores\n" );

    for ( id1 = 0; id1 < nseq; id1++ )
    {
        uint64_t beg  = offsets[ id1 ];
        uint64_t end  = offsets[ id1 + 1 ];
        uint32_t lr[] = {0, 0, 0, 0}; // LL / LR / RL / RR
        uint64_t len1 = seqlen[ id1 ];

        // printf("%s %d %d\n", ctx->seqname[id1], beg, end);

        if ( beg < end )
        {
            assert( links[ beg ] == id1 );
        }

        if ( seqstatus[ id1 ] & SFLAG_CONTAINED )
        {
            continue;
        }

        while ( beg < end )
        {
            uint64_t id2  = links[ beg + 1 ];
            uint64_t pos1 = links[ beg + 2 ];
            uint64_t pos2 = links[ beg + 3 ];
            uint64_t len2 = seqlen[ id2 ];

            if ( id1 == id2 )
            {
                beg += 4;
                continue;
            }

            if ( seqstatus[ id2 ] & SFLAG_CONTAINED )
            {
                beg += 4;
                continue;
            }

            if ( pos1 < midpoint[id1] ) // L_
            {
                if ( pos2 < midpoint[id2] ) // LL
                {
                    lr[ 0 ] += 1;
                }
                else // LR
                {
                    lr[ 1 ] += 1;
                }
            }
            else // R_
            {
                if ( pos2 < midpoint[id2] ) // RL
                {
                    lr[ 2 ] += 1;
                }
                else // RR
                {
                    lr[ 3 ] += 1;
                }
            }

            beg += 4;

            if ( beg >= end || links[ beg + 1 ] != id2 )
            {
                printf("%s %s %d %d %d %d\n", ctx->seqname[id1], ctx->seqname[id2], lr[0], lr[1], lr[2], lr[3]);

                // assert( lr[ 0 ] + lr[ 1 ] + lr[ 2 ] + lr[ 3 ] >= minlinks );

                // if ( lr[ 0 ] + lr[ 1 ] + lr[ 2 ] + lr[ 3 ] >= minlinks )
                {
                    for ( i = 0; i < 4; i++ )
                    {
                        if (lr[i] < minlinks)
                        {
                            continue;
                        }

                        // COUNTS: LL LR RL RR
                        // BEST:   L1 L2 R1 R2

                        // link score
                        uint64_t efflen1 = efflen[id1];
                        uint64_t efflen2 = efflen[id2];
                        uint64_t links1 = (offsets[id1 + 1] - offsets[id1]) / 4;
                        uint64_t links2 = (offsets[id2 + 1] - offsets[id2]) / 4;

                        printf("%llu %llu %llu %llu %llu %llu\n", id1, efflen1, links1, id2, efflen2, links2);

                        assert( id1 < id2 );

                        double s = ( 1.0 * lr[ i ] / log(efflen1) ) / log(efflen2);

                        if ( curlscores + 1 >= maxlscores )
                        {
                            uint32_t old = maxlscores;
                            maxlscores   = maxlscores * 1.2 + 1000;
                            lscores      = realloc( lscores, sizeof( LinkScore ) * maxlscores );
                            bzero( lscores + old, sizeof( LinkScore ) * ( maxlscores - old ) );
                        }

                        lscores[ curlscores ].id1   = id1;
                        lscores[ curlscores ].id2   = id2;
                        lscores[ curlscores ].score = s;
                        lscores[ curlscores ].type  = i;
                        curlscores += 1;

                        // 0    off1 L  off2 L
                        // 1    off1 L  off2 R
                        // 2    off1 R  off2 L
                        // 3    off1 R  off2 R

                        // track best and 2nd best scores of each sequence

                        uint32_t off1 = id1 * 4 + ( i & 2 );
                        uint32_t off2 = id2 * 4 + ( i & 1 ) * 2;

                        if ( scores[ off1 ] < s )
                        {
                            scores[ off1 + 1 ] = scores[ off1 ];
                            scores[ off1 ]     = s;
                        }
                        else if ( scores[ off1 + 1 ] < s )
                        {
                            scores[ off1 + 1 ] = s;
                        }

                        if ( scores[ off2 ] < s )
                        {
                            scores[ off2 + 1 ] = scores[ off2 ];
                            scores[ off2 ]     = s;
                        }
                        else if ( scores[ off2 + 1 ] < s )
                        {
                            scores[ off2 + 1 ] = s;
                        }
                    }
                }

                bzero( lr, sizeof( uint32_t ) * 4 );
            }
        }
    }

    printf( "  confidence of %d links\n", curlscores );

    for ( i = 0; i < curlscores; i++ )
    {
        LinkScore* ls = lscores + i;

        uint32_t off1 = ls->id1 * 4 + ( ls->type & 2 );
        uint32_t off2 = ls->id2 * 4 + ( ls->type & 1 ) * 2;
        double best[ 4 ];

        assert( !( seqstatus[ ls->id1 ] & SFLAG_CONTAINED ) );
        assert( !( seqstatus[ ls->id2 ] & SFLAG_CONTAINED ) );

        assert( ls->id1 != ls->id2 );

        best[ 0 ] = scores[ off1 ];
        best[ 1 ] = scores[ off1 + 1 ];
        best[ 2 ] = scores[ off2 ];
        best[ 3 ] = scores[ off2 + 1 ];

        qsort( best, 4, sizeof( double ), cmp_double );

        uint32_t next = 2;

        if ( AlmostEqualRelative( best[ 2 ], best[ 3 ] ) )
        {
            next = 1;
        }

        if ( AlmostEqualRelative( 0.0, best[ next ] ) )
        {
            ls->confidence = 0.0;
        }
        else
        {
            ls->confidence = ls->score / best[ next ];
        }
    }

    qsort( lscores, curlscores, sizeof( LinkScore ), cmp_link_confidence_desc );

    for ( i = 0; i < curlscores; i++ )
    {
        LinkScore* ls = lscores + i;

        if ( ls->confidence < 1.0 )
        {
            break;
        }
    }

    curlscores = i;

    // build scaffolds

    printf( "  building scaffolds for %d high confidence links\n", curlscores );

    uint32_t joined = 0;
    // uint32_t prev_scaffolds = 0;

    /*
        TODO --- EXPERIMENT
            if one of the ids is forbidden -> tag the other one
            if USED -> tag both ids as forbidden
    */

    for ( i = 0; i < curlscores; i++ )
    {
        LinkScore* ls = lscores + i;

        /*
        uint32_t scaffolds = count_scaffolds(ctx);
        printf("%d %d %d\n", prev_scaffolds, scaffolds, ctx->minclusters);

        if ( prev_scaffolds > 0 && scaffolds <= ctx->minclusters )
        {
            break;
        }
        prev_scaffolds = scaffolds;
        */

        // assert( ls->confidence >= 1.0 );

        uint32_t id1  = ls->id1;
        uint32_t id2  = ls->id2;
        uint32_t type = ls->type;

        assert( id1 != id2 );

        // already assigned to another scaffold

        // if ( ( ctx->seqstatus[ id1 ] & SFLAG_CONTAINED ) || ( ctx->seqstatus[ id2 ] & SFLAG_CONTAINED ) )
        if ( ctx->seqstatus[ id1 ] || ctx->seqstatus[ id2 ] )
        {
            printf( "  %6" PRIu64 "/%6d  %c %6d (%10" PRIu64 " / %10" PRIu64 ") + %c %6d (%10" PRIu64 " / %10" PRIu64 ") %d %.2e %.2e -> USED\n",
                    i, curlscores,
                    ctx->seqsource[ id1 * 2 ] == ctx->seqsource[ id1 * 2 + 1 ] ? ' ' : '*',
                    id1, ctx->seqlen[ id1 ], ctx->midpoint[ id1 ],
                    ctx->seqsource[ id2 * 2 ] == ctx->seqsource[ id2 * 2 + 1 ] ? ' ' : '*',
                    id2, ctx->seqlen[ id2 ], ctx->midpoint[ id2 ],
                    type, ls->score, ls->confidence );

            ctx->seqstatus[ id1 ] |= SFLAG_FORBIDDEN;
            ctx->seqstatus[ id2 ] |= SFLAG_FORBIDDEN;

            continue;
            // break;
        }

        // create new scaffold

        uint32_t idnew = scaffold_sequences( ctx, id1, id2, type );
        joined += 1;

        printf( "  %6" PRIu64 "/%6d  %c %6d (%10" PRIu64 " / %10" PRIu64 ") + %c %6d (%10" PRIu64 " / %10" PRIu64 ") %d %.2e %.2e -> %6d\n",
                i, curlscores,
                ctx->seqsource[ id1 * 2 ] == ctx->seqsource[ id1 * 2 + 1 ] ? ' ' : '*',
                id1, ctx->seqlen[ id1 ], ctx->midpoint[ id1 ],
                ctx->seqsource[ id2 * 2 ] == ctx->seqsource[ id2 * 2 + 1 ] ? ' ' : '*',
                id2, ctx->seqlen[ id2 ], ctx->midpoint[ id2 ],
                type, ls->score, ls->confidence,
                idnew );
    }

    for ( i = 0; i < nseq; i++ )
    {
        ctx->seqstatus[ i ] &= ~SFLAG_FORBIDDEN;
    }

    // get some pointers/values again, since those might have changed in scaffold_sequences

    seqstatus       = ctx->seqstatus;
    nseq            = ctx->nseq;
    offsets         = ctx->offsets;
    links           = ctx->links;
    uint64_t nlinks = ctx->nlinks; // offsets[ nseq ];
    uint64_t ncur   = 0;

    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1 = links[ i ];
        uint64_t id2 = links[ i + 1 ];

        if ( ( seqstatus[ id1 ] & SFLAG_CONTAINED ) || ( seqstatus[ id2 ] & SFLAG_CONTAINED ) )
        {
            continue;
        }

        if ( i != ncur )
        {
            memcpy( links + ncur, links + i, sizeof( uint64_t ) * 4 );
        }

        ncur += 4;
    }

    nlinks = ctx->nlinks = ncur;

    printf( "  restoring triangular matrix\n" );

    bzero( offsets, sizeof( uint64_t ) * ( nseq + 1 ) );

    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1  = links[ i ];
        uint64_t id2  = links[ i + 1 ];
        uint64_t pos1 = links[ i + 2 ];
        uint64_t pos2 = links[ i + 3 ];

        if ( id1 > id2 )
        {
            links[ i ]     = id2;
            links[ i + 1 ] = id1;
            links[ i + 2 ] = pos2;
            links[ i + 3 ] = pos1;

            offsets[ id2 ] += 4;
        }
        else
        {
            offsets[ id1 ] += 4;
        }
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

    printf( "  sorting %" PRIu64 " links\n", nlinks / 4 );
    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_links4 );

    free( scores );
    free( lscores );

    return joined;
}

static void scaffold_replay( ScaffoldContext* ctx, FILE* fin )
{
    char* line;
    int32_t idprev = -1;
    int32_t flip1  = 0;

    while ( ( line = fgetln( fin, NULL ) ) )
    {
        if ( line[ 0 ] == '>' )
        {
            purge_links( ctx );

            printf( "  %s", line + 1 );

            idprev = -1;
            continue;
        }

        char* seqcur = line;
        int len      = strlen( seqcur );

        if ( len - 1 <= 0 )
        {
            continue;
        }

        if ( seqcur[ len - 1 ] == '\n' )
        {
            seqcur[ len - 1 ] = '\0';
        }

        char* comments = strchr( seqcur, ' ' );
        if ( comments )
        {
            *comments = '\0';
        }

        int type;
        if ( seqcur[ 0 ] == '-' )
        {
            seqcur += 1;

            if ( idprev == -1 )
            {
                flip1 = 1;
            }
            else
            {
                if ( flip1 )
                {
                    type = 1;
                }
                else
                {
                    type = 3;
                }

                flip1 = 0;
            }
        }
        else
        {
            if ( flip1 )
            {
                type = 0;
            }
            else
            {
                type = 2;
            }

            flip1 = 0;
        }

        int32_t idcur = name_to_id( ctx, seqcur );
        assert( idcur != -1 );

        if ( idprev != -1 )
        {
            idprev = scaffold_sequences( ctx, idprev, idcur, type );
        }
        else
        {
            idprev = idcur;
        }
    }

    uint32_t nseq     = ctx->nseq;
    uint64_t* offsets = ctx->offsets;
    uint64_t* links   = ctx->links;

    printf( "  restoring triangular matrix\n" );

    uint64_t nlinks = offsets[ nseq ];
    bzero( offsets, sizeof( uint64_t ) * ( nseq + 1 ) );

    uint64_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        uint64_t id1 = links[ i ];
        uint64_t id2 = links[ i + 1 ];

        if ( id1 > id2 )
        {
            uint64_t temp  = links[ i + 2 ];
            links[ i ]     = id2;
            links[ i + 1 ] = id1;
            links[ i + 2 ] = links[ i + 3 ];
            links[ i + 3 ] = temp;
            offsets[ id2 ] += 4;
        }
        else
        {
            offsets[ id1 ] += 4;
        }
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

    printf( "  sorting %" PRIu64 " links\n", nlinks / 4 );
    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_links4 );
}

static void usage( char* app )
{
    printf( "%s [-mls n] [-p string] [-i file] <fasta.in> <links.in> <scaffolds.out>\n", app );
    printf( "options:   -m n ... MAPQ cutoff (default %d)\n", DEF_ARG_M );
    printf( "           -l n ... minimum number of links (default %d)\n", DEF_ARG_L );
    printf( "           -s n ... minimum sequence length (default %d)\n", DEF_ARG_S );
    printf( "           -p s ... scaffold names prefix\n" );
    printf( "           -c n ... min amount of clusters\n" );
    printf( "           -i f ... initialize with scaffolds from file f\n" );
    printf( "           -t n ... number of threads\n" );
    printf( "           -M f ... output merge tree to f\n" );
}

int main( int argc, char* argv[] )
{
    ScaffoldContext ctx;

    bzero( &ctx, sizeof( ScaffoldContext ) );

    // process arguments

    ctx.mapq             = DEF_ARG_M;
    ctx.minlinks         = DEF_ARG_L;
    ctx.minseqlen        = DEF_ARG_S;
    ctx.minclusters      = DEF_ARG_C;
    ctx.nthreads         = 1;
    ctx.seqnameprefix    = NULL;
    char* path_mergetree = NULL;
    char* path_init      = NULL;

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "l:m:s:p:c:i:t:M:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'M':
                path_mergetree = optarg;
                break;

            case 't':
#ifdef THREADED_SCAFFOLDING
                ctx.nthreads = atoi( optarg );

                if ( ctx.nthreads < 1 )
                {
                    fprintf( stderr, "error: invalid number of threads\n" );
                    exit( 1 );
                }
#else
                fprintf( stderr, "warning: ignoring -t argument. compiled without thread support.\n" );
#endif
                break;

            case 'i':
                path_init = optarg;
                break;

            case 'c':
                ctx.minclusters = atoi( optarg );
                break;

            case 'p':
                ctx.seqnameprefix = optarg;
                break;

            case 'l':
                ctx.minlinks = atoi( optarg );

                if ( ctx.minlinks < 1 )
                {
                    fprintf( stderr, "error: invalid min number of links between contigs\n" );
                    exit( 1 );
                }
                break;

            case 'm':
                ctx.mapq = atoi( optarg );

                if ( ctx.mapq < 1 )
                {
                    fprintf( stderr, "error: invalid mapq cutoff\n" );
                    exit( 1 );
                }

                break;

            case 's':
                ctx.minseqlen = atoi( optarg );

                if ( ctx.minseqlen < 1 )
                {
                    fprintf( stderr, "error: invalid min sequence length\n" );
                    exit( 1 );
                }

                break;

            default:
                usage( argv[ 0 ] );
                exit( 1 );
        }
    }

    if ( argc - optind < 3 )
    {
        usage( argv[ 0 ] );
        exit( 1 );
    }

    char* path_fasta = argv[ optind++ ];
    char* path_links = argv[ optind++ ];
    char* path_scaf  = argv[ optind++ ];

    FILE* fout_scaf = fopen( path_scaf, "w" );

    if ( !fout_scaf )
    {
        fprintf( stderr, "could not open %s\n", path_scaf );
        return 1;
    }

    if ( path_mergetree )
    {
        ctx.fout_mergetree = fopen( path_mergetree, "w" );

        if ( !ctx.fout_mergetree )
        {
            fprintf( stderr, "coult not open %s\n", path_mergetree );
            return 1;
        }
        fprintf( ctx.fout_mergetree, "digraph mergetree {\n" );
        fprintf( ctx.fout_mergetree, "ordering=out;\n" );
    }

    if ( !process_fasta( &ctx, path_fasta ) )
    {
        return 1;
    }

    if ( !process_links( &ctx, path_links ) )
    {
        return 1;
    }

    remove_dupe_links(&ctx);

    compute_midpoint(&ctx);

    if ( path_init )
    {
        FILE* fin = fopen( path_init, "r" );
        if ( !fin )
        {
            fprintf( stderr, "failed to open %s\n", path_init );
            exit( 1 );
        }

        printf( "initializing from %s\n", path_init );
        scaffold_replay( &ctx, fin );

        fclose( fin );
    }

    // filter_links(&ctx);

    uint32_t joined;

#ifdef DEBUG_DUMP_ITERATIONS
    uint32_t iteration = 0;
#endif

    uint32_t scaf = 0;

    do
    {
        joined = scaffold( &ctx );
        scaf   = count_scaffolds( &ctx );

        printf( "joined %d | scaffold clusters %d\n", joined, scaf );

#ifdef DEBUG_DUMP_ITERATIONS
        char fname[ PATH_MAX ];
        sprintf( fname, "%s_it.%d", path_scaf, iteration );
        FILE* fout = fopen( fname, "w" );

        write_scaffolds( &ctx, fout, 0 );

        fclose( fout );

        iteration += 1;
#endif
    } while ( joined != 0 );

    write_scaffolds( &ctx, fout_scaf, 0 );

    // calls free() on each key
    hashmap_close( &( ctx.mapseqname ) );

    uint32_t i;
    uint32_t active = 0;
    for ( i = 0; i < ctx.nseq; i++ )
    {
        if ( !( ctx.seqstatus[ i ] & SFLAG_CONTAINED ) )
        {
            active += 1;
        }

        free( ctx.seqname[ i ] );
    }

    printf( "%d active\n", active );

    free( ctx.seqlen );
    free( ctx.offsets );
    free( ctx.links );
    free( ctx.seqstatus );
    free( ctx.seqname );

    if ( ctx.fout_mergetree )
    {
        fprintf( ctx.fout_mergetree, "}\n" );
        fclose( ctx.fout_mergetree );
    }

    return 0;
}
