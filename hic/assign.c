
#ifndef __clang__
#define _GNU_SOURCE
#endif

#include "hashmap.h"

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

// macros

// command line defaults

#define DEF_ARG_S 20000

// toggles

#define SHOW_LINKS_READING_PROGRESS

// scaffold graph nodes and edges

// constants

typedef struct ScafGraphNode ScafGraphNode;
typedef struct AssignContext AssignContext;

struct ScafGraphNode
{
    char* seqname;
    uint64_t len;
    uint64_t selflinks;
};

// maintains the state of the app

struct AssignContext
{
    // command line arguments

    uint16_t minseqlen; // minimum sequence length
    char* seqnameprefix;
    char* path_scaffold;

    // sequences

    Hashmap mapseqname;

    // graph nodes

    ScafGraphNode* sgnodes;
    uint64_t nsgnodes;
    uint64_t maxsgnodes;

    // links on which the edges are based on

    uint64_t* links; // sequence links array of (seqid_1, seqid_2, pos_1, pos_2)
    uint64_t maxlinks;
    uint64_t nlinks;

    uint64_t maxoffsets_links;
    uint64_t* offsets_links;

};

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

// maps the sequence "name" to its id using the hash map

inline static int32_t name_to_id( AssignContext* ctx, char* name )
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

static int cmp_uint64_counts( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    int i;
    for ( i = 1; i >= 0; i-- )
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

static void links_sort( AssignContext* ctx, uint64_t offset )
{
    uint64_t* links = ctx->links + offset;
    uint64_t nlinks = ctx->nlinks - offset;

    printf( "  sorting %" PRIu64 " links\n", nlinks / 4 );
    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_uint64_4 );
}

static ScafGraphNode* node_new( AssignContext* ctx, char* seqname, uint64_t seqlen )
{
    if ( ctx->nsgnodes + 1 >= ctx->maxsgnodes )
    {
        size_t prev = ctx->maxsgnodes;

        ctx->maxsgnodes = 1.2 * ctx->maxsgnodes + 100;
        ctx->sgnodes    = realloc( ctx->sgnodes, ctx->maxsgnodes * sizeof( ScafGraphNode ) );

        bzero( ctx->sgnodes + prev, ( ctx->maxsgnodes - prev ) * sizeof( ScafGraphNode ) );
    }

    ScafGraphNode* node = ctx->sgnodes + ctx->nsgnodes;
    node->seqname       = seqname;
    node->len           = seqlen;

    ctx->nsgnodes += 1;

    return node;
}

static int process_fasta( AssignContext* ctx, const char* path_fasta )
{
    printf( "processing fasta\n" );

    // look for <path_fasta>.fai

    FILE* fin;
    char* line;
    char* seqname   = NULL;
    uint64_t seqlen = 0;
    uint16_t minseqlen = ctx->minseqlen;
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

            if ( seqlen < minseqlen )
            {
                continue;
            }

            node_new( ctx, strdup( line ), seqlen );
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
                if ( seqlen > minseqlen )
                {
                    node_new( ctx, seqname, seqlen );
                }

                seqname = strdup( line + 1 );
                seqlen  = 0;
            }
        }

        fclose( fin );

        if ( seqlen )
        {
            node_new( ctx, strdup( line ), seqlen );
        }
    }

    uint64_t nsgnodes      = ctx->nsgnodes;
    ScafGraphNode* sgnodes = ctx->sgnodes;

    hashmap_open( &( ctx->mapseqname ), nsgnodes );

    uint64_t i;
    for ( i = 0; i < nsgnodes; i++ )
    {
        hashmap_put( &( ctx->mapseqname ),
                     sgnodes[ i ].seqname,
                     sgnodes[ i ].seqname + strlen( sgnodes[ i ].seqname ),
                     &i,
                     (void*)( &i ) + sizeof( uint64_t ) );
    }

    printf( "  %" PRIu64 " nodes\n", nsgnodes );

    return 1;
}

inline static void link_new( AssignContext* ctx,
                             uint64_t id1, uint64_t id2, uint64_t pos1, uint64_t pos2,
                             uint8_t symmetric )
{
    uint64_t* links   = ctx->links;
    uint64_t maxlinks = ctx->maxlinks;
    uint64_t nlinks   = ctx->nlinks;

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

static int process_links( AssignContext* ctx, const char* path_links, int sort )
{
    printf( "processing links in %s\n", path_links );

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
        int id1, id2;
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
                    // mapqtemp = strtol( token, NULL, 10 );
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
                    // mapqtemp = strtol( token, NULL, 10 );
                    skip = 1;

                    break;
            }

            col += 1;
        }

        if ( id1 == id2 )
        {
            continue;
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
    printf( "\n" );
#endif

    fclose( fin );

    if ( sort )
    {
        links_sort( ctx, 0 );
    }

    return 1;
}

static void filter_links( AssignContext* ctx, uint8_t drop_self )
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

    links = realloc(links, nlinks * sizeof( uint64_t ));

    ctx->nlinks = nlinks;
}

static void update_offsets_links( AssignContext* ctx )
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

static void best_interactors( AssignContext* ctx, FILE* fout )
{
    uint64_t nsgnodes = ctx->nsgnodes;
    ScafGraphNode* sgnodes = ctx->sgnodes;
    uint64_t* offsets = ctx->offsets_links;
    uint64_t* links = ctx->links;

    uint64_t* counts = calloc(nsgnodes * 2, sizeof(uint64_t));

    uint64_t i;
    for ( i = 0 ; i < nsgnodes ; i++ )
    {
        ScafGraphNode* sgnode = sgnodes + i;
        uint64_t beg = offsets[i];
        uint64_t end = offsets[i+1];

        if ( beg >= end )
        {
            continue;
        }

        uint64_t ncounts = 0;
        uint64_t total = (end - beg) / 4;

        uint64_t count = 0;
        uint64_t id2_prev = links[beg+1];

        while ( beg < end )
        {
            assert( links[beg] == i );
            uint64_t id2 = links[beg+1];

            if (id2 != id2_prev )
            {
                counts[ ncounts ] = id2_prev;
                counts[ ncounts+1] = count;
                ncounts += 2;

                count = 1;
                id2_prev = id2;
            }
            else
            {
                count += 1;
            }

            beg += 4;
        }

        counts[ ncounts ] = id2_prev;
        counts[ ncounts+1] = count;
        ncounts += 2;

        qsort(counts, ncounts / 2, sizeof(uint64_t) * 2, cmp_uint64_counts);

        // fprintf(fout, "%" PRIu64 " %s %" PRIu64 , sgnode->len, sgnode->seqname, total);
        fprintf(fout, "%s %" PRIu64 , sgnode->seqname, total);

        uint64_t j = 0;
        while ( ncounts > 0 && j < 10 )
        {
            ncounts -= 2;
            uint64_t id = counts[ncounts];
            count = counts[ncounts + 1];
            j += 1;

            fprintf(fout, " %s %" PRIu64, sgnodes[id].seqname, count);
        }

        fprintf(fout, "\n");
    }

    free(counts);
}

static void usage( char* app )
{
    printf( "%s [-s n] -L links.in [... links.in] -f fasta.in -o counts.out\n", app );

    printf( "           -s n ... minimum sequence length (default %d)\n", DEF_ARG_S );
    printf( "           -L f ... links files\n" );
}

int main( int argc, char* argv[] )
{
    AssignContext ctx;

    bzero( &ctx, sizeof( ctx ) );

    // process arguments

    ctx.minseqlen     = DEF_ARG_S;

    char* path_fasta     = NULL;
    char* path_out = NULL;

    opterr = 0;
    char** links    = calloc( argc * 2, sizeof( char* ) );
    uint64_t nlinks = 0;

    int c;
    while ( ( c = getopt( argc, argv, "f:L:o:s:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'f':
                path_fasta = optarg;
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

                break;

            case 'o':
                path_out = optarg;
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
                printf( "malformed command line arguments\n" );
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

    if ( !path_fasta )
    {
        printf( "missing mandatory arguments\n" );
        usage( argv[ 0 ] );
        exit( 1 );
    }

    FILE* fout;

    if ( (fout = fopen(path_out, "w")) == NULL )
    {
        fprintf(stderr, "failed to open %s\n", path_out);
        usage( argv[0] );
        exit(1);
    }

    if ( !process_fasta( &ctx, path_fasta ) )
    {
        exit( 1 );
    }

    uint64_t i = 0;
    for ( i = 0; i < nlinks ; i++)
    {
        if ( process_links( &ctx, links[ i ], 0 ) == 0 )
        {
            exit( 1 );
        }
    }

    links_sort( &ctx, 0 );

    filter_links( &ctx, 1 );

    update_offsets_links( &ctx );

    best_interactors( &ctx, fout );

    // calls free() on each key
    hashmap_close( &( ctx.mapseqname ) );

    for ( i = 0; i < ctx.nsgnodes; i++ )
    {
        free( ctx.sgnodes[ i ].seqname );
    }

    free( ctx.sgnodes );
    free( ctx.links );
    free( ctx.offsets_links );

    return 0;
}
