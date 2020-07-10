
/*
    HiC playground ...

*/

#include "hashmap.h"
#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <search.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

// command line defaults

#define DEF_ARG_Q 1
#define DEF_ARG_B 1000
#define DEF_ARG_N 0

// toggles

#define SHOW_LINKS_READING_PROGRESS

// maintains the state of the app

typedef struct
{
    // command line arguments

    uint16_t mapq; // MAPQ threshold
    uint32_t binsize;
    uint8_t nself;

    char** seqoutput; // names of sequences to output

    // read id <-> sequence name mappings

    char** seqname;     // read id to sequence name
    Hashmap mapseqname; // sequence name to read id

    // sequences and their links

    uint64_t* seqlen; // sequence lengths
    uint32_t nseq;    // number of sequences
    uint32_t maxseq;

    uint32_t* offsets; // index of a sequence's links
    uint32_t maxoffsets;

    uint64_t* links; // sequence links array of (sequence_2, pos_1, pos_2)
    uint64_t maxlinks;

    uint32_t* counts;
    uint64_t maxcounts;

} MisjoinContext;

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

// maps the sequence name to its id using the hash map

inline static int32_t name_to_id( MisjoinContext* ctx, char* name )
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

static int process_fasta( MisjoinContext* ctx, const char* path_fasta )
{
    printf( "processing fasta\n" );

    FILE* fin;
    char* line;
    int32_t nseq    = 0;
    char* seqname   = NULL;
    uint64_t seqlen = 0;

    int32_t seqmax    = 100;
    char** seqnames   = malloc( sizeof( char** ) * seqmax );
    uint64_t* seqlens = malloc( sizeof( uint64_t ) * seqmax );

    // look for <path_fasta>.fai

    char path_lengths[ PATH_MAX ];
    sprintf( path_lengths, "%s.fai", path_fasta );

    if ( ( fin = fopen( path_lengths, "r" ) ) )
    {
        while ( ( line = fgetln( fin, NULL ) ) )
        {
            // <name>\t<length>\t...\n

            char* num = line;
            strsep( &num, " \t" );

            seqlen = strtoull( num, NULL, 10 );

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
                fprintf( stderr, "line too long in %s\n", path_fasta );
                fclose( fin );
                return 0;
            }

            line[ lenline - 1 ] = '\0';

            if ( line[ 0 ] == '>' )
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

                seqname = strdup( line + 1 );
                seqlen  = 0;
            }
            else
            {
                seqlen += lenline - 1;
            }
        }

        if ( seqlen )
        {
            seqlens[ nseq ]  = seqlen;
            seqnames[ nseq ] = seqname;
            nseq += 1;
        }
    }

    fclose( fin );

    ctx->nseq   = nseq;
    ctx->maxseq = seqmax;

    hashmap_open( &( ctx->mapseqname ), nseq );

    int32_t i;
    for ( i = 0; i < nseq; i++ )
    {
        hashmap_put( &( ctx->mapseqname ), seqnames[ i ], seqnames[ i ] + strlen( seqnames[ i ] ), &i, (void*)( &i ) + sizeof( int32_t ) );
    }

    ctx->seqname = seqnames;
    ctx->seqlen  = seqlens;

    printf( "  %d sequences\n", nseq );

    return 1;
}

static int process_links( MisjoinContext* ctx, const char* path_links )
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
    uint32_t binsize = ctx->binsize;


#ifdef SHOW_LINKS_READING_PROGRESS
    fseek( fin, 0, SEEK_END );
    off_t finlen = ftello( fin );
    rewind( fin );

    off_t finchunk     = finlen / 100;
    off_t finnext      = finchunk;
    uint32_t finchunkn = 1;
#endif

    char* line;

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
            fprintf( stderr, "line too long in %s\n", path_links );
            fclose( fin );
            return 0;
        }

        line[ lenline - 1 ] = '\0';

        char* token;
        char* next = line;
        int col    = 0;
        int skip   = 0;
        int32_t id1, id2, mapqtemp;
        uint64_t pos1, pos2;

        id1 = id2 = -1;
        pos1 = pos2 = 0;

        while ( skip == 0 && ( token = strsep( &next, " \t" ) ) != NULL )
        {
            switch ( col )
            {
                case 0:
                    id1 = name_to_id( ctx, token );
                    if ( id1 == -1 )
                    {
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
                    if ( token[0] != '\0' )
                    {
                        id2 = name_to_id( ctx, token );

                        if ( id2 == -1 )
                        {
                            skip = 1;
                        }
                    }
                    else
                    {
                        id2 = id1;
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
            maxlinks     = maxlinks * 1.2 + 1000;
            links        = realloc( links, sizeof( uint64_t ) * maxlinks );

            bzero( links + old, ( maxlinks - old ) * sizeof( uint64_t ) );
        }

        if ( id1 == id2 )
        {
            links[ nlinks ]     = id1;
            links[ nlinks + 1 ] = pos1 / binsize;
            links[ nlinks + 2 ] = pos2 / binsize;

            nlinks += 3;

            links[ nlinks ]     = id2;
            links[ nlinks + 1 ] = pos2 / binsize;
            links[ nlinks + 2 ] = pos1 / binsize;

            nlinks += 3;
        }
    }

#ifdef SHOW_LINKS_READING_PROGRESS
    printf( "\n" );
#endif

    printf( "  sorting %" PRIu64 " links\n", nlinks / 4 );

    qsort( links, nlinks / 3, sizeof( uint64_t ) * 3, cmp_uint64_3 );

    printf( "  computing offsets\n" );

    uint32_t* offsets = malloc( sizeof( uint32_t ) * ( nseq + 1 ) );
    bzero( offsets, sizeof( uint32_t ) * ( nseq + 1 ) );

    uint64_t i;
    for ( i = 0; i < nlinks; i += 3 )
    {
        offsets[ links[ i ] ] += 3;
    }

    uint32_t off = 0;
    uint32_t coff;

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

    return 1;
}

static void dump_matrix( MisjoinContext* ctx, uint32_t seqid, FILE* fileout)
{
    uint32_t* offsets = ctx->offsets;
    uint64_t* links   = ctx->links;

    uint32_t* counts = ctx->counts;
    uint64_t maxcounts = ctx->maxcounts;
    uint64_t ncounts = 0;
    char* seqname = ctx->seqname[seqid];

    printf("writing matrix for %s\n", seqname);

    uint32_t obeg = offsets[ seqid ];
    uint32_t oend = offsets[ seqid + 1 ];

    if ( obeg >= oend )
    {
        return ;
    }

    uint64_t bin1_prev = UINT64_MAX;
    uint64_t bin2_prev = UINT64_MAX;

    while ( obeg < oend )
    {
        assert( links[ obeg ] == seqid );

        uint64_t id1 = links[ obeg ];
        uint64_t bin1 = links[ obeg + 1 ];
        uint64_t bin2 = links[ obeg + 2 ];

        if ( bin1 < bin2 )
        {
            obeg += 3;
            continue;
        }

        assert( id1 == seqid );

        if ( bin1 != bin1_prev || bin2 != bin2_prev )
        {
            if ( ncounts + 3 >= maxcounts )
            {
                maxcounts = maxcounts * 1.2 + 1000;
                counts = realloc(counts, maxcounts * sizeof(uint32_t) );
            }

            counts[ ncounts ] = bin1;
            counts[ ncounts + 1 ] = bin2;
            counts[ ncounts + 2 ] = 1;

            ncounts += 3;

            bin1_prev = bin1;
            bin2_prev = bin2;
        }
        else
        {
            counts[ncounts - 1] += 1;
        }

        obeg += 3;
    }

    uint64_t i;
    for ( i = 0; i < ncounts; i += 3)
    {
        fprintf(fileout, "%" PRIu32 " %s %d %d %d\n", seqid, seqname, counts[i], counts[i+1], counts[i+2]);

        if ( counts[i] != counts[i+1] )
        {
            fprintf(fileout, "%" PRIu32 " %s %d %d %d\n", seqid, seqname, counts[i+1], counts[i], counts[i+2]);
        }
    }
}

static void usage( char* app )
{
    printf( "%s [-qb n] [-s seqname] [[-s seqname] ...] <fasta.in> <links.in> <matrix.out>\n", app );
    printf( "           -q n    ... MAPQ cutoff (default %d)\n", DEF_ARG_Q );
    printf( "           -b n    ... bin size (default %d)\n", DEF_ARG_B );
    printf( "           -n      ... include non-self hits\n" );
}

int main( int argc, char* argv[] )
{
    MisjoinContext ctx;

    bzero( &ctx, sizeof( MisjoinContext ) );

    // process arguments

    ctx.mapq      = DEF_ARG_Q;
    ctx.binsize   = DEF_ARG_B;
    ctx.nself = DEF_ARG_N;
    ctx.seqoutput = (char**)malloc( sizeof( char* ) * argc );

    uint32_t nseq = 0;

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "b:nq:s:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'b':
                ctx.binsize = atoi( optarg );
                break;

            case 'n':
                ctx.nself = 1;
                break;

            case 'q':
                ctx.mapq = atoi( optarg );
                break;

            case 's':
                ctx.seqoutput[ nseq ] = strdup( optarg );
                nseq += 1;
                break;

            default:
                usage( argv[ 0 ] );
                exit( 1 );
        }
    }

    ctx.seqoutput[ nseq ] = NULL;

    if ( argc - optind != 3 )
    {
        usage( argv[ 0 ] );
        exit( 1 );
    }

    char* path_fasta_in   = argv[ optind++ ];
    char* path_links_in   = argv[ optind++ ];
    char* path_matrix_out = argv[ optind++ ];
    FILE* fileout;

    if ( !( fileout = fopen( path_matrix_out, "w" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_matrix_out );
        return 1;
    }

    if ( !process_fasta( &ctx, path_fasta_in ) )
    {
        return 1;
    }

    if ( !process_links( &ctx, path_links_in ) )
    {
        return 1;
    }

    uint64_t i;
    for ( i = 0; i < nseq; i++)
    {
        char* seq = ctx.seqoutput[i];

        int32_t seqid = name_to_id(&ctx, seq);

        if ( seqid == -1 )
        {
            printf("unknown sequence %s\n", seq);
            continue;
        }

        dump_matrix( &ctx, seqid, fileout );
    }

    hashmap_close( &( ctx.mapseqname ) );

    while ( nseq )
    {
        nseq -= 1;
        free( ctx.seqoutput[ nseq ] );
    }

    fclose(fileout);

    free( ctx.seqoutput );
    free( ctx.seqlen );
    free( ctx.offsets );
    free( ctx.links );
    free( ctx.seqname );

    return 0;
}
