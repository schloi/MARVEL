
/*
    HiC playground ...

    apply previously detected HI-C misjoins to the links and fasta files
*/

#include "hashmap.h"
#include <assert.h>
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
#include <sys/param.h>

// command line defaults

// maintains the state of the app

typedef struct
{
    // command line arguments

    // read id <-> sequence name mappings

    char** seqname;     // read id to sequence name
    Hashmap mapseqname; // sequence name to read id

    // sequences and their links

    uint64_t* seqlen; // sequence lengths
    uint32_t nseq;    // number of sequences
    uint32_t maxseq;

    // detected misjoins

    uint64_t* segments;
    uint32_t nsegments;
    uint32_t maxsegments;

} MisjoinContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

void strreplace(char* str, char oldChar, char newChar)
{
    int i = 0;

    while(str[i] != '\0')
    {
        if(str[i] == oldChar)
        {
            str[i] = newChar;
        }

        i++;
    }
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

static void wrap_write( FILE* fileOut, char* seq, int len, int width )
{
    int j;
    for ( j = 0; j + width < len; j += width )
    {
        fprintf( fileOut, "%.*s\n", width, seq + j );
    }

    if ( j < len )
    {
        fprintf( fileOut, "%.*s\n", len - j, seq + j );
    }
}

static void rewrite_fasta_sequence( MisjoinContext* ctx, FILE* fout, char* seq, char* seqname, uint32_t* _cursegments )
{
    uint64_t* segments   = ctx->segments;
    uint32_t nsegments   = ctx->nsegments;
    uint32_t cursegments = *_cursegments;

    uint32_t seqid = name_to_id( ctx, seqname );

    while ( cursegments < nsegments && segments[ cursegments ] < seqid )
    {
        cursegments += 3;
    }

    if ( cursegments < nsegments && segments[ cursegments ] == seqid )
    {
        int part = 1;

        while ( cursegments < nsegments && segments[ cursegments ] == seqid )
        {
            uint64_t sbeg = segments[ cursegments + 1 ];
            uint64_t send = segments[ cursegments + 2 ];

            fprintf( fout, ">%s.%d\n", seqname, part );
            wrap_write( fout, seq + sbeg, send - sbeg, 50 );

            // printf( "split %s %" PRIi64 "..%" PRIi64 "\n", seqname, sbeg, send );

            cursegments += 3;
            part += 1;
        }
    }
    else
    {
        fprintf( fout, ">%s\n", seqname );
        wrap_write( fout, seq, strlen( seq ), 50 );
    }

    *_cursegments = cursegments;
}

static int rewrite_fasta( MisjoinContext* ctx, const char* path_fasta_in, const char* path_fasta_out )
{
    printf( "rewriting fasta %s -> %s\n", path_fasta_in, path_fasta_out );

    FILE* fin;
    FILE* fout;

    if ( !( fin = fopen( path_fasta_in, "r" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_fasta_in );
        return 0;
    }

    if ( !( fout = fopen( path_fasta_out, "w" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_fasta_out );
        return 0;
    }

    char* line;
    char* seqname   = NULL;
    char* seq       = NULL;
    uint32_t nseq   = 0;
    uint32_t maxseq = 0;

    uint32_t cursegments = 0;

    while ( ( line = fgetln( fin, NULL ) ) )
    {
        int lenline = strlen( line );

        if ( line[ lenline - 1 ] != '\n' )
        {
            fprintf( stderr, "line too long in %s\n", path_fasta_in );
            fclose( fin );
            return 0;
        }

        line[ lenline - 1 ] = '\0';

        if ( line[ 0 ] == '>' )
        {
            if ( nseq != 0 )
            {
                rewrite_fasta_sequence( ctx, fout, seq, seqname, &cursegments );
                nseq = 0;
            }

            seqname = strdup( line + 1 );
        }
        else
        {
            if ( nseq + lenline >= maxseq )
            {
                maxseq = ( nseq + lenline ) * 1.2;
                seq    = realloc( seq, maxseq );
            }

            strcpy( seq + nseq, line );
            nseq += lenline - 1;
        }
    }

    if ( nseq != 0 )
    {
        rewrite_fasta_sequence( ctx, fout, seq, seqname, &cursegments );
        nseq = 0;
    }

    return 1;
}

static int rewrite_links_remap( MisjoinContext* ctx, uint32_t* soffsets, uint32_t* _id, uint64_t* _pos )
{
    uint32_t id  = *_id;
    uint64_t pos = *_pos;

    uint64_t* segments = ctx->segments;

    uint32_t beg = soffsets[ id ];
    uint32_t end = soffsets[ id + 1 ];

    if ( beg >= end )
    {
        *_id = 0;
        return 1;
    }

    uint32_t part = 1;
    int validpos  = 0;

    while ( beg < end )
    {
        assert( segments[ beg ] == id );
        uint64_t part_beg = segments[ beg + 1 ];
        uint64_t part_end = segments[ beg + 2 ];

        if ( part_beg <= pos && pos < part_end )
        {
            pos -= part_beg;
            validpos = 1;

            break;
        }

        part += 1;
        beg += 3;
    }

    *_id  = part;
    *_pos = pos;

    return validpos;
}

static int rewrite_links_entry( MisjoinContext* ctx, FILE* fout, uint32_t* soffsets,
                                uint32_t id1, uint32_t id2,
                                uint64_t pos1, uint64_t pos2,
                                uint32_t mapq1, uint32_t mapq2 )
{
    char* seqname1 = ctx->seqname[ id1 ];
    char* seqname2 = ctx->seqname[ id2 ];

    if ( rewrite_links_remap( ctx, soffsets, &id1, &pos1 ) == 0 )
    {
        return 0;
    }

    if ( rewrite_links_remap( ctx, soffsets, &id2, &pos2 ) == 0 )
    {
        return 0;
    }

    if ( id1 == 0 )
    {
        fprintf( fout, "%s %" PRIu64 " %d", seqname1, pos1, mapq1 );
    }
    else
    {
        fprintf( fout, "%s.%d %" PRIu64 " %d", seqname1, id1, pos1, mapq1 );
    }

    if ( id2 == 0 )
    {
        fprintf( fout, " %s %" PRIu64 " %d\n", seqname2, pos2, mapq2 );
    }
    else
    {
        fprintf( fout, " %s.%d %" PRIu64 " %d\n", seqname2, id2, pos2, mapq2 );
    }

    return 1;
}

static int rewrite_links( MisjoinContext* ctx, const char* path_links_in, const char* path_links_out )
{
    printf( "rewriting links %s -> %s\n", path_links_in, path_links_out );

    FILE* fin;
    FILE* fout;

    if ( !( fin = fopen( path_links_in, "r" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_links_in );
        return 0;
    }

    if ( !( fout = fopen( path_links_out, "w" ) ) )
    {
        fprintf( stderr, "failed to open %s\n", path_links_out );
        return 0;
    }

    // index segments

    uint32_t nseq      = ctx->nseq;
    uint64_t* segments = ctx->segments;
    uint32_t nsegments = ctx->nsegments;
    uint32_t* soffsets = malloc( sizeof( uint32_t ) * ( nseq + 1 ) );
    bzero( soffsets, sizeof( uint32_t ) * ( nseq + 1 ) );
    uint32_t i;
    for ( i = 0; i < nsegments; i += 3 )
    {
        soffsets[ segments[ i ] ] += 3;
    }

    uint32_t off = 0;
    uint32_t coff;

    uint64_t j;
    for ( j = 0; j <= nseq; j++ )
    {
        coff          = soffsets[ j ];
        soffsets[ j ] = off;
        off += coff;
    }

    // rewrite links

    uint64_t nline = 0;
    uint64_t entries = 0;
    uint64_t entries_rewritten = 0;

    char* line;

    while ( ( line = fgetln( fin, NULL ) ) )
    {
        nline += 1;

        int lenline = strlen( line );

        if ( line[ lenline - 1 ] != '\n' )
        {
            fprintf( stderr, "line too long in %s\n", path_links_in );
            fclose( fin );
            return 0;
        }

        line[ lenline - 1 ] = '\0';

        char* token;
        char* next = line;
        int col    = 0;
        int skip   = 0;

        int id1, id2, mapq1, mapq2;
        uint64_t pos1, pos2;

        strreplace(line, '\t', ' ');

        entries += 1;

        while ( skip == 0 && ( token = strsep( &next, " " ) ) != NULL )
        {
            switch ( col )
            {
                case 0:
                    id1 = name_to_id( ctx, token );
                    if ( id1 == -1 )
                    {
                        printf( "rewrite_link> unknown sequence %s\n", token );
                        skip = 1;
                    }
                    break;

                case 1:
                    pos1 = strtol( token, NULL, 10 );
                    break;

                case 2:
                    mapq1 = strtol( token, NULL, 10 );
                    break;

                case 3:
                    id2 = name_to_id( ctx, token );
                    if ( id2 == -1 )
                    {
                        printf( "rewrite_links> unknown sequence %s\n", token );
                        skip = 1;
                    }
                    break;

                case 4:
                    pos2 = strtol( token, NULL, 10 );
                    break;

                case 5:
                    mapq2 = strtol( token, NULL, 10 );
                    break;
            }

            col += 1;
        }

        if ( skip == 1 )
        {
            continue;
        }

        if ( rewrite_links_entry( ctx, fout, soffsets, id1, id2, pos1, pos2, mapq1, mapq2 ) )
        {
            entries_rewritten += 1;
        }
    }

    printf("read %" PRIu64 " entries, wrote %" PRIu64 "\n", entries, entries_rewritten);

    fclose( fin );
    fclose( fout );

    free( soffsets );

    return 1;
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

static void add_segment( MisjoinContext* ctx, uint64_t sid, uint64_t beg, uint64_t end )
{
    uint64_t* segments   = ctx->segments;
    uint64_t nsegments   = ctx->nsegments;
    uint64_t maxsegments = ctx->maxsegments;

    if ( nsegments + 3 >= maxsegments )
    {
        ctx->maxsegments = maxsegments = 1.2 * maxsegments + 100;
        ctx->segments = segments = realloc( segments, sizeof( uint64_t ) * maxsegments );
    }

    segments[ nsegments ]     = sid;
    segments[ nsegments + 1 ] = beg;
    segments[ nsegments + 2 ] = end;
    nsegments += 3;

    ctx->nsegments = nsegments;
}

static int process_segments( MisjoinContext* ctx, const char* path_in )
{
    FILE* fin;

    fin = fopen( path_in, "r" );

    if (!fin)
    {
        fprintf(stderr, "failed to open %s\n", path_in);
        return 0;
    }

    char* line;

    while ( ( line = fgetln( fin, NULL ) ) )
    {
        if ( line[0] == '#' )
        {
            continue;
        }

        char* token;
        char* next = line;
        int col    = 0;
        int skip   = 0;

        int32_t id;
        uint64_t pos1, pos2;

        strreplace(line, '\t', ' ');

        while ( skip == 0 && ( token = strsep( &next, " " ) ) != NULL )
        {
            switch ( col )
            {
                case 0:
                    break;

                case 1:
                    break;

                case 2:
                    id = name_to_id( ctx, token );
                    if ( id == -1 )
                    {
                        printf( "process_segments> unknown sequence %s\n", token );
                        skip = 1;
                    }
                    break;

                case 3:
                    pos1 = strtol( token, NULL, 10 );
                    break;

                case 4:
                    pos2 = strtol( token, NULL, 10 );
                    break;

                case 5:
                    break;

                default:
                    printf("failed to parse in segments file: %s\n", line);
                    skip = 1;
                    break;
            }

            col += 1;
        }

        if (skip)
        {
            break;
        }

        add_segment(ctx, id, pos1, pos2);
    }

    return 1;
}

static void usage( char* app )
{
    printf( "%s <segments.in> <fasta.in> <fasta.out> <links.in> <links.out> [<links.in> <links.out>] ...\n", app );
}

int main( int argc, char* argv[] )
{
    MisjoinContext ctx;

    bzero( &ctx, sizeof( MisjoinContext ) );

    // process arguments

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "" ) ) != -1 )
    {
        switch ( c )
        {
            default:
                usage( argv[ 0 ] );
                exit( 1 );
        }
    }

    if ( argc - optind < 5 )
    {
        usage( argv[ 0 ] );
        exit( 1 );
    }

    char* path_segments_in = argv[ optind++ ];
    char* path_fasta_in    = argv[ optind++ ];
    char* path_fasta_out   = argv[ optind++ ];

    if ( !process_fasta( &ctx, path_fasta_in ) )
    {
        return 1;
    }

    if ( !process_segments(&ctx, path_segments_in))
    {
        return 1;
    }

    rewrite_fasta( &ctx, path_fasta_in, path_fasta_out );

    while ( optind != argc )
    {
        char* path_links_in    = argv[ optind++ ];
        char* path_links_out   = argv[ optind++ ];

        rewrite_links( &ctx, path_links_in, path_links_out );
    }

    hashmap_close( &( ctx.mapseqname ) );

    free( ctx.seqlen );
    free( ctx.seqname );
    free( ctx.segments );

    return 0;
}
