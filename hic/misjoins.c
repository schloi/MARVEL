
/*
    HiC playground ...

    HiC based misjoin detection

    Future:
        - direct mapping of HiC reads to assembly graph
        - use those links to aid touring through the graph
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

// toggles

#define ENABLE_DEBUG_OUTPUT_FILES

// command line defaults

#define DEF_ARG_Q 1
#define DEF_ARG_S 20000

#define MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )

// maintains the state of the app

typedef struct
{
    // command line arguments

    uint16_t mapq;      // MAPQ threshold
    uint32_t minseqlen; // minimum sequence length

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
    uint32_t maxlinks;

    // detected misjoins

    uint64_t* segments;
    uint32_t nsegments;
    uint32_t maxsegments;

    // working storage for misjoin detection

    uint32_t maxdiag;
    uint64_t* diag;

    uint32_t maxcomp;

    int64_t* compfrom;
    int64_t* compto;
    uint32_t* compcont;
    uint32_t* comphisto;

#ifdef ENABLE_DEBUG_OUTPUT_FILES
    uint32_t maxmatrix;
    uint8_t* matrix;
    FILE* foutmatrix;
#endif

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

#define CMP_RETURN(a, b, i) { if ( a[ i ] < b[ i ] ) { return -1; } else if ( a[ i ] > b[ i ] ) { return 1; } }

static int cmp_segments( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    CMP_RETURN(a, b, 0);
    CMP_RETURN(a, b, 2);

    return 0;
}

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

static int cmp_links4_distance( const void* x, const void* y )
{
    uint64_t* a = (uint64_t*)x;
    uint64_t* b = (uint64_t*)y;

    return llabs( (int64_t)a[ 3 ] - (int64_t)a[ 2 ] ) - llabs( (int64_t)b[ 3 ] - (int64_t)b[ 2 ] );

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
    uint32_t maxlinks = 0;
    uint32_t nlinks   = 0;
    uint16_t mapq     = ctx->mapq;
    uint64_t nline    = 0;
    uint64_t* seqlen  = ctx->seqlen;

    char* line;

    while ( ( line = fgetln( fin, NULL ) ) )
    {
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
        int id1, mapqtemp;
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
                    if ( token[0] != '\0' )
                    {
                        int32_t idtemp = name_to_id( ctx, token );
                        if ( idtemp == -1 )
                        {
                            printf( "unknown sequence %s\n", token );
                            skip = 1;
                        }
                        else if ( idtemp != id1 )
                        {
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

        if ( skip == 1 || pos1 == pos2 )
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

        if ( pos2 >= seqlen[ id1 ] )
        {
            if ( pos2 - 100 < seqlen[ id1 ] )
            {
                pos2 -= 100;
            }
            else
            {
                printf( "line %" PRIu64 " mapping position %" PRIu64 " beyond contig length %" PRIu64 "\n",
                        nline, pos2, seqlen[ id1 ] );
                continue;
            }
        }

        if ( nlinks + 4 >= maxlinks )
        {
            uint32_t old = maxlinks;
            maxlinks     = maxlinks * 1.5 + 1000;
            links        = realloc( links, sizeof( uint64_t ) * maxlinks );

            bzero( links + old, ( maxlinks - old ) * sizeof( uint64_t ) );
        }

        if ( pos1 > pos2 )
        {
            uint64_t t = pos1;
            pos1       = pos2;
            pos2       = t;
        }

        links[ nlinks ]     = id1;
        links[ nlinks + 1 ] = id1;
        links[ nlinks + 2 ] = pos1;
        links[ nlinks + 3 ] = pos2;

        nlinks += 4;
    }

    printf( "  sorting %d links\n", nlinks / 4 );

    qsort( links, nlinks / 4, sizeof( uint64_t ) * 4, cmp_links4 );

    printf( "  computing offsets\n" );

    uint32_t* offsets = malloc( sizeof( uint32_t ) * ( nseq + 1 ) );
    bzero( offsets, sizeof( uint32_t ) * ( nseq + 1 ) );

    uint32_t i;
    for ( i = 0; i < nlinks; i += 4 )
    {
        offsets[ links[ i ] ] += 4;
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

#ifdef ENABLE_DEBUG_OUTPUT_FILES
    FILE* fileout = fopen( "linkstats.txt", "w" );

    for ( i = 0; i < nseq; i++ )
    {
        uint64_t obeg = offsets[ i ];
        uint64_t oend = offsets[ i + 1 ];

        fprintf( fileout, "%s %" PRIu64 " %" PRIu64 "\n", ctx->seqname[ i ], seqlen[ i ], ( oend - obeg ) / 4 );
    }

    fclose( fileout );
#endif

    ctx->maxoffsets = nseq + 1;
    ctx->offsets    = offsets;
    ctx->maxlinks   = maxlinks;
    ctx->links      = links;

    return 1;
}

static void add_segment( MisjoinContext* ctx, uint64_t sid, uint64_t comp, uint64_t beg, uint64_t end )
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
    segments[ nsegments + 1 ] = comp;
    segments[ nsegments + 2 ] = beg;
    segments[ nsegments + 3 ] = end;
    nsegments += 4;

    ctx->nsegments = nsegments;
}

static void build_fuzzy_graph( MisjoinContext* ctx, uint32_t seqid,
                               uint16_t threshold, uint16_t cutter )
{
    uint32_t* offsets   = ctx->offsets;
    uint64_t* links     = ctx->links;
    uint64_t* seqlen    = ctx->seqlen;
    uint64_t* diag      = ctx->diag;
    uint32_t maxcomp    = ctx->maxcomp;
    int64_t* compfrom   = ctx->compfrom;
    int64_t* compto     = ctx->compto;
    uint32_t* compcont  = ctx->compcont;
    uint32_t* comphisto = ctx->comphisto;

    uint32_t obeg = offsets[ seqid ];
    uint32_t oend = offsets[ seqid + 1 ];

    if ( obeg >= oend )
    {
        return;
    }

    uint32_t nlinks      = ( oend - obeg ) / 4;
    uint16_t granularity = MAX( 4 ^ cutter, 2 * seqlen[ seqid ] / nlinks );

    printf("read %d %s\n", seqid, ctx->seqname[seqid]);
    printf( "granularity %d | cutter %d | seqlen %" PRIu64 " links %d\n", granularity, cutter, seqlen[seqid], nlinks );

    uint64_t gseqlen = ( seqlen[ seqid ] + granularity - 1 ) / granularity;

    if ( gseqlen > ctx->maxdiag )
    {
        ctx->maxdiag = gseqlen;
        ctx->diag = diag = realloc( ctx->diag, sizeof( uint64_t ) * ctx->maxdiag );
    }

    if ( gseqlen > maxcomp )
    {
        maxcomp = ctx->maxcomp = gseqlen;

        compfrom = ctx->compfrom = realloc( compfrom, maxcomp * sizeof( int64_t ) );
        compto = ctx->compto = realloc( compto, maxcomp * sizeof( int64_t ) );
        compcont = ctx->compcont = realloc( compcont, maxcomp * sizeof( uint32_t ) );
        comphisto = ctx->comphisto = realloc( comphisto, maxcomp * sizeof( uint32_t ) );
    }

    bzero( diag, sizeof( uint64_t ) * ctx->maxdiag );

    // project onto diagonal

    uint64_t diag_active = 0;

    for ( ; obeg < oend; obeg += 4 )
    {
        assert( links[ obeg ] == seqid );

        if ( links[ obeg + 1 ] != seqid )
        {
            continue;
        }

        uint64_t pos1 = links[ obeg + 2 ];
        uint64_t pos2 = links[ obeg + 3 ];

        if ( pos1 >= pos2 )
        {
            printf( "WARNING %s %" PRIu64 " >= %" PRIu64 "\n", ctx->seqname[ seqid ], pos1, pos2 );
            fflush( stdout );
        }

        assert( pos1 < pos2 );

        assert( pos1 < seqlen[ seqid ] );
        assert( pos2 < seqlen[ seqid ] );

        uint64_t gpos1 = pos1 / granularity;
        uint64_t gpos2 = pos2 / granularity;

        assert( gpos1 < gseqlen );
        assert( gpos2 < gseqlen );

        if ( diag[ gpos1 ] == 0 )
        {
            diag[ gpos1 ] = 1;
            diag_active += 1;
        }

        if ( diag[ gpos2 ] == 0 )
        {
            diag[ gpos2 ] = 1;
            diag_active += 1;
        }
    }

    printf( "%" PRIu64 " of %" PRIu64 " active on diagonal\n", diag_active, gseqlen );

    // assign components

    uint32_t ccur = 1;

    int32_t cbeg = -1;
    int32_t cend = -1;

    uint64_t i;
    for ( i = 0; i < gseqlen; i++ )
    {
        if ( diag[ i ] == 0 )
        {
            if ( cbeg != -1 )
            {
                cend = i - 1;

                compfrom[ ccur ] = cbeg;
                compto[ ccur ]   = cend;
                compcont[ ccur ] = 0;

                if ( ccur > 1 && cbeg - compto[ ccur - 1 ] < 2 )
                {
                    compto[ ccur - 1 ] = cend;

                    uint32_t j;
                    for ( j = cbeg; j < i; j++ )
                    {
                        diag[ j ] = ccur - 1;
                    }
                }
                else
                {
                    uint32_t j;
                    for ( j = cbeg; j < i; j++ )
                    {
                        diag[ j ] = ccur;
                    }

                    ccur += 1;
                }

                cbeg = -1;
            }
        }
        else
        {
            if ( cbeg == -1 )
            {
                cbeg = i;
            }
        }
    }

    if ( cbeg != -1 )
    {
        compfrom[ ccur ] = cbeg;
        compto[ ccur ]   = i - 1;
        compcont[ ccur ] = 0;

        uint32_t j;
        for ( j = cbeg; j < i; j++ )
        {
            diag[ j ] = ccur;
        }

        ccur += 1;
    }

    printf( "sequence %d | %d components | %" PRIu64 " gseqlen\n", seqid, ccur - 1, gseqlen );

    obeg = offsets[ seqid ];
    oend = offsets[ seqid + 1 ];

    qsort( links + obeg, ( oend - obeg ) / 4, sizeof( uint64_t ) * 4, cmp_links4_distance );

    for ( ; obeg < oend; obeg += 4 )
    {
        if ( links[ obeg + 1 ] != seqid )
        {
            continue;
        }

        uint64_t pos1 = links[ obeg + 2 ];
        uint64_t pos2 = links[ obeg + 3 ];

        if ( pos2 - pos1 < threshold )
        {
            continue;
        }

        uint64_t gpos1 = pos1 / granularity;
        uint64_t gpos2 = pos2 / granularity;

        if ( gpos1 == gpos2 )
        {
            continue;
        }

        uint32_t c1 = diag[ gpos1 ];
        uint32_t c2 = diag[ gpos2 ];

        if ( c1 == 0 || c2 == 0 || c1 == c2 )
        {
            continue;
        }

        if ( ccur + 1 >= maxcomp )
        {
            maxcomp = ctx->maxcomp = maxcomp * 1.2 + 100;

            compfrom = ctx->compfrom = realloc( compfrom, maxcomp * sizeof( int64_t ) );
            compto = ctx->compto = realloc( compto, maxcomp * sizeof( int64_t ) );
            compcont = ctx->compcont = realloc( compcont, maxcomp * sizeof( uint32_t ) );
            comphisto = ctx->comphisto = realloc( comphisto, maxcomp * sizeof( uint32_t ) );
        }

        assert( compcont[ c1 ] == 0 );
        assert( compcont[ c2 ] == 0 );

        // force to signed, otherwise overflow detection in debug build complains

        compfrom[ ccur ] = ( -1 ) * (int32_t)c1;
        compto[ ccur ]   = ( -1 ) * (int32_t)c2;
        compcont[ ccur ] = 0;

        compcont[ c1 ] = ccur;
        compcont[ c2 ] = ccur;

        for ( i = 0; i < gseqlen; i++ )
        {
            if ( diag[ i ] == c1 || diag[ i ] == c2 )
            {
                diag[ i ] = ccur;
            }
        }

        ccur += 1;
    }

    bzero( comphisto, ccur * sizeof( uint32_t ) );

    uint32_t cactive = 0;

    for ( i = 0; i < gseqlen; i++ )
    {
        uint32_t c = diag[ i ];

        if ( comphisto[ c ] == 0 )
        {
            cactive += 1;
        }

        comphisto[ c ] += 1;
    }

    /*
    for ( i = 0; i < gseqlen; i++ )
    {
        printf("DIAG %d %" PRIu64 " %" PRIu64 "\n", seqid, i * granularity, diag[i]);
    }
    */

    printf( "%d active components\n", cactive );

    uint16_t wndsize   = 10; // TODO: hardcoded
    uint16_t nsegments = 0;

    while ( 1 )
    {
        uint32_t cmax   = 0;
        int32_t cmaxidx = -1;

        uint32_t j;
        for ( j = 1; j < ccur; j++ )
        {
            if ( cmaxidx == -1 || comphisto[ j ] > cmax )
            {
                cmax    = comphisto[ j ];
                cmaxidx = j;
            }
        }

        if ( cmax == 0 )
        {
            break;
        }

        printf( "component %d occurances %d\n", cmaxidx, cmax );

        assert( cmaxidx > 0 );

        comphisto[ cmaxidx ] = 0;

        j = 0;

        while ( 1 )
        {
            while ( j < gseqlen && diag[ j ] != (uint64_t)cmaxidx )
            {
                j += 1;
            }

            if ( j == gseqlen )
            {
                break;
            }

            uint64_t beg       = j;
            uint64_t last_seen = j;
            uint64_t dist2last = 0;

            j += 1;

            while ( j < gseqlen && dist2last < wndsize )
            {
                uint64_t c = diag[ j ];

                if ( c == (uint64_t)cmaxidx )
                {
                    dist2last = 0;
                    last_seen = j;
                }
                else if ( c != 0 )
                {
                    dist2last += 1;
                }

                j += 1;
            }

            printf( "%d @ %" PRIu64 "..%" PRIu64 " %" PRIu64 "..%" PRIu64 " (dist %" PRIu64 ")\n",
                    cmaxidx,
                    beg, last_seen,
                    beg * granularity,
                    MIN( seqlen[ seqid ], ( last_seen + 1 ) * granularity - 1 ),
                    dist2last );

            if ( beg < last_seen )
            {
                add_segment( ctx, seqid, cmaxidx,
                             beg * granularity,
                             MIN( seqlen[ seqid ], ( last_seen + 1 ) * granularity - 1 ) );
                nsegments += 1;
            }

            uint64_t k;
            for ( k = beg; k < last_seen; k++ )
            {
                diag[ k ] = cmaxidx;
            }

            j = last_seen + 1;
        }
    }

#ifdef ENABLE_DEBUG_OUTPUT_FILES

    if ( nsegments > 1 )
    {
        uint16_t binsize   = 1000;
        uint32_t bins      = ( seqlen[ seqid ] + binsize - 1 ) / binsize;
        uint32_t maxmatrix = bins * bins;
        uint8_t* matrix    = ctx->matrix;

        if ( maxmatrix > ctx->maxmatrix )
        {
            ctx->maxmatrix = maxmatrix;
            ctx->matrix = matrix = realloc( ctx->matrix, maxmatrix );
        }

        uint32_t obeg = offsets[ seqid ];
        uint32_t oend = offsets[ seqid + 1 ];

        bzero( matrix, maxmatrix );

        printf( "SEGMENTS %d %d %d\n", seqid, nsegments, ( oend - obeg ) / 4 );

        while ( obeg < oend )
        {
            uint64_t pos1 = links[ obeg + 2 ];
            uint64_t pos2 = links[ obeg + 3 ];

            if ( pos1 > seqlen[ seqid ] )
            {
                continue;
            }

            if ( pos2 > seqlen[ seqid ] )
            {
                continue;
            }

            uint64_t bin1 = pos1 / binsize;
            uint64_t bin2 = pos2 / binsize;

            uint64_t el1 = bin1 * bins + bin2;
            uint64_t el2 = bin2 * bins + bin1;

            if ( matrix[ el1 ] < 255 )
            {
                matrix[ el1 ] += 1;
            }

            if ( matrix[ el2 ] < 255 )
            {
                matrix[ el2 ] += 1;
            }

            obeg += 4;
        }

        uint32_t j;
        for ( j = 0; j < bins; j++ )
        {
            uint32_t k;
            for ( k = 0; k < bins; k++ )
            {
                if ( matrix[ j * bins + k ] > 0 )
                {
                    fprintf( ctx->foutmatrix, "%d %d %d %d\n", seqid, j, k, matrix[ j * bins + k ] );
                }
            }
        }
    }
#endif

    printf( "########\n" );
}

static void write_segments( MisjoinContext* ctx, const char* pathout )
{
    uint64_t* seqlen   = ctx->seqlen;
    uint64_t* segments = ctx->segments;
    uint32_t nsegments = ctx->nsegments;
    char** seqname     = ctx->seqname;
    uint32_t minseqlen = ctx->minseqlen;

    qsort( segments, nsegments / 4, sizeof( uint64_t ) * 4, cmp_segments );

    FILE* fout = fopen( pathout, "w" );
    if ( fout == NULL )
    {
        printf( "failed to open %s\n", pathout );
        return;
    }

    fprintf( fout, "# sequence start end\n" );

    uint64_t seqid_prev = INT_MAX;
    uint32_t i;
    for ( i = 0; i < nsegments; i += 4 )
    {
        uint64_t seqid = segments[ i ];
        uint64_t comp  = segments[ i + 1 ];
        uint64_t beg   = segments[ i + 2 ];
        uint64_t end   = segments[ i + 3 ];

        if ( seqid_prev != seqid )
        {
            beg = 0;
            seqid_prev = seqid;
        }

        if ( i + 4 >= nsegments || segments[i + 4] != seqid )
        {
            end = seqlen[seqid];
        }

        if ( ( beg == 0 && end == seqlen[seqid] )
            || (end - beg) < minseqlen )
        {
            fprintf( fout, "# " );
            // continue
        }

        fprintf( fout, "%" PRIu64 " %" PRIu64 " %s %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", seqid, comp, seqname[ seqid ], beg, end, seqlen[ seqid ] );
    }

    fclose( fout );
}

static void misjoins( MisjoinContext* ctx )
{
    printf( "searching for misjoins...\n" );

    uint32_t i;
    uint32_t nseq      = ctx->nseq;
    uint32_t minseqlen = ctx->minseqlen;
    uint64_t* seqlen   = ctx->seqlen;
    uint32_t* offsets   = ctx->offsets;

    for ( i = 0; i < nseq; i++ )
    {
        uint64_t len = seqlen[i];

        if ( len < minseqlen )
        {
            continue;
        }

        uint32_t obeg = offsets[ i ];
        uint32_t oend = offsets[ i + 1 ];
        uint32_t nlinks      = ( oend - obeg ) / 4;

#warning "DEBUG CODE - REMOVE"
        {
        if (nlinks == 0) continue;
        printf("read %d len %" PRIu64 " nlinks %d len/nlinks %" PRIu64 "\n", i, len, nlinks, len / nlinks);
        }

        if ( nlinks == 0 || len / nlinks > 2000 )
        {
            continue;
        }

        build_fuzzy_graph( ctx, i, 1000, 5 );
    }

    free( ctx->compfrom );
    free( ctx->compto );
    free( ctx->compcont );
    free( ctx->comphisto );
    free( ctx->diag );
}

static void usage( char* app )
{
    printf( "%s [-qs n] <fasta.in> <links.in> <segments.out>\n", app );
    printf( "options:   -q n    ... MAPQ cutoff (default %d)\n", DEF_ARG_Q );
    printf( "           -s n    ... minimum sequence length (default %d)\n", DEF_ARG_S );
}

int main( int argc, char* argv[] )
{
    MisjoinContext ctx;

    bzero( &ctx, sizeof( MisjoinContext ) );

    // process arguments

    ctx.mapq      = DEF_ARG_Q;
    ctx.minseqlen = DEF_ARG_S;

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "q:s:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'q':
                ctx.mapq = atoi( optarg );
                break;

            case 's':
                ctx.minseqlen = atoi( optarg );
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

    char* path_fasta_in     = argv[ optind++ ];
    char* path_links_in     = argv[ optind++ ];
    char* path_segments_out = argv[ optind++ ];

    if ( !process_fasta( &ctx, path_fasta_in ) )
    {
        return 1;
    }

    process_links( &ctx, path_links_in );

    uint64_t maxlen = 0;
    uint32_t i;
    uint32_t nseq    = ctx.nseq;
    uint64_t* seqlen = ctx.seqlen;
    for ( i = 0; i < nseq; i++ )
    {
        if ( maxlen < seqlen[ i ] )
        {
            maxlen = seqlen[ i ];
        }
    }

#ifdef ENABLE_DEBUG_OUTPUT_FILES
    ctx.foutmatrix = fopen( "linkmatrix.txt", "w" );
#endif

    misjoins( &ctx );

#ifdef ENABLE_DEBUG_OUTPUT_FILES
    fclose( ctx.foutmatrix );
#endif

    write_segments( &ctx, path_segments_out );

    hashmap_close( &( ctx.mapseqname ) );

    free( ctx.seqlen );
    free( ctx.offsets );
    free( ctx.links );
    free( ctx.seqname );
    free( ctx.segments );

    return 0;
}
