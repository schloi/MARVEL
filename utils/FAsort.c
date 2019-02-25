
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

typedef struct
{
    off_t start;
    off_t end;
    uint64_t len;
    uint32_t fidx;
} FA_ENTRY;

static int cmp_entries_len( const void* a, const void* b )
{
    FA_ENTRY* x = (FA_ENTRY*)a;
    FA_ENTRY* y = (FA_ENTRY*)b;

    return x->len - y->len;
}

static void print_entry( FILE* ffasta, FA_ENTRY* fe )
{
    static char* buf      = NULL;
    static int64_t maxbuf = 0;

    if ( maxbuf < fe->end - fe->start + 2 )
    {
        maxbuf = ( fe->end - fe->start ) * 1.2 + 1000;
        buf    = realloc( buf, maxbuf );
    }

    fseeko( ffasta, fe->start, SEEK_SET );
    fread( buf, fe->end - fe->start + 1, 1, ffasta );
    buf[fe->end - fe->start + 1] = '\0';

    printf( "%s", buf );
}

static void usage()
{
    fprintf( stderr, "usage: [-sr] [-mM n] <fasta> [<fasta> ...]\n" );
    fprintf( stderr, "options: -s ... sort by sequence length\n" );
    fprintf( stderr, "         -r ... reverse\n" );
    fprintf( stderr, "         -m ... minimum sequence length\n" );
    fprintf( stderr, "         -M ... maximum sequence length\n" );
}

int main( int argc, char* argv[] )
{
    int reverse     = 0;
    int sort        = 0;
    uint32_t minlen = 0;
    uint32_t maxlen = 0;

    int c;
    opterr = 0;
    while ( ( c = getopt( argc, argv, "srm:M:" ) ) != -1 )
    {
        switch ( c )
        {
            case 's':
                sort = 1;
                break;

            case 'r':
                reverse = 1;
                break;

            case 'm':
                minlen = strtol( optarg, NULL, 10 );
                break;

            case 'M':
                maxlen = strtol( optarg, NULL, 10 );
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 1 )
    {
        usage();
        exit( 1 );
    }

    if ( reverse && !sort )
    {
        fprintf( stderr, "reverse without sort\n" );
        exit( 1 );
    }

    if ( minlen == 0 && maxlen == 0 && sort == 0 )
    {
        fprintf( stderr, "neither sorting nor filtering requested\n" );
        exit( 1 );
    }

    if ( minlen != 0 && maxlen != 0 && minlen > maxlen )
    {
        fprintf( stderr, "minimum sequences length is larger than maximume sequence length\n" );
        exit( 1 );
    }

    FILE** files        = malloc( sizeof( FILE* ) * ( argc - optind ) );
    uint32_t nfiles     = 0;

    FA_ENTRY* entries   = NULL;
    uint64_t maxentries = 0;
    uint64_t nentries   = 0;

    while ( optind < argc )
    {
        char* pathin    = argv[ optind++ ];
        FILE* fin       = fopen( pathin, "r" );
        files[ nfiles ] = fin;

        if ( !fin )
        {
            fprintf( stderr, "failed to open %s\n", pathin );
            exit( 1 );
        }

        ssize_t len;
        char* line = NULL;
        size_t maxline = 0;
        uint64_t nline = 0;

        while ( ( len = getline( &line, &maxline, fin ) ) > 0 )
        {
            assert( line[ strlen(line) - 1 ] == '\n' );

            if ( line[ 0 ] == '>' )
            {
                if ( nentries > 0 )
                {
                    entries[ nentries - 1 ].end = ftello( fin ) - strlen( line ) - 1;
                }

                if ( nentries + 1 >= maxentries )
                {
                    maxentries = maxentries * 1.2 + 1000;
                    entries    = realloc( entries, sizeof( FA_ENTRY ) * maxentries );
                }

                entries[ nentries ].start = ftello( fin ) - strlen( line );
                entries[ nentries ].len   = 0;
                entries[ nentries ].fidx  = nfiles;

                nentries += 1;
            }
            else
            {
                entries[ nentries - 1 ].len += strlen( line ) - 1;
            }

            nline += 1;
        }

        if ( nentries > 0 )
        {
            entries[ nentries - 1 ].end = ftello( fin ) - strlen( line ) - 1;
        }

        nfiles += 1;
    }

    if ( sort )
    {
        qsort( entries, nentries, sizeof( FA_ENTRY ), cmp_entries_len );
    }

    int64_t i;
    short incr;
    int64_t end;
    if ( reverse )
    {
        i    = nentries - 1;
        incr = -1;
        end  = -1;
    }
    else
    {
        i    = 0;
        incr = 1;
        end  = nentries;
    }

    for ( ; i != end; i += incr )
    {
        if ( ( minlen != 0 && entries[ i ].len < minlen ) ||
             ( maxlen != 0 && entries[ i ].len > maxlen ) )
        {
            continue;
        }

        print_entry( files[ entries[ i ].fidx ], entries + i );
    }

    for ( i = 0; i < nfiles; i++ )
    {
        fclose( files[ i ] );
    }

    free( files );
    free( entries );

    return 0;
}
