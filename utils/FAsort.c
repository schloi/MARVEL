
#include "lib/utils.h"
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

// fasta entry

typedef struct
{
    off_t start;            // offset of '>'
    off_t end;              // offset of final '\n'
    uint64_t len;           // sequence length
    uint32_t fidx;          // fasta file index
} FA_ENTRY;

static int cmp_entries_len( const void* a, const void* b )
{
    FA_ENTRY* x = (FA_ENTRY*)a;
    FA_ENTRY* y = (FA_ENTRY*)b;

    return x->len - y->len;
}

static void print_entry( FILE* fout, FILE* ffasta, FA_ENTRY* fe )
{
    static char* buf      = NULL;
    static int64_t maxbuf = 0;

    assert( fe->start < fe->end );

    if ( maxbuf < fe->end - fe->start + 2 )
    {
        maxbuf = ( fe->end - fe->start + 2 ) * 1.2 + 1000;
        buf    = realloc( buf, maxbuf );
    }

    fseeko( ffasta, fe->start, SEEK_SET );
    fread( buf, fe->end - fe->start + 1, 1, ffasta );
    buf[ fe->end - fe->start + 1 ] = '\0';

    fprintf( fout, "%s", buf );
}

static void usage( const char* app )
{
    fprintf( stderr, "usage: %s [-sr] [-mMx n] [-oO <file>] <fasta> [<fasta> ...]\n", app );
    fprintf( stderr, "options: -s ... sort by sequence length\n" );
    fprintf( stderr, "         -r ... reverse\n" );
    fprintf( stderr, "         -m ... minimum sequence length\n" );
    fprintf( stderr, "         -M ... maximum sequence length\n" );
    fprintf( stderr, "         -o ... output to file\n" );
    fprintf( stderr, "         -x ... maximum number of bases to output\n" );
    fprintf( stderr, "         -O ... needs -x, output remaining sequences into this file\n" );
}

int main( int argc, char* argv[] )
{
    int reverse             = 0;
    int sort                = 0;
    uint32_t minlen         = 0;
    uint32_t maxlen         = 0;
    char* pathout           = NULL;
    char* pathout_remainder = NULL;
    uint64_t maxbases       = 0;

    int c;
    opterr = 0;
    while ( ( c = getopt( argc, argv, "m:M:o:O:rsx:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'm':
                minlen = strtol( optarg, NULL, 10 );
                break;

            case 'M':
                maxlen = strtol( optarg, NULL, 10 );
                break;

            case 'o':
                pathout = optarg;
                break;

            case 'O':
                pathout_remainder = optarg;
                break;

            case 'r':
                reverse = 1;
                break;

            case 's':
                sort = 1;
                break;

            case 'x':
                maxbases = bp_parse( optarg );
                break;

            default:
                usage( argv[ 0 ] );
                exit( 1 );
        }
    }

    if ( argc - optind < 1 )
    {
        usage( argv[ 0 ] );
        exit( 1 );
    }

    if ( reverse && !sort )
    {
        fprintf( stderr, "reverse without sort\n" );
        exit( 1 );
    }

    if ( minlen == 0 && maxlen == 0 && sort == 0 && maxbases == 0 )
    {
        fprintf( stderr, "nothing to do\n" );
        exit( 1 );
    }

    if ( minlen != 0 && maxlen != 0 && minlen > maxlen )
    {
        fprintf( stderr, "minimum sequences length is larger than maximume sequence length\n" );
        exit( 1 );
    }

    FILE* fout;
    if ( pathout )
    {
        fout = fopen( pathout, "w" );
        if ( fout == NULL )
        {
            fprintf( stderr, "failed to open %s\n", pathout );
            exit( 1 );
        }
    }
    else
    {
        fout = stdout;
    }

    FILE* fout_remainder;
    if ( pathout_remainder )
    {
        if ( maxbases == 0 )
        {
            fprintf( stderr, "-o without -x\n" );
            exit( 1 );
        }

        fout_remainder = fopen( pathout_remainder, "w" );
        if ( fout == NULL )
        {
            fprintf( stderr, "failed to open %s\n", pathout_remainder );
            exit( 1 );
        }
    }
    else
    {
        fout_remainder = NULL;
    }

    FILE** files    = malloc( sizeof( FILE* ) * ( argc - optind ) );
    uint32_t nfiles = 0;

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

        fprintf( stderr, "processing %s\n", pathin );

        ssize_t len;
        char* line             = NULL;
        size_t maxline         = 0;
        uint64_t prev_nentries = nentries;

        while ( ( len = getline( &line, &maxline, fin ) ) > 0 )
        {
            assert( line[ strlen( line ) - 1 ] == '\n' );

            if ( line[ 0 ] == '>' )
            {
                if ( prev_nentries != nentries )
                {
                    entries[ nentries - 1 ].end = ftello( fin ) - strlen( line ) - 1;
                }

                if ( nentries + 1 >= maxentries )
                {
                    maxentries = maxentries * 1.2 + 1000;
                    entries    = realloc( entries, sizeof( FA_ENTRY ) * maxentries );

                    if ( entries == NULL )
                    {
                        fprintf( stderr, "failed to allocate space for FA_ENTRYs\n" );
                        exit( 1 );
                    }
                }

                entries[ nentries ].fidx  = nfiles;
                entries[ nentries ].start = ftello( fin ) - strlen( line );
                entries[ nentries ].end   = 0;
                entries[ nentries ].len   = 0;

                nentries += 1;
            }
            else
            {
                entries[ nentries - 1 ].len += strlen( line ) - 1;
            }
        }

        if ( prev_nentries != nentries )
        {
            rewind( fin );
            fseeko( fin, 0, SEEK_END );
            entries[ nentries - 1 ].end = ftello( fin );
        }

        nfiles += 1;
    }

    fprintf( stderr, "sorting %" PRIu64 " entries\n", nentries );

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

    uint64_t bases   = 0;
    FILE* fout_print = fout;
    for ( ; i != end; i += incr )
    {
        if ( ( minlen != 0 && entries[ i ].len < minlen ) ||
             ( maxlen != 0 && entries[ i ].len > maxlen ) )
        {
            continue;
        }

        print_entry( fout_print, files[ entries[ i ].fidx ], entries + i );
        bases += entries[ i ].len;

        if ( maxbases != 0 && maxbases < bases )
        {
            if ( fout_remainder )
            {
                fout_print = fout_remainder;
                maxbases   = 0;
            }
            else
            {
                break;
            }
        }
    }

    for ( i = 0; i < nfiles; i++ )
    {
        fclose( files[ i ] );
    }

    free( files );
    free( entries );

    if ( pathout )
    {
        fclose( fout );
    }

    if ( fout_remainder )
    {
        fclose( fout_remainder );
    }

    return 0;
}
