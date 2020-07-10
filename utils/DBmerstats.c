
#include <assert.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/param.h>

#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include "dalign/align.h"
#include "db/DB.h"


typedef struct
{
    HITS_DB* db;
} MerStatsContext;

typedef struct
{
    uint64_t mer;
    uint64_t count;
} MerCount;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    fprintf( stderr, "usage: database\n");

    fprintf( stderr, "options:\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    MerStatsContext ctx;

    bzero( &ctx, sizeof( MerStatsContext ) );

    // process arguments

    int c;

    while ( ( c = getopt( argc, argv, "" ) ) != -1 )
    {
        switch ( c )
        {

            default:
                printf( "Unknown option: %s\n", argv[ optind - 1 ] );
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 2 )
    {
        usage();
        exit( 1 );
    }

    char* pathdb = argv[ optind++ ];

    if ( Open_DB( pathdb, &db ) )
    {
        printf( "could not open '%s'\n", pathdb );
    }

    Close_DB( &db );

    return 0;
}
