/*******************************************************************************************
 *
 *  Displays the contents of a .las file
 *
 *  Date    : May 2015
 *
 *  Author  : MARVEL Team
 *
 *******************************************************************************************/

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

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

typedef struct
{

} TestContext;

static void pre_test( PassContext* pctx, TestContext* tctx )
{
}

static void post_test()
{
}

static int handler_test( void* _ctx, Overlap* ovls, int novl )
{
    TestContext* ctx = (TestContext*)_ctx;

    return 1;
}

static void usage()
{
    printf( "<db> <overlaps>\n" );
};

int main( int argc, char* argv[] )
{
    HITS_DB db;

    PassContext* pctx;
    TestContext tctx;

    bzero( &tctx, sizeof( TestContext ) );

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

    char* dbpath  = argv[ optind++ ];
    char* ovlpath = argv[ optind++ ];

    if ( Open_DB( dbpath, &db ) )
    {
        printf( "could not open '%s'\n", dbpath );
        exit( 1 );
    }

    Close_DB( &db );

    return 0;
}
