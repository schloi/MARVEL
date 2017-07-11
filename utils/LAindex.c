/**
 * Index an overlap file.
 * - index is used in LAexplorer
 */

/**
 * Idea: add repeats track to contig database
 *
 * Input:
 *  1. Contig-db with reads track (readID,beg,end)
 *  2. database with interval track that should be added to Contig-db
 *  3. interval track
 */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "db/DB.h"
#include "lib/lasidx.h"

static void usage()
{
    printf( "usage: database input.ovl\n\n" );
    printf( "Creates an index file for input.las containing the offsets of the A reads.\n" );
    printf( "Index files are used by some tools to locate the alignments of a read quickly.\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;

    if ( argc != 3 )
    {
        usage();
        exit( 1 );
    }

    if ( Open_DB( argv[ 1 ], &db ) )
    {
        fprintf( stderr, "could not open '%s'\n", argv[ 1 ] );
        exit( 1 );
    }

    lasidx* idx = lasidx_create( &db, argv[ 2 ] );

    if ( idx == NULL )
    {
        fprintf( stderr, "Cannot create index file '%s'\n", argv[ 2 ] );
        exit( 1 );
    }

    // cleanup
    free( idx );
    Close_DB( &db );

    return 0;
}
