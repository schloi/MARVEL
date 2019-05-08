/*
    takes two tracks containing intervals and merges them
    into a single track, removing contained/duplicate intervals
    in the process and merging overlapping ones.

    Source tracks and result track can be trimmed and/or untrimmed

    Created: March 2015
    Rewrite: April 2019
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lib/colors.h"
#include "lib/tracks.h"

#include "db/DB.h"

// toggles

#undef DEBUG

static int cmp_trackdata_3( const void* a, const void* b )
{
    track_data* x = (track_data*)a;
    track_data* y = (track_data*)b;

    int i;
    for ( i = 0; i < 3; i++ )
    {
        track_data d = x[ i ] - y[ i ];

        if ( d != 0 )
        {
            return d;
        }
    }

    return 0;
}

static void usage()
{
    printf( "usage: [-vd] database track.out [ <track.in1> ... | #.track ]\n\n" );

    printf( "Combines annotation tracks with overlapping intervals into a single track.\n\n" );

    printf( "options: -v  verbose\n" );
    printf( "         -d  remove input tracks after combining\n" );
    printf( "    #.track  prefixing the track name with #. selects all tracks 1.track ... database_block.tracks\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;

    int verbose = 0;
    int delete  = 0;

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "hvd" ) ) != -1 )
    {
        switch ( c )
        {
            case 'h':
                usage();
                exit( 1 );

            case 'v':
                verbose = 1;
                break;

            case 'd':
                delete = 1;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 3 )
    {
        usage();
        exit( 1 );
    }

    char* pathReadsIn     = argv[ optind++ ];
    char* nameTrackResult = argv[ optind++ ];

    int ntracks       = argc - optind;
    char** track_name = malloc( sizeof( char* ) * ntracks );

    int i, j;
    i = 0;
    while ( optind != argc )
    {
        track_name[ i ] = argv[ optind ];
        optind++;
        i++;
    }

    if ( Open_DB( pathReadsIn, &db ) )
    {
        printf( "could not open db %s\n", pathReadsIn );
        exit( 1 );
    }

    int nblocks = DB_Blocks( pathReadsIn );
    if ( nblocks < 1 )
    {
        fprintf( stderr, "failed to get number of blocks\n" );
        exit( 1 );
    }

    // read all tracks into an array of triplets (read_id,  begin, end)

    int nreads = db.ureads;

    uint64_t tmax    = 1000;
    track_data* temp = malloc( sizeof( track_data ) * tmax );
    uint64_t tcur    = 0;

    uint64_t noverlap = 0;
    uint64_t ncontain = 0;

    HITS_TRACK* inTrack;
    char* tmpTrackName = malloc( 1000 );
    for ( i = 0; i < ntracks; i++ )
    {
        int cBlock = 1;
        int cont   = 1;

        while ( cont && cBlock <= nblocks )
        {
            if ( track_name[ i ][ 0 ] == '#' )
            {
                sprintf( tmpTrackName, "%d.%s", cBlock++, track_name[ i ] + 2 );

                if ( ( inTrack = track_load( &db, tmpTrackName ) ) == NULL )
                {
                    fprintf( stderr, "could not open track %s\n", tmpTrackName );
                    exit( 1 );
                }
            }
            else
            {
                if ( ( inTrack = track_load( &db, track_name[ i ] ) ) == NULL )
                {
                    fprintf( stderr, "could not open track %s\n", track_name[ i ] );
                    exit( 1 );
                }

                cont = 0;
            }

            track_anno* anno_in = inTrack->anno;
            track_data* data_in = inTrack->data;

            if ( verbose )
            {
                printf( "%lld intervals in %s\n", anno_in[ db.nreads ] / ( 2 * sizeof( track_data )), inTrack->name );
            }

            uint64_t needed = anno_in[ db.nreads ] / 2 + anno_in[ db.nreads ] / sizeof( track_data );

            if ( tcur + needed > tmax )
            {
                tmax = ( tcur + needed ) * 1.2 + 1000;
                temp = realloc( temp, tmax * sizeof( track_data ) );
            }

            for ( j = 0; j < db.nreads; j++ )
            {
                track_anno i_ob = anno_in[ j ] / sizeof( track_data );
                track_anno i_oe = anno_in[ j + 1 ] / sizeof( track_data );

                assert( i_ob <= i_oe );

                if ( i_ob == i_oe )
                {
                    continue;
                }

                while ( i_ob < i_oe )
                {
                    temp[ tcur ]     = j;
                    temp[ tcur + 1 ] = data_in[ i_ob ];
                    temp[ tcur + 2 ] = data_in[ i_ob + 1];

                    i_ob += 2;
                    tcur += 3;
                }
            }
        }
    }

    printf("%lld intervals\n", tcur / 3);

    // sort triplets

    qsort( temp, tcur / 3, sizeof( track_data ) * 3, cmp_trackdata_3 );

    uint64_t discarded = 0;

    for ( i = 3; i < tcur; i += 3 )
    {
        // read id changed
        if ( temp[ i - 3 ] != temp[ i ] )
        {
            continue;
        }

        // start > previous end -> neither contained not intersecting
        if ( temp[ i + 1 ] > temp[ i - 1 ] )
        {
            continue;
        }

        // overlapping, extend previous intervals
        if ( temp[ i + 2 ] > temp[ i - 1 ] )
        {
            temp[ i - 1 ] = temp[ i + 2 ];
        }

        // discard current
        memset( temp + i, 0, sizeof( track_data ) * 3 );
        discarded += 3;
    }

    printf("%lld intervals discarded\n", discarded / 3);

    // sort triplets again, moving discarded (0, 0, 0) triplets to the front

    qsort( temp, tcur / 3, sizeof( track_data ) * 3, cmp_trackdata_3 );

    // count number of intervals for each read, discard read id
    // and store (begin, end) tuples at beginning of the array

    track_anno* offset_out = (track_anno*)malloc( sizeof( track_anno ) * ( nreads + 1 ) );
    bzero( offset_out, sizeof( track_anno ) * ( nreads + 1 ) );

    uint64_t tcur_free = 0;
    for ( i = discarded ; i < tcur ; i += 3 )
    {
        offset_out[ temp[i] ] += sizeof(track_data) * 2;

        temp[tcur_free] = temp[i + 1];
        temp[tcur_free + 1] = temp[i + 2];

        tcur_free += 2;
    }

    // turn interval counts per read into offsets

    track_anno coff, off;
    off = 0;

    for ( j = 0; j <= nreads; j++ )
    {
        coff            = offset_out[ j ];
        offset_out[ j ] = off;
        off += coff;
    }

    // sanity checks

#ifdef DEBUG
    for ( j = 0; j < nreads; j++ )
    {
        assert( offset_out[ j ] <= offset_out[ j + 1 ] );
    }
#endif

    // write track, free memory and optionally delete input tracks

    track_write( &db, nameTrackResult, 0, offset_out, temp,
                 offset_out[ nreads ] / sizeof( track_data ) );

    free( temp );
    free( offset_out );

    if ( delete )
    {
        for ( i = 0; i < ntracks; i++ )
        {
            if ( track_name[ i ][ 0 ] == '#' )
            {
                for ( j = 1; j <= nblocks; j++ )
                {
                    sprintf( tmpTrackName, "%d.%s", j, track_name[ i ] + 2 );
                    track_delete( &db, tmpTrackName );
                }
            }
            else
            {
                track_delete( &db, track_name[ i ] );
            }
        }
    }

    free( track_name );
    free( tmpTrackName );

    Close_DB( &db );

    return 0;
}
