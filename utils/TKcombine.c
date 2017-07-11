/*
 takes two tracks containing intervals and merges them
 into a single track, removing contained/duplicate intervals
 in the process and merging overlapping ones.

 Source tracks and result track can be trimmed and/or untrimmed

 Date: March 2015
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

static int cmp_intervals( const void* a, const void* b )
{
    track_data* x = (track_data*)a;
    track_data* y = (track_data*)b;

    return x[ 0 ] - y[ 0 ];
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
    int ntracks;
    char** track_name;

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

	/*
    if ( argc - optind == 3 )
    {
        if ( argv[ optind + 2 ][ 0 ] != '#' )
        {
            usage();
            exit( 1 );
        }
    }
    else */ if ( argc - optind < 3 )
    {
        usage();
        exit( 1 );
    }

    char* pathReadsIn     = argv[ optind++ ];
    char* nameTrackResult = argv[ optind++ ];

    ntracks    = argc - optind;
    track_name = malloc( sizeof( char* ) * ntracks );

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

    int nreads = db.ureads;

    track_anno* offset_out = (track_anno*)malloc( sizeof( track_anno ) * ( nreads + 1 ) );
    bzero( offset_out, sizeof( track_anno ) * ( nreads + 1 ) );

    int dcur             = 0;
    int dmax             = 100;
    track_data* data_out = (track_data*)malloc( sizeof( track_data ) * dmax );

    int tcur               = 0;
    int tmax               = 100;
    track_data* temp       = (track_data*)malloc( sizeof( track_data ) * tmax );
    track_anno* offset_tmp = (track_anno*)malloc( sizeof( track_anno ) * ( nreads + 1 ) );
    bzero( offset_tmp, sizeof( track_anno ) * ( nreads + 1 ) );

    int64 noverlap = 0;
    int64 ncontain = 0;

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
                printf( "%lld in %s", anno_in[ db.nreads ] / sizeof( track_data ), inTrack->name );
            }

            bzero( offset_tmp, sizeof( track_anno ) * ( nreads + 1 ) );
            tcur = 0;
            dcur = 0;
            for ( j = 0; j < db.nreads; j++ )
            {
                // check input track
                int ispace, ospace;

                offset_tmp[ j ] = tcur;

                track_anno i_ob = anno_in[ j ] / sizeof( track_data );
                track_anno i_oe = anno_in[ j + 1 ] / sizeof( track_data );

                assert( i_ob <= i_oe );

                ispace = i_oe - i_ob;

                // check current out track
                track_anno o_ob = offset_out[ j ] / sizeof( track_data );
                track_anno o_oe = offset_out[ j + 1 ] / sizeof( track_data );

                assert( o_ob <= o_oe );

                ospace = o_oe - o_ob;

                if ( ispace == 0 && ospace == 0 )
                    continue;

                if ( tcur + ospace + ispace >= tmax )
                {
                    tmax = tmax * 1.2 + ospace + ispace;
                    temp = (track_data*)realloc( temp, sizeof( track_data ) * tmax );
                }

                // merge inTrack and current out-track into temp
                if ( ispace > 0 )
                {
                    memcpy( temp + tcur, data_in + i_ob, sizeof( track_data ) * ( i_oe - i_ob ) );
                    tcur += ispace;
                }
                if ( ospace > 0 )
                {
                    memcpy( temp + tcur, data_out + o_ob, sizeof( track_data ) * ( o_oe - o_ob ) );
                    tcur += ospace;
                }

                if ( ispace != 0 && ospace != 0 )
                {
                    qsort( temp + ( tcur - ispace - ospace ), ( ispace + ospace ) / 2, sizeof( track_data ) * 2, cmp_intervals );

                    int k;
                    for ( k = tcur - ispace - ospace + 2; k < tcur; k += 2 )
                    {
                        // contained -> replace with previous
                        if ( temp[ k + 1 ] <= temp[ k - 1 ] )
                        {
                            temp[ k ]     = temp[ k - 2 ];
                            temp[ k + 1 ] = temp[ k - 1 ];

                            temp[ k - 2 ] = temp[ k - 1 ] = -1;

                            ncontain++;
                        }
                        // overlapping
                        else if ( temp[ k ] <= temp[ k - 1 ] )
                        {
                            temp[ k ] = temp[ k - 2 ];

                            temp[ k - 2 ] = temp[ k - 1 ] = -1;

                            noverlap++;
                        }
                    }
                }
            }

            // copy tmp into data_out
            if ( dcur + tcur >= dmax )
            {
                dmax     = dmax * 1.2 + tcur;
                data_out = (track_data*)realloc( data_out, sizeof( track_data ) * dmax );
            }
            bzero( offset_out, sizeof( track_anno ) * ( nreads + 1 ) );

            offset_tmp[ nreads ] = tcur;

            for ( j = 0; j < nreads; j++ )
            {
                track_anno k;
                for ( k = offset_tmp[ j ]; k < offset_tmp[ j + 1 ]; k += 2 )
                {
                    if ( temp[ k ] == -1 )
                    {
                        continue;
                    }

                    data_out[ dcur++ ] = temp[ k ];
                    data_out[ dcur++ ] = temp[ k + 1 ];

                    offset_out[ j ] += sizeof( track_data ) * 2;
                }
            }

            track_anno coff, off;
            off = 0;

            for ( j = 0; j <= nreads; j++ )
            {
                coff = offset_out[ j ];
                offset_out[ j ] = off;
                off += coff;
            }

            for ( j = 0; j < nreads; j++ )
            {
                assert( offset_out[ j ] <= offset_out[ j + 1 ] );
            }

            // TODO use track_close, but it has to be adapted to update the linked list of DB tracks
            Close_Track( &db, inTrack->name );

            if ( verbose )
            {
                printf( " %lld contained, %lld overlapped, %lld cum\n", ncontain, noverlap, offset_out[ nreads ] / sizeof( track_data ) );
            }
        }
    }

    if ( verbose )
    {
        printf( "%lld contained\n%lld overlapped\n", ncontain, noverlap );
        printf( "%lld in %s\n", offset_out[ nreads ] / sizeof( track_data ),
                nameTrackResult );
    }

    track_write( &db, nameTrackResult, 0, offset_out, data_out,
                 offset_out[ nreads ] / sizeof( track_data ) );

    free( temp );
    free( data_out );
    free( offset_out );
    free( offset_tmp );

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
