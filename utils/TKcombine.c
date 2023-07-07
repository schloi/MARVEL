/*
    takes two tracks containing intervals and merges them
    into a single track, removing contained/duplicate intervals
    in the process and merging overlapping ones.

    Source tracks and result track can be trimmed and/or untrimmed
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/limits.h>

#include "lib/colors.h"
#include "lib/tracks.h"

#include "db/DB.h"

// toggles

#undef DEBUG

// getopt

extern char* optarg;
extern int optind, opterr, optopt;


static int cmp_trackdata_3( const void* a, const void* b )
{
    const track_data* x = (const track_data*)a;
    const track_data* y = (const track_data*)b;

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

static uint64_t remove_redundancies(track_data* data, uint64_t ndata)
{
    printf("%" PRIu64 " intervals\n", ndata / 3);

    // sort triplets

    qsort( data, ndata / 3, sizeof( track_data ) * 3, cmp_trackdata_3 );

    uint64_t discarded = 0;
    uint64_t iprev = 0;
    uint64_t i;

    for ( i = 3; i < ndata; i += 3 )
    {
        // read id changed
        if ( data[ iprev ] != data[ i ] )
        {
            iprev = i;
            continue;
        }

        // start > previous end -> neither contained not intersecting
        if ( data[ i + 1 ] > data[ iprev + 2 ] )
        {
            iprev = i;
            continue;
        }

        // overlapping, extend previous intervals
        if ( data[ i + 2 ] > data[ iprev + 2 ] )
        {
            data[ iprev + 2 ] = data[ i + 2 ];
        }

        // discard current
        memset( data + i, 0, sizeof( track_data ) * 3 );
        discarded += 3;
    }

    printf("%" PRIu64 " intervals discarded\n", discarded / 3);

    // sort triplets again, moving discarded (0, 0, 0) triplets to the front

    qsort( data, ndata / 3, sizeof( track_data ) * 3, cmp_trackdata_3 );

    // move the remaining data to the front of the array

    memmove(data, data + discarded, sizeof(track_data) * (ndata - discarded));

    return ndata - discarded;
}

static void usage()
{
    printf( "usage: [-dfv] [-r n] database track.out [ <track.in1> ... | #.track ]\n\n" );

    printf( "Combines annotation tracks with overlapping intervals into a single track.\n\n" );

    printf( "options: -v  verbose\n" );
    printf( "         -d  remove input tracks after combining\n" );
    printf( "         -r  perform a redundancy removal after n files have been loaded\n");
    printf( "         -f  force continue on error\n");
    printf( "    #.track  prefixing the track name with #. selects all tracks 1.track ... database_block.tracks\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;

    int force_continue = 0;
    int verbose = 0;
    int delete  = 0;
    uint32_t remove_redundancies_after = 0;

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "dfhvr:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'r':
                remove_redundancies_after = strtol(optarg, NULL, 10);
                break;

            case 'h':
                usage();
                exit( 1 );

            case 'v':
                verbose = 1;
                break;

            case 'd':
                delete = 1;
                break;

            case 'f':
                force_continue = 1;
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

    uint32_t ntracks       = argc - optind;
    char** track_name = malloc( sizeof( char* ) * ntracks );

    uint32_t i, j;
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

    uint32_t nblocks = DB_Blocks( pathReadsIn );
    if ( nblocks < 1 )
    {
        fprintf( stderr, "failed to get number of blocks\n" );
        exit( 1 );
    }

    // read all tracks into an array of triplets (read_id,  begin, end)

    uint32_t nreads = db.ureads;

    size_t tmax    = 1000;
    track_data* temp = malloc( sizeof( track_data ) * tmax );
    uint64_t tcur    = 0;

    HITS_TRACK* inTrack;
    char* tmpTrackName = malloc( PATH_MAX );
    for ( i = 0; i < ntracks; i++ )
    {
        uint32_t cBlock = 1;
        uint32_t cont   = 1;
        uint32_t tracks_processed = 0;

        while ( cont && cBlock <= nblocks )
        {
            char* opened_track_name;

            if ( track_name[ i ][ 0 ] == '#' )
            {
                sprintf( tmpTrackName, "%d.%s", cBlock++, track_name[ i ] + 2 );
                opened_track_name = tmpTrackName;
            }
            else
            {
                opened_track_name = track_name[i];
                cont = 0;
            }

            if ( ( inTrack = track_load( &db, opened_track_name ) ) == NULL )
            {
                fprintf( stderr, "could not open track %s\n", opened_track_name );

                if (force_continue)
                {
                    fprintf( stderr, "warning: continuing despite error\n");
                    cont = 0;
                    nblocks = cBlock - 1;
                    break;
                }
                else
                {
                    exit( 1 );
                }
            }

            tracks_processed += 1;

            track_anno* anno_in = inTrack->anno;
            track_data* data_in = inTrack->data;

            if ( verbose )
            {
                printf( "%lld intervals in %s\n", anno_in[ nreads ] / ( 2 * sizeof( track_data )), inTrack->name );
            }

            uint64_t needed = anno_in[ nreads ] / 2 + anno_in[ nreads ] / sizeof( track_data );

            if ( tcur + needed > tmax )
            {
                tmax = ( tcur + needed ) * 1.2 + 1000;
                temp = realloc( temp, tmax * sizeof( track_data ) );

                if ( temp == NULL )
                {
                    fprintf(stderr, "failed to extend temporary storage\n");
                    exit(1);
                }
            }

            for ( j = 0; j < nreads; j++ )
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

            Close_Track(&db, opened_track_name);

            if ( remove_redundancies_after > 0 && tracks_processed % remove_redundancies_after == 0 )
            {
                tcur = remove_redundancies(temp, tcur);
            }
        }
    }

    tcur = remove_redundancies(temp, tcur);

    // count number of intervals for each read, discard read id
    // and store (begin, end) tuples at beginning of the array

    track_anno* offset_out = (track_anno*)malloc( sizeof( track_anno ) * ( nreads + 1 ) );
    bzero( offset_out, sizeof( track_anno ) * ( nreads + 1 ) );

    uint64_t tcur_free = 0;
    for ( i = 0 ; i < tcur ; i += 3 )
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

    int twritten = track_write( &db, nameTrackResult, 0, offset_out, temp,
                              offset_out[ nreads ] / sizeof( track_data ) );

    free( temp );
    free( offset_out );

    if ( delete )
    {
        if ( twritten )
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
        else
        {
            fprintf(stderr, "skipping input track deletion due to track write failure\n");
        }
    }

    free( track_name );
    free( tmpTrackName );

    Close_DB( &db );

    return 0;
}
