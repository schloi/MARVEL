
#include "tracks.h"
#include "lib/compression.h"
#include "pass.h"
#include "tracks.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined( __APPLE__ )
#include <sys/syslimits.h>
#else
#include <linux/limits.h>
#endif

HITS_TRACK* track_load_block( HITS_DB* db, char* tname )
{
    HITS_TRACK* track = track_load( db, tname );

    if ( track == NULL )
    {
        return NULL;
    }

    uint64_t bfirst = db->ufirst;
    uint64_t nreads = db->nreads;

    track_anno* anno = track->anno;
    track_data* data = track->data;

    memmove( anno, anno + bfirst, sizeof( track_anno ) * ( nreads + 1 ) );

    uint64 i;
    track_anno off = anno[ 0 ];
    for ( i = 0; i <= nreads; i++ )
    {
        anno[ i ] -= off;
    }
    uint64 dlen = anno[ nreads ];

    // DATA: data[ off : off + dlen ]

    memmove( data, data + off / sizeof( track_data ), dlen );

    return track;
}

HITS_TRACK* track_load( HITS_DB* db, char* track )
{
    FILE* afile = fopen( Catenate( db->path, ".", track, ".a2" ), "r" );

    // printf("%s\n", Catenate(db->path, ".", track, ".a2"));

    if ( afile == NULL )
    {
        // printf("fall back to Load_Track\n");
        return Load_Track( db, track );
    }

    track_anno_header header;

    if ( fread( &header, sizeof( track_anno_header ), 1, afile ) != 1 )
    {
        fprintf( stderr, "ERROR: could not read header of track %s\n", track );
        return NULL;
    }

    if ( header.size <= 0 )
    {
        fprintf( stderr, "ERROR: invalid size field in track %s\n", track );
        return NULL;
    }

    uint64 nreads = db->nreads;

    if ( db->part == 0 && header.len != nreads )
    {
        fprintf( stderr, "ERROR: invalid track length in header of track %s\n", track );
        return NULL;
    }

    void* canno = malloc( header.clen );
    bzero( canno, header.clen );

    if ( header.clen > 0 )
    {
        if ( fread( canno, header.clen, 1, afile ) != 1 )
        {
            fprintf( stderr, "ERROR: failed to read anno track %s\n", track );
            return NULL;
        }
    }

    fclose( afile );

    // printf("header len %" PRIu64 " clen %" PRIu64 " cdlen %" PRIu64 "\n", header.len, header.clen, header.cdlen);

    uint64 alen  = header.size * ( header.len + 1 );
    uint64* anno = malloc( alen );
    bzero( anno, alen );

    uncompress_chunks( canno, header.clen, anno, alen );

    free( canno );

    char* name  = Catenate( db->path, ".", track, ".d2" );
    FILE* dfile = fopen( name, "r" );

    if ( dfile == NULL )
    {
        fprintf( stderr, "ERROR: failed to open data file for track %s\n", track );
        return NULL;
    }

    void* cdata = malloc( header.cdlen );
    bzero( cdata, header.cdlen );

    if ( header.cdlen > 0 )
    {
        if ( fread( cdata, header.cdlen, 1, dfile ) != 1 )
        {
            fprintf( stderr, "ERROR: failed to read data track %s\n", track );
            return NULL;
        }
    }

    fclose( dfile );

    void* data = malloc( anno[ header.len ] );
    uncompress_chunks( cdata, header.cdlen, data, anno[ header.len ] );

    free( cdata );

    if ( db->ufirst > 1 )
    {
        memmove( anno, anno + db->ufirst, ( db->nreads + 1 ) * sizeof( uint64 ) );
        uint64_t start = anno[ 0 ];
        uint64_t end   = anno[ db->nreads ];

        int i;
        for ( i = 0; i <= db->nreads; i++ )
        {
            anno[ i ] -= start;
        }

        memmove( data, data + start, end - start );
    }

    HITS_TRACK* record = malloc( sizeof( HITS_TRACK ) );

    record->next = db->tracks;
    db->tracks   = record;

    record->name = strdup( track );
    record->data = data;
    record->anno = anno;
    record->size = header.size;

    return record;
}

void track_close( HITS_TRACK* track )
{
    free( track->name );
    free( track->anno );
    free( track->data );

    free( track );
}

char* track_name( HITS_DB* db, const char* track, int block )
{
    char* name = (char*)malloc( PATH_MAX );

    if ( block > 0 )
    {
        sprintf( name, "%s.%d.%s", db->path, block, track );
    }
    else
    {
        sprintf( name, "%s.%s", db->path, track );
    }

    return name;
}

int track_delete( HITS_DB* db, const char* track )
{
    char* root_track = track_name( db, track, 0 );
    char path[ PATH_MAX ];
    int suc = 1;

    sprintf( path, "%s.a2", root_track );
    if ( unlink( path ) != 0 )
    {
        suc = 0;
    }

    sprintf( path, "%s.d2", root_track );
    if ( unlink( path ) != 0 )
    {
        suc = 0;
    }

    return suc;
}

void track_write( HITS_DB* db, const char* track, int block, track_anno* anno, track_data* data, uint64_t dlen )
{
    uint64_t tlen = DB_NREADS( db );

    char* path_track = track_name( db, track, block );
    int end          = strlen( path_track );

    // offsets

    strcat( path_track, ".a2" );

    FILE* afile = fopen( path_track, "w" );

    if ( afile == NULL )
    {
        fprintf( stderr, "failed to open %s\n", path_track );
        return;
    }

    track_anno_header ahead;
    bzero( &ahead, sizeof( track_anno_header ) );

    ahead.version = TRACK_VERSION_2;
    ahead.len     = tlen;
    ahead.size    = sizeof( track_anno );

    if ( fwrite( &ahead, sizeof( track_anno_header ), 1, afile ) != 1 )
    {
        fprintf( stderr, "failed to write track header\n" );
        return;
    }

    void* canno;
    uint64_t clen;

    compress_chunks( anno, sizeof( track_anno ) * ( tlen + 1 ), &canno, &clen );

    if ( fwrite( canno, clen, 1, afile ) != 1 )
    {
        fprintf( stderr, "failed to write track data offsets\n" );
        return;
    }

    ahead.clen = clen;

    free( canno );

    // data

    if ( data != NULL )
    {
        strcpy( path_track + end, ".d2" );

        FILE* dfile = fopen( path_track, "w" );

        if ( dfile == NULL )
        {
            fprintf( stderr, "failed to open %s\n", path_track );
            return;
        }

        compress_chunks( data, sizeof( track_data ) * dlen, &canno, &clen );

        if ( fwrite( canno, clen, 1, dfile ) != 1 )
        {
            fprintf( stderr, "failed to write track data of %" PRIu64 " (%" PRIu64 ") bytes\n", sizeof( track_data ) * dlen, clen );
            return;
        }

        ahead.cdlen = clen;

        free( canno );

        fclose( dfile );
    }
    else if ( dlen != 0 )
    {
        ahead.cdlen = dlen;
    }

    free( path_track );

    rewind( afile );
    if ( fwrite( &ahead, sizeof( track_anno_header ), 1, afile ) != 1 )
    {
        fprintf( stderr, "failed to write track header\n" );
        return;
    }

    fclose( afile );
}

static void write_track( HITS_DB* db, const char* track, int block, track_header_len tlen, track_anno* anno, track_data* data, uint64_t dlen )
{
    char* path_track = track_name( db, track, block );
    int end          = strlen( path_track );

    // offsets

    strcat( path_track, ".anno" );

    FILE* fileTrack = fopen( path_track, "w" );

    if ( fileTrack == NULL )
    {
        fprintf( stderr, "failed to open %s\n", path_track );
        return;
    }

    track_header_size tsize = sizeof( track_anno );

    if ( fwrite( &tlen, sizeof( track_header_len ), 1, fileTrack ) != 1 )
    {
        fprintf( stderr, "failed to write track header\n" );
        return;
    }

    if ( fwrite( &tsize, sizeof( track_header_size ), 1, fileTrack ) != 1 )
    {
        fprintf( stderr, "failed to write track header\n" );
        return;
    }

    if ( fwrite( anno, sizeof( track_anno ), tlen + 1, fileTrack ) != ( size_t )( tlen + 1 ) )
    {
        fprintf( stderr, "failed to write track data offsets\n" );
        return;
    }

    fclose( fileTrack );

    // data

    strcpy( path_track + end, ".data" );

    if ( ( fileTrack = fopen( path_track, "w" ) ) == NULL )
    {
        fprintf( stderr, "failed to open %s\n", path_track );
        return;
    }

    if ( fwrite( data, sizeof( track_data ), dlen, fileTrack ) != dlen )
    {
        fprintf( stderr, "failed to write track data\n" );
        return;
    }

    fclose( fileTrack );

    free( path_track );
}

void write_track_trimmed( HITS_DB* db, const char* track, int block, track_anno* anno, track_data* data, uint64_t dlen )
{
    write_track( db, track, block, db->nreads, anno, data, dlen );
}

void write_track_untrimmed( HITS_DB* db, const char* track, int block, track_anno* anno, track_data* data, uint64_t dlen )
{
    write_track( db, track, block, DB_NREADS( db ), anno, data, dlen );
}
