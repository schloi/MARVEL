
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <limits.h>

#include "align.h"
#include "ovlbuffer.h"

// set initial overlap buffer size to 500'000 overlaps
#define BUFFER_NOVLS (500 * 1000)

extern int SYMMETRIC;

int Num_Threads( Align_Spec* espec )
{
    return ( ( (Align_Spec*)espec )->nthreads );
}

Overlap_IO_Buffer* OVL_IO_Buffer( Align_Spec* espec )
{
    return (Overlap_IO_Buffer*)( espec->ioBuffer );
}

Overlap_IO_Buffer* CreateOverlapBuffer( int nthreads, int tbytes, int no_trace )
{
    Overlap_IO_Buffer* iobuf = (Overlap_IO_Buffer*)malloc( sizeof( Overlap_IO_Buffer ) );
    if ( iobuf == NULL )
    {
        fprintf( stderr, "[ERROR] - Cannot allocate Overlap_IO_Buffer!\n" );
        return NULL;
    }

    iobuf->omax     = BUFFER_NOVLS / nthreads + 1;
    iobuf->otop     = 0;
    iobuf->ovls     = (Overlap*)malloc( sizeof( Overlap ) * iobuf->omax );
    iobuf->no_trace = no_trace;

    if ( iobuf->ovls == NULL )
    {
        fprintf( stderr, "[ERROR] - Cannot allocate Overlap buffer of size: %d!\n", iobuf->omax );
        return NULL;
    }

    if ( no_trace )
    {

        iobuf->tbytes = 0;
        iobuf->tmax   = 0;
        iobuf->ttop   = 0;
        return iobuf;
    }

    if ( tbytes < 1 || tbytes > 2 )
    {
        fprintf( stderr, "[ERROR] - Unsupported size of trace: %d!\n", tbytes );
        return NULL;
    }
    iobuf->tbytes = tbytes;
    iobuf->tmax   = iobuf->omax * 120; // assumption: in average 120 trace points per overlap
    iobuf->ttop   = 0;
    if ( tbytes == 1 )
        iobuf->trace = (uint8*)malloc( sizeof( uint8 ) * iobuf->tmax );
    else
        iobuf->trace = (uint16*)malloc( sizeof( uint16 ) * iobuf->tmax );

    if ( iobuf->trace == NULL )
    {
        fprintf( stderr, "[ERROR] - Cannot allocate trace buffer of size: %" PRIu64 "!\n", iobuf->tmax );
        return NULL;
    }

    return iobuf;
}

int AddOverlapToBuffer( Overlap_IO_Buffer* iobuf, Overlap* ovl, int tbytes )
{
    if ( iobuf == NULL )
    {
        fprintf( stderr, "[ERROR] - Cannot add overlap to Overlap_IO_Buffer. Buffer is NULL!\n" );
        return 1;
    }

    // reallocate buffers
    if ( iobuf->otop == iobuf->omax )
    {
        iobuf->omax = ( iobuf->omax * 1.2 ) + 1000;
        iobuf->ovls = (Overlap*)realloc( iobuf->ovls, sizeof( Overlap ) * iobuf->omax );
        if ( iobuf->ovls == NULL )
        {
            fprintf( stderr, "[ERROR] - Cannot add increase overlap buffer size to %d!\n", iobuf->omax );
            return 1;
        }
    }

    if ( ovl->path.trace != NULL && ( iobuf->no_trace == 0 ) )
    {
        if ( iobuf->ttop + ovl->path.tlen >= iobuf->tmax )
        {
            iobuf->tmax = ( iobuf->tmax * 1.2 ) + 1000;
            void* t;
            if ( iobuf->tbytes == 1 )
                t = (uint8*)realloc( iobuf->trace, sizeof( uint8 ) * iobuf->tmax );
            else
                t = (uint16*)realloc( iobuf->trace, sizeof( uint16 ) * iobuf->tmax );

            if ( t == NULL )
            {
                fprintf( stderr, "[ERROR] - Cannot add increase trace point buffer size to %" PRIu64 "!\n", iobuf->tmax );
                return 1;
            }

            if ( t != iobuf->trace )
            {
                // adjust trace offset
                int i, cumOff = 0;
                for ( i = 0; i < iobuf->otop; i++ )
                {
                    iobuf->ovls[ i ].path.trace = t + cumOff;
                    cumOff += iobuf->ovls[ i ].path.tlen;
                }

                iobuf->trace = t;
            }
        }
    }

    // add overlap
    Overlap* o    = iobuf->ovls + iobuf->otop;
    o->aread      = ovl->aread;
    o->bread      = ovl->bread;
    o->flags      = ovl->flags;
    o->path.abpos = ovl->path.abpos;
    o->path.aepos = ovl->path.aepos;
    o->path.bbpos = ovl->path.bbpos;
    o->path.bepos = ovl->path.bepos;
    o->path.diffs = ovl->path.diffs;

    // add trace
    if ( ovl->path.trace == NULL || iobuf->no_trace )
    {
        o->path.trace = NULL;
        o->path.tlen  = 0;
    }
    else
    {
        o->path.tlen = ovl->path.tlen;
        memcpy( iobuf->trace + iobuf->ttop, ovl->path.trace, tbytes * ovl->path.tlen );
        o->path.trace = iobuf->trace + iobuf->ttop;
    }

    // adjust offsets
    iobuf->otop++;
    iobuf->ttop += tbytes * ovl->path.tlen;
    return 0;
}

static int SORT_OVL( const void* x, const void* y )
{
    Overlap* l = (Overlap*)x;
    Overlap* r = (Overlap*)y;

    int al, ar;
    int bl, br;

    al = l->aread;
    bl = l->bread;

    ar = r->aread;
    br = r->bread;

    if ( al != ar )
        return ( al - ar );

    if ( bl != br )
        return ( bl - br );

    if ( COMP( l->flags ) > COMP( r->flags ) )
        return 1;

    if ( COMP( l->flags ) < COMP( r->flags ) )
        return -1;

    return ( l->path.abpos - r->path.abpos );
}

int mkdir_p( const char* path )
{
    const size_t len = strlen( path );
    char _path[ PATH_MAX ];
    char* p;

    errno = 0;

    if ( len > sizeof( _path ) - 1 )
    {
        errno = ENAMETOOLONG;
        return -1;
    }
    strcpy( _path, path );

    for ( p = _path + 1; *p; p++ )
    {
        if ( *p == '/' )
        {
            *p = '\0';

            if ( mkdir( _path, S_IRWXU ) != 0 )
            {
                if ( errno != EEXIST )
                    return -1;
            }

            *p = '/';
        }
    }

    if ( mkdir( _path, S_IRWXU ) != 0 )
    {
        if ( errno != EEXIST )
            return -1;
    }

    return 0;
}

char* pathLas( const char* base, char* aroot, int ablockID, char* broot, int bblockID )
{
    char* path = malloc( PATH_MAX );
    char* prefix = malloc( PATH_MAX );

    if ( ablockID > 0 && bblockID > 0 )
    {
        size_t len = strlen(base);
        if ( base[len - 1] != '/' )
        {
            sprintf(prefix, "%s_%05d", base, ablockID);
        }
        else
        {
            sprintf(prefix, "%s%05d", base, ablockID);
        }


        if ( mkdir_p(prefix) != 0 )
        {
            fprintf(stderr, "failed to create %s\n", prefix);
            exit(1);
        }

        sprintf( path, "%s/%s.%d.%s.%d.las", prefix, aroot, ablockID, broot, bblockID );
    }
    else if ( ablockID > 0 )
        sprintf( path, "%s/%s.%d.%s.las", base, aroot, ablockID, broot );
    else if ( bblockID > 0 )
        sprintf( path, "%s/%s.%s.%d.las", base, aroot, broot, bblockID );
    else
        sprintf( path, "%s/%s.%s.las", base, aroot, broot );

    free(prefix);

    return path;
}

int split_blockname(char* block, char* root)
{
    int bid = 0;
    char* dot;

    dot = strrchr( block, '.' );
    if ( dot != NULL )
    {
        bid = atoi( ++dot );
        snprintf( root, dot - block, "%s", block );
    }

    return bid;
}

void Write_Overlap_Buffer( Align_Spec* spec, const char* base, char* ablock, char* bblock, int lastRead )
{
    // sort all overlaps
    Overlap_IO_Buffer* buf = OVL_IO_Buffer( spec );
    int nthreads           = Num_Threads( spec );
    int symmetric          = SYMMETRIC;

    int nallOvls = 0;
    int i, j;
    for ( i = 0; i < nthreads; i++ )
        nallOvls += buf[ i ].otop;

    // TODO: use smaller chunks for real sorting and merging
    Overlap* allOvls = (Overlap*)malloc( sizeof( Overlap ) * nallOvls );
    if ( allOvls == NULL )
    {
        fprintf( stderr, "[ERROR] - Write_Overlap_Buffer: Cannot create file overlap buffer for all threads\n" );
        exit( 1 );
    }

    // TODO: use Overlap**

    int count = 0;
    for ( i = 0; i < nthreads; i++ )
        for ( j = 0; j < buf[ i ].otop; j++ )
            allOvls[ count++ ] = buf[ i ].ovls[ j ];

    assert( count == nallOvls );

    // sort overlaps
    qsort( allOvls, nallOvls, sizeof( Overlap ), SORT_OVL );

    // get blocks ids and root
    int ablockID, bblockID;
    char* aroot = malloc( strlen( ablock ) + 10 );
    char* broot = malloc( strlen( bblock ) + 10 );

    ablockID = split_blockname(ablock, aroot);
    bblockID = split_blockname(bblock, broot);

    /*

    ablockID = bblockID = 0;
    {
        char* dot;

        dot = strrchr( ablock, '.' );
        if ( dot != NULL )
        {
            ablockID = atoi( ++dot );
            snprintf( aroot, dot - ablock, "%s", ablock );
        }

        dot = strrchr( bblock, '.' );
        if ( dot != NULL )
        {
            bblockID = atoi( ++dot );
            snprintf( broot, dot - bblock, "%s", bblock );
        }
    }

    */

    // if parts are equal, then dump out all overlaps into a single file
    if ( strcmp( ablock, bblock ) == 0 || symmetric == 0 )
    {
        char* path = pathLas( base, aroot, ablockID, broot, bblockID );

        FILE* out = fopen( path, "w" );

        if ( out == NULL )
        {
            fprintf( stderr, "[ERROR] - Write_Overlap_Buffer: Cannot open file %s for writing\n", path );
            free( path );
            exit( 1 );
        }

        int64 nhits = 0;
        int tspace  = Trace_Spacing( spec );
        int tbytes  = buf->tbytes;
        fwrite( &nhits, sizeof( int64 ), 1, out );
        fwrite( &tspace, sizeof( int ), 1, out );

        for ( j = 0; j < nallOvls; j++, nhits++ )
            Write_Overlap( out, allOvls + j, tbytes );

        assert( nhits == nallOvls );
        rewind( out );
        fwrite( &nhits, sizeof( int64 ), 1, out );
        fclose( out );

        // cleanup
        free( path );
    }
    else // dump out ablock-vs-bblock ovls into one file and bblock-vs-ablock into a second file
    {
        char* path1 = NULL;
        char* path2 = NULL;

        if ( ablockID > 0 )
        {
            path1 = pathLas( base, aroot, ablockID, broot, bblockID );
        }

        if ( bblockID > 0 )
        {
            path2 = pathLas( base, broot, bblockID, aroot, ablockID );
        }

        if ( bblockID < ablockID )
        {
            char* tmp = path1;
            path1     = path2;
            path2     = tmp;
        }

        //        printf("path1: %s, path2: %s\n", path1, path2);

        // dump out reads to first overlap file
        FILE* out = fopen( path1, "w" );
        if ( out == NULL )
        {
            fprintf( stderr, "[ERROR] - Write_Overlap_Buffer: Cannot open file %s for writing\n", path1 );
            exit( 1 );
        }
        int64 nhits = 0;
        int tspace  = Trace_Spacing( spec );
        int tbytes  = buf->tbytes;
        fwrite( &nhits, sizeof( int64 ), 1, out );
        fwrite( &tspace, sizeof( int ), 1, out );

        for ( j = 0; j < nallOvls; j++, nhits++ )
        {
            if ( allOvls[ j ].aread > lastRead )
                break;
            Write_Overlap( out, allOvls + j, tbytes );
        }

        rewind( out );
        fwrite( &nhits, sizeof( int64 ), 1, out );
        fclose( out );

        // dump out reads to second overlap file
        out = fopen( path2, "w" );
        if ( out == NULL )
        {
            fprintf( stderr, "[ERROR] - Write_Overlap_Buffer: Cannot open file %s for writing\n", path2 );
            exit( 1 );
        }

        nhits = 0;
        fwrite( &nhits, sizeof( int64 ), 1, out );
        fwrite( &tspace, sizeof( int ), 1, out );

        for ( ; j < nallOvls; j++, nhits++ )
            Write_Overlap( out, allOvls + j, tbytes );

        rewind( out );
        fwrite( &nhits, sizeof( int64 ), 1, out );
        fclose( out );

        // cleanup
        free( path1 );
        free( path2 );
    }

    // cleanup
    free( allOvls );

    free(aroot);
    free(broot);
}

void Reset_Overlap_Buffer( Align_Spec* spec )
{
    Overlap_IO_Buffer* buf = OVL_IO_Buffer( spec );
    int nthreads           = Num_Threads( spec );

    int i;
    for ( i = 0; i < nthreads; i++ )
    {
        buf[ i ].otop = 0;
        buf[ i ].ttop = 0;
    }
}
