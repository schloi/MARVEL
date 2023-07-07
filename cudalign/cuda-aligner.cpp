
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cuda/utils.cuh>
#include <future>
#include <string>
#include <sys/param.h>
#if defined( BSD )
#include <sys/sysctl.h>
#endif

#include "cuda-match-filter.h"
#include "cuda/resource-manager.h"

#ifndef MAX_CONCURRENCY
#define MAX_CONCURRENCY 1
#endif

#define DAZZ_TRACK HITS_TRACK
// #define BLOCK_SYMBOL     '_'

static void usage(FILE* out, const char* Prog_Name)
{
    fprintf( out, "Usage: %s [-vaAI] [-k<int(16)>] [-w<int(6)>] [-h<int(50)>] [-t<int>] [-M<int>]\n", Prog_Name);
    fprintf( out, "       %*s [-e<double(.75)] [-l<int(1500)>] [-s<int(100)>] [-H<int>]\n", (int)strlen( Prog_Name ), "" );
    fprintf( out, "       %*s [-T<int(4)>] [-P<dir(/tmp)>] [-m<track>]+\n", (int)strlen( Prog_Name ), "");
    fprintf( out, "       %*s <subject:db|dam> <target:db|dam> ...\n", (int)strlen( Prog_Name ), "");
    fprintf( out, "\n" );
    fprintf( out, "      -k: k-mer size (must be <= 32).\n" );
    fprintf( out, "      -w: Look for k-mers in averlapping bands of size 2^-w.\n" );
    fprintf( out, "      -h: A seed hit if the k-mers in band cover >= -h bps in the" );
    fprintf( out, " targest read.\n" );
    fprintf( out, "      -t: Ignore k-mers that occur >= -t times in a block.\n" );
    fprintf( out, "      -M: Use only -M GB of memory by ignoring most frequent k-mers.\n" );
    fprintf( out, "\n" );
    fprintf( out, "      -e: Look for alignments with -e percent similarity.\n" );
    fprintf( out, "      -l: Look for alignments of length >= -l.\n" );
    fprintf( out, "      -s: The trace point spacing for encoding alignments.\n" );
    fprintf( out, "      -H: HGAP option: align only target reads of length >= -H.\n" );
    fprintf( out, "\n" );
    fprintf( out, "      -T: Use -T threads.\n" );
    fprintf( out, "      -P: Do block level sorts and merges in directory -P.\n" );
    fprintf( out, "      -m: Soft mask the blocks with the specified mask.\n" );
    fprintf( out, "\n" );
    fprintf( out, "      -v: Verbose mode, output statistics as proceed.\n" );
    fprintf( out, "      -a: sort .las by A-read,A-position pairs for map usecase\n" );
    fprintf( out, "          off => sort .las by A,B-read pairs for overlap piles\n" );
    fprintf( out, "      -A: Compare subjet to target, but not vice versa.\n" );
    fprintf( out, "      -I: Compare reads to themselves\n" );
    fprintf( out, "\n" );
    fprintf( out, "      -g: Set the GPU to be used. Default: 0\n" );
    fprintf( out, "      -S: Number of GPU Streams to be used: Default: 8\n" );
    fprintf( out, "      -L: Max Read Length that the GPU should handle. It is\n" );
    fprintf( out, "          constraint by device memory. Default 1000000\n" );
}

int VERBOSE; //   Globally visible to filter.c
int MINOVER;
int SYMMETRIC;
int IDENTITY;

uint64_t MEM_LIMIT;
uint64_t MEM_PHYSICAL;

/*  Adapted from code by David Robert Nadeau (http://NadeauSoftware.com) licensed under
 *     "Creative Commons Attribution 3.0 Unported License"
 *          (http://creativecommons.org/licenses/by/3.0/deed.en_US)
 *
 *   I removed Windows options, reformated, and return int64 instead of size_t
 */

static int64 getMemorySize()
{
#if defined( CTL_HW ) && ( defined( HW_MEMSIZE ) || defined( HW_PHYSMEM64 ) )

    // OSX, NetBSD, OpenBSD

    int mib[ 2 ];
    size_t size = 0;
    size_t len  = sizeof( size );

    mib[ 0 ] = CTL_HW;
#if defined( HW_MEMSIZE )
    mib[ 1 ] = HW_MEMSIZE; // OSX
#elif defined( HW_PHYSMEM64 )
    mib[ 1 ] = HW_PHYSMEM64; // NetBSD, OpenBSD
#endif
    if ( sysctl( mib, 2, &size, &len, NULL, 0 ) == 0 )
        return ( (size_t)size );
    return ( 0 );

#elif defined( _SC_AIX_REALMEM )

    // AIX

    return ( (size_t)sysconf( _SC_AIX_REALMEM ) * ( (size_t)1024L ) );

#elif defined( _SC_PHYS_PAGES ) && defined( _SC_PAGESIZE )

    // FreeBSD, Linux, OpenBSD, & Solaris

    size_t size = 0;

    size = (size_t)sysconf( _SC_PHYS_PAGES );
    return ( size * ( (size_t)sysconf( _SC_PAGESIZE ) ) );

#elif defined( _SC_PHYS_PAGES ) && defined( _SC_PAGE_SIZE )

    // ? Legacy ?

    size_t size = 0;

    size = (size_t)sysconf( _SC_PHYS_PAGES );
    return ( size * ( (size_t)sysconf( _SC_PAGE_SIZE ) ) );

#elif defined( CTL_HW ) && ( defined( HW_PHYSMEM ) || defined( HW_REALMEM ) )

    // DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX

    int mib[ 2 ];
    unsigned int size = 0;
    size_t len        = sizeof( size );

    mib[ 0 ] = CTL_HW;
#if defined( HW_REALMEM )
    mib[ 1 ] = HW_REALMEM; // FreeBSD
#elif defined( HW_PYSMEM )
    mib[ 1 ] = HW_PHYSMEM; // Others
#endif
    if ( sysctl( mib, 2, &size, &len, NULL, 0 ) == 0 )
        return (size_t)size;
    return ( 0 );

#else

    return ( 0 );

#endif
}

typedef struct
{
    int* ano;
    int* end;
    int idx;
    int out;
} Event;

static void reheap( int s, Event** heap, int hsize )
{
    int c, l, r;
    Event *hs, *hr, *hl;

    c  = s;
    hs = heap[ s ];
    while ( ( l = 2 * c ) <= hsize )
    {
        r  = l + 1;
        hl = heap[ l ];
        hr = heap[ r ];
        if ( hr->idx > hl->idx )
        {
            if ( hs->idx > hl->idx )
            {
                heap[ c ] = hl;
                c         = l;
            }
            else
                break;
        }
        else
        {
            if ( hs->idx > hr->idx )
            {
                heap[ c ] = hr;
                c         = r;
            }
            else
                break;
        }
    }
    if ( c != s )
        heap[ c ] = hs;
}

static int64 merge_size( DAZZ_DB* block, int mtop )
{
    Event ev[ mtop + 1 ];
    Event* heap[ mtop + 2 ];
    int r, mhalf;
    int64 nsize;

    {
        DAZZ_TRACK* track;
        int i;

        track = block->tracks;
        for ( i = 0; i < mtop; i++ )
        {
            ev[ i ].ano   = ( (int*)( track->data ) ) + ( (int64*)( track->anno ) )[ 0 ];
            ev[ i ].out   = 1;
            heap[ i + 1 ] = ev + i;
            track         = track->next;
        }
        ev[ mtop ].idx   = INT32_MAX;
        heap[ mtop + 1 ] = ev + mtop;
    }

    mhalf = mtop / 2;

    nsize = 0;
    for ( r = 0; r < block->nreads; r++ )
    {
        int i, level, hsize;
        DAZZ_TRACK* track;

        track = block->tracks;
        for ( i = 0; i < mtop; i++ )
        {
            ev[ i ].end = ( (int*)( track->data ) ) + ( (int64*)( track->anno ) )[ r + 1 ];
            if ( ev[ i ].ano < ev[ i ].end )
                ev[ i ].idx = *( ev[ i ].ano );
            else
                ev[ i ].idx = INT32_MAX;
            track = track->next;
        }
        hsize = mtop;

        for ( i = mhalf; i > 1; i-- )
            reheap( i, heap, hsize );

        level = 0;
        while ( 1 )
        {
            Event* p;

            reheap( 1, heap, hsize );

            p = heap[ 1 ];
            if ( p->idx == INT32_MAX )
                break;

            p->out = 1 - p->out;
            if ( p->out )
            {
                level -= 1;
                if ( level == 0 )
                    nsize += 1;
            }
            else
            {
                if ( level == 0 )
                    nsize += 1;
                level += 1;
            }
            p->ano += 1;
            if ( p->ano >= p->end )
                p->idx = INT32_MAX;
            else
                p->idx = *( p->ano );
        }
    }

    return ( nsize );
}

static DAZZ_TRACK* merge_tracks( DAZZ_DB* block, int mtop, int64 nsize )
{
    DAZZ_TRACK* ntrack;
    Event ev[ mtop + 1 ];
    Event* heap[ mtop + 2 ];
    int r, mhalf;
    int64* anno;
    int* data;

    ntrack = (DAZZ_TRACK*)Malloc( sizeof( DAZZ_TRACK ), "Allocating merged track" );
    if ( ntrack == NULL )
        exit( 1 );
    ntrack->name = Strdup( "merge", "Allocating merged track" );
    ntrack->anno = anno = (int64*)Malloc( sizeof( int64 ) * ( block->nreads + 1 ), "Allocating merged track" );
    ntrack->data = data = (int*)Malloc( sizeof( int ) * nsize, "Allocating merged track" );
    ntrack->size        = sizeof( int );
    ntrack->next        = NULL;
    // ntrack->loaded = 1;
    if ( anno == NULL || data == NULL || ntrack->name == NULL )
        exit( 1 );

    {
        DAZZ_TRACK* track;
        int i;

        track = block->tracks;
        for ( i = 0; i < mtop; i++ )
        {
            ev[ i ].ano   = ( (int*)( track->data ) ) + ( (int64*)( track->anno ) )[ 0 ];
            ev[ i ].out   = 1;
            heap[ i + 1 ] = ev + i;
            track         = track->next;
        }
        ev[ mtop ].idx   = INT32_MAX;
        heap[ mtop + 1 ] = ev + mtop;
    }

    mhalf = mtop / 2;

    nsize = 0;
    for ( r = 0; r < block->nreads; r++ )
    {
        int i, level, hsize;
        DAZZ_TRACK* track;

        anno[ r ] = nsize;

        track = block->tracks;
        for ( i = 0; i < mtop; i++ )
        {
            ev[ i ].end = ( (int*)( track->data ) ) + ( (int64*)( track->anno ) )[ r + 1 ];
            if ( ev[ i ].ano < ev[ i ].end )
                ev[ i ].idx = *( ev[ i ].ano );
            else
                ev[ i ].idx = INT32_MAX;
            track = track->next;
        }
        hsize = mtop;

        for ( i = mhalf; i > 1; i-- )
            reheap( i, heap, hsize );

        level = 0;
        while ( 1 )
        {
            Event* p;

            reheap( 1, heap, hsize );

            p = heap[ 1 ];
            if ( p->idx == INT32_MAX )
                break;

            p->out = 1 - p->out;
            if ( p->out )
            {
                level -= 1;
                if ( level == 0 )
                    data[ nsize++ ] = p->idx;
            }
            else
            {
                if ( level == 0 )
                    data[ nsize++ ] = p->idx;
                level += 1;
            }
            p->ano += 1;
            if ( p->ano >= p->end )
                p->idx = INT32_MAX;
            else
                p->idx = *( p->ano );
        }
    }
    anno[ r ] = nsize;

    return ( ntrack );
}

static void complement( char* s, int len )
{
    char* t;
    int c;

    t = s + ( len - 1 );
    while ( s < t )
    {
        c  = *s;
        *s = (char)( 3 - *t );
        *t = (char)( 3 - c );
        s += 1;
        t -= 1;
    }
    if ( s == t )
        *s = (char)( 3 - *s );
}

static HITS_DB* complement_DB( HITS_DB* block, int inplace )
{
    static HITS_DB _cblock, *cblock = &_cblock;
    int nreads;
    HITS_READ* reads;
    char* seq;

    nreads = block->nreads;
    reads  = block->reads;

    if ( inplace )
    {
        seq    = (char*)block->bases;
        cblock = block;
    }
    else
    {
        seq = (char*)Malloc( block->reads[ nreads ].boff + 1, "Allocating dazzler sequence block" );
        if ( seq == NULL )
            exit( 1 );
        *seq++ = 4;
        memcpy( seq, block->bases, block->reads[ nreads ].boff );
        *cblock        = *block;
        cblock->bases  = (void*)seq;
        cblock->tracks = NULL;
        cblock->path   = static_cast<char*>( malloc( strlen( block->path ) * sizeof( char ) + 1 ) );
        strcpy( cblock->path, block->path );
    }

    {
        int i;
        float x;

        x                 = cblock->freq[ 0 ];
        cblock->freq[ 0 ] = cblock->freq[ 3 ];
        cblock->freq[ 3 ] = x;

        x                 = cblock->freq[ 1 ];
        cblock->freq[ 1 ] = cblock->freq[ 2 ];
        cblock->freq[ 2 ] = x;

        for ( i = 0; i < nreads; i++ )
            complement( seq + reads[ i ].boff, reads[ i ].rlen );
    }

    {
        HITS_TRACK *src, *trg;
        int *data, *tata;
        int i, x, rlen;
        int64 *tano, *anno;
        int64 j, k;

        for ( src = block->tracks; src != NULL; src = src->next )
        {
            tano = (int64*)src->anno;
            tata = (int*)src->data;

            if ( inplace )
            {
                data = tata;
                anno = tano;
                trg  = src;
            }
            else
            {
                data = (int*)Malloc( sizeof( int ) * tano[ nreads ], "Allocating dazzler interval track data" );
                anno = (int64*)Malloc( sizeof( int64 ) * ( nreads + 1 ), "Allocating dazzler interval track index" );
                trg  = (HITS_TRACK*)Malloc( sizeof( HITS_TRACK ), "Allocating dazzler interval track header" );
                if ( data == NULL || trg == NULL || anno == NULL )
                    exit( 1 );

                trg->name = Strdup( src->name, "Copying track name" );
                if ( trg->name == NULL )
                    exit( 1 );

                trg->size      = 4;
                trg->anno      = (void*)anno;
                trg->data      = (void*)data;
                trg->next      = cblock->tracks;
                cblock->tracks = trg;
            }

            for ( i = 0; i < nreads; i++ )
            {
                rlen      = reads[ i ].rlen;
                anno[ i ] = tano[ i ];
                j         = tano[ i + 1 ] - 1;
                k         = tano[ i ];
                while ( k < j )
                {
                    x           = tata[ j ];
                    data[ j-- ] = rlen - tata[ k ];
                    data[ k++ ] = rlen - x;
                }
                if ( k == j )
                    data[ k ] = rlen - tata[ k ];
            }
            anno[ nreads ] = tano[ nreads ];
        }
    }
    return ( cblock );
}

static int read_DB( DAZZ_DB* block, char* name, char** mask, int* mstat, int mtop, int kmer )
{
    int i, isdam, stop;

    isdam = Open_DB( name, block );
    if ( isdam < 0 )
        exit( 1 );

    /*
  for (i = 0; i < mtop; i++)
    { status = Check_Track(block,mask[i]); // ,&kind);
      if (status >= 0)
        if (kind == MASK_TRACK)
          mstat[i] = 0;
        else
          { if (mstat[i] != 0)
              mstat[i] = -3;
          }
      else
        { if (mstat[i] == -2)
            mstat[i] = status;
        }
      if (status == 0 && kind == MASK_TRACK)
        Open_Track(block,mask[i]);
    }
    */

    // Trim_DB(block);

    stop = 0;
    for ( i = 0; i < mtop; i++ )
    {
        DAZZ_TRACK* track;
        int64* anno;
        int j;

        /*
          status = Check_Track(block,mask[i]); // ,&kind);
          if (status < 0 || kind != MASK_TRACK)
            continue;
            */

        stop += 1;

        if ( block->part > 0 )
        {
            track = track_load_block( block, mask[ i ] );
        }
        else
        {
            track = track_load( block, mask[ i ] );
        }

        if ( track == NULL )
        {
            printf( "unable to load track %s\n", mask[ i ] );
            exit( 1 );
        }

        mstat[ i ] = 0;

        anno = (int64*)( track->anno );
        for ( j = 0; j <= block->nreads; j++ )
            anno[ j ] /= sizeof( int );
    }

    if ( stop > 1 )
    {
        int64 nsize;
        DAZZ_TRACK* track;

        nsize = merge_size( block, stop );
        track = merge_tracks( block, stop, nsize );

        while ( block->tracks != NULL )
            Close_Track( block, block->tracks->name );

        block->tracks = track;
    }

    for ( i = 0; i < block->nreads; i++ )
        if ( block->reads[ i ].rlen < kmer )
        {
            fprintf( stderr, "%s: Block %s contains reads < %dbp long !  Run DBsplit.\n", Prog_Name, name, kmer );
            exit( 1 );
        }

    // Load_All_Reads(block,0);
    Read_All_Sequences( block, 0 );

    return ( isdam );
}

/**
 * Structure to carry the information for the async match filter call
 */
typedef struct MatchFilterData
{
    Align_Spec* alignSpec;
    DAZZ_DB* aBlock;
    DAZZ_DB* bBlock;
    KmerPos* aIndex;
    KmerPos* bIndex;
    char* aRoot;
    char* bRoot;
    SequenceInfo* aDeviceBlock;
    SequenceInfo* bDeviceBlock;
    SequenceInfo* bDeviceComplementBlock;
} MatchFilterData;

/**
 * This function is used to trigger a asynchronous Block comparisson.
 *
 * @param data
 * @param alen A Length returned by ::Sort_Kmers call.
 * @param numberOfThreads The number of thread that should be used for computing the index
 * @param deviceId the GPU device id/number to be used
 * @param resourceManager The resource manager to be used
 * @param numberOfStrems The number of GPU stream to be used for the alignment
 * @param sortPath block level sorts and merges directory
 * @param threadNumber Thread number which this function is running in.
 */
void matchFilterAsync( MatchFilterData* data,
                       int alen,
                       int numberOfThreads,
                       int deviceId,
                       ResourceManager* resourceManager,
                       int numberOfStrems,
                       const std::string& sortPath,
                       int threadNumber )
{

    CUDA_SAFE_CALL( cudaSetDevice( deviceId ) );

    printf( "Thread %d: %s x %s\n", threadNumber, data->aRoot, data->bRoot );

    int blen;
    resourceManager->lockCpuResources();
    if ( !data->bIndex )
    {
        if ( VERBOSE )
            printf( "\nBuilding index for %s\n", data->bRoot );

        auto range   = rangeStartWithColor( "Sort_Kmers", 0xff2060ffu );
        data->bIndex = Sort_Kmers( data->bBlock, &blen );
        rangeEnd( range );
    }
    else
    {
        blen = alen;
    }
    resourceManager->unlockCpuResources();
    GPU_Match_Filter( data->aRoot,
                      data->aBlock,
                      data->bRoot,
                      data->bBlock,
                      data->aIndex,
                      alen,
                      &( data->bIndex ),
                      blen,
                      data->alignSpec,
                      numberOfThreads,
                      deviceId,
                      resourceManager,
                      numberOfStrems,
                      sortPath,
                      data->aDeviceBlock,
                      data->bDeviceBlock,
                      data->bDeviceComplementBlock );
};

int main( int argc, char* argv[] )
{
    DAZZ_DB _ablock;
    DAZZ_DB* ablock = &_ablock;
    char* afile;
    char *apath, *bpath;
    char *aroot, *broot;
    KmerPos *aindex, *bindex;
    int alen;
    int isdam;
    int MMAX, MTOP, *MSTAT;
    char** MASK;

    int KMER_LEN = 16;
    int BIN_SHIFT = 6;
    int MAX_REPS = 0;
    int HIT_MIN = 50;
    double AVE_ERROR = .75;
    int SPACING = 100;
    int NTHREADS = 4;
    int MAX_READ_LENGTH = -1;

    int deviceId        = 0;
    int numberOfStreams = 8;

    std::string SORT_PATH = "/tmp";

    {
        int i, j, k;
        int flags[ 128 ];
        char* eptr;

        ARG_INIT( "cudaligner" )

        MINOVER         = 1500; //   Globally visible to filter.c

        MEM_PHYSICAL = getMemorySize();
        MEM_LIMIT    = MEM_PHYSICAL;

        if ( MEM_PHYSICAL == 0 )
        {
            fprintf( stderr, "\nWarning: Could not get physical memory size\n" );
            fflush( stderr );
        }

        MTOP  = 0;
        MMAX  = 10;
        MASK  = (char**)Malloc( MMAX * sizeof( char* ), "Allocating mask track array" );
        MSTAT = (int*)Malloc( MMAX * sizeof( int ), "Allocating mask status array" );
        if ( MASK == nullptr || MSTAT == nullptr )
            exit( 1 );

        j = 1;
        for ( i = 1; i < argc; i++ )
            if ( argv[ i ][ 0 ] == '-' )
                switch ( argv[ i ][ 1 ] )
                {
                    default:
                        ARG_FLAGS( "vaAI" )
                        break;
                    case 'k':
                        ARG_POSITIVE( KMER_LEN, "K-mer length" )
                        if ( KMER_LEN > 32 )
                        {
                            fprintf( stderr, "%s: K-mer length must be 32 or less\n", Prog_Name );
                            exit( 1 );
                        }
                        break;
                    case 'g':
                        ARG_NON_NEGATIVE( deviceId, "Cuda Device ID" )
                        break;
                    case 'S':
                        ARG_POSITIVE( numberOfStreams, "Number of Cuda Streams" )
                        break;
                    case 'w':
                        ARG_POSITIVE( BIN_SHIFT, "Log of bin width" )
                        break;
                    case 'h':
                        ARG_POSITIVE( HIT_MIN, "Hit threshold (in bp.s)" )
                        break;
                    case 't':
                        ARG_POSITIVE( MAX_REPS, "Tuple supression frequency" )
                        break;
                    case 'e':
                        ARG_REAL( AVE_ERROR )
                        if ( AVE_ERROR < .65 || AVE_ERROR >= 1. )
                        {
                            fprintf( stderr, "%s: Average correlation must be in [.7,1.) (%g)\n", Prog_Name, AVE_ERROR );
                            exit( 1 );
                        }
                        break;
                    case 'l':
                        ARG_POSITIVE( MINOVER, "Minimum alignment length" )
                        break;
                    case 's':
                        ARG_POSITIVE( SPACING, "Trace spacing" )
                        break;
                    case 'M':
                    {
                        int limit;

                        ARG_NON_NEGATIVE( limit, "Memory allocation (in Gb)" )
                        MEM_LIMIT = limit * 0x40000000ll;
                        break;
                    }
                    case 'm':
                        if ( MTOP >= MMAX )
                        {
                            MMAX  = 1.2 * MTOP + 10;
                            MASK  = (char**)Realloc( MASK, MMAX * sizeof( char* ), "Reallocating mask track array" );
                            MSTAT = (int*)Realloc( MSTAT, MMAX * sizeof( int ), "Reallocating mask status array" );
                            if ( MASK == NULL || MSTAT == NULL )
                                exit( 1 );
                        }
                        MASK[ MTOP++ ] = argv[ i ] + 2;
                        break;
                    case 'P':
                        SORT_PATH = argv[ i ] + 2;
                        break;
                    case 'T':
                        ARG_POSITIVE( NTHREADS, "Number of threads" )
                        break;
                    case 'L':
                        ARG_POSITIVE( MAX_READ_LENGTH, "Max Read Length that the GPU can handle" )
                        break;
                }
            else
                argv[ j++ ] = argv[ i ];
        argc = j;

        cudaSetDevice( deviceId );

        VERBOSE   = flags[ (int)'v' ]; //  Globally declared in filter.h
        SYMMETRIC = 1 - flags[ (int)'A' ];
        IDENTITY  = flags[ (int)'I' ];

        if ( argc <= 2 )
        {
            usage(stderr, Prog_Name);
            exit( 1 );
        }

        for ( j = 0; j < MTOP; j++ )
            MSTAT[ j ] = -2;
    }

    MINOVER *= 2;
    Set_Filter_Params( KMER_LEN, BIN_SHIFT, MAX_REPS, HIT_MIN, NTHREADS, SPACING );
    Set_Radix_Params( NTHREADS, VERBOSE );

    // Create directory in SORT_PATH for file operations

    {
        char* newpath;

        newpath = (char*)Malloc( strlen( SORT_PATH.c_str() ) + 30, "Allocating sort path" );
        if ( newpath == NULL )
            exit( 1 );
        // sprintf(newpath,"%s/daligner.%d",SORT_PATH,getpid());
        sprintf( newpath, "%s", SORT_PATH.c_str() );

        if ( mkdir( newpath, S_IRWXU ) != 0 && errno != EEXIST )
        {
            fprintf( stderr, "%s: Could not create directory %s\n", Prog_Name, newpath );
            perror( "failed" );
            exit( 1 );
        }
        SORT_PATH = newpath;
        free( newpath );
    }

    // Read in the reads in A

    afile = argv[ 1 ];
    isdam = read_DB( ablock, afile, MASK, MSTAT, MTOP, KMER_LEN );
    if ( isdam )
        aroot = Root( afile, ".dam" );
    else
        aroot = Root( afile, ".db" );
    apath = PathTo( afile );

    // Compare against reads in B in both orientations

    {

        std::vector<MatchFilterData> matchFilterData;
        if ( VERBOSE )
            printf( "\nBuilding index for %s\n", aroot );

        auto range = rangeStartWithColor( "Sort_Kmers", 0xff2060ffu );
        aindex     = Sort_Kmers( ablock, &alen );
        rangeEnd( range );

        size_t maxReadLength = 0;
        if ( MAX_READ_LENGTH == -1 )
        {
            range = rangeStartWithColor( "Computing Longest Read", 0xff2060ffu );
            for ( int j = 0; j < ablock->nreads; j++ )
            {
                if ( maxReadLength < ablock->reads[ j ].rlen )
                {
                    maxReadLength = ablock->reads[ j ].rlen;
                }
            }
            rangeEnd( range );
        }

        for ( int i = 2; i < argc; i++ )
        {
            char* bfile = argv[ i ];

            broot       = Root( bfile, ".db" );
            bpath       = PathTo( bfile );
            auto bblock = static_cast<DAZZ_DB*>( malloc( sizeof( DAZZ_DB ) ) );
            if ( strcmp( bpath, apath ) != 0 || strcmp( broot, aroot ) != 0 )
            {
                bfile = Strdup( Catenate( bpath, "/", broot, "" ), "Allocating path" );
                read_DB( bblock, bfile, MASK, MSTAT, MTOP, KMER_LEN );
                if ( MAX_READ_LENGTH == -1 )
                {
                    range = rangeStartWithColor( "Computing Longest Read", 0xff2060ffu );
                    for ( int j = 0; j < bblock->nreads; j++ )
                    {
                        if ( maxReadLength < bblock->reads[ j ].rlen )
                        {
                            maxReadLength = bblock->reads[ j ].rlen;
                        }
                    }
                    rangeEnd( range );
                }

                free( bfile );
                bindex = nullptr;
            }
            else
            {
                free( broot );
                free( bblock );
                bblock = ablock;
                broot  = aroot;
                bindex = aindex;
            }

            free( bpath );

            MatchFilterData data = { .alignSpec              = New_Align_Spec( AVE_ERROR, SPACING, ablock->freq, 1, NTHREADS ),
                                     .aBlock                 = ablock,
                                     .bBlock                 = bblock,
                                     .aIndex                 = aindex,
                                     .bIndex                 = bindex,
                                     .aRoot                  = aroot,
                                     .bRoot                  = broot,
                                     .aDeviceBlock           = nullptr,
                                     .bDeviceBlock           = nullptr,
                                     .bDeviceComplementBlock = nullptr };

            matchFilterData.push_back( data );
        }

        aindex = nullptr;
        broot  = nullptr;
        if ( MAX_READ_LENGTH == -1 )
        {
            printf( "Longest Read has %zu bases. Using this value for memory allocation.\n", maxReadLength );
            MAX_READ_LENGTH = maxReadLength;
        }
        ResourceManager deviceMemoryManager( MAX_CONCURRENCY, MAX_CONCURRENCY, MAX_READ_LENGTH );

        INIT_TIMING
        START_TIMING
        deviceMemoryManager.allocWorkMemory();
        END_TIMING("allocWorkMemory")

        SequenceInfo* deviceABlock;

        START_TIMING
        deviceABlock = deviceMemoryManager.copyBlock2Device( ablock );
        END_TIMING("copyBlock2Device")

        std::for_each( matchFilterData.begin(), matchFilterData.end(), [ & ]( MatchFilterData& data ) {
            data.aDeviceBlock = deviceABlock;
            if ( data.aBlock == data.bBlock )
            {
                data.bDeviceBlock = data.aDeviceBlock;
            }
            else
            {
                data.bDeviceBlock = deviceMemoryManager.copyBlock2Device( data.bBlock );
            }

            DAZZ_DB* complement         = complement_DB( data.bBlock, 0 );
            data.bDeviceComplementBlock = deviceMemoryManager.copyBlock2Device( complement );

            complement->reads = nullptr;
            Close_DB( complement );
        } );

        std::vector<std::future<void>> futures;
        futures.reserve( matchFilterData.size() );
        for ( int i = 0; i < matchFilterData.size(); i++ )
        {
            futures.push_back(
                std::async( matchFilterAsync, matchFilterData.data() + i, alen, NTHREADS, deviceId, &deviceMemoryManager, numberOfStreams, SORT_PATH, i ) );
        }
        for ( auto& f : futures )
        {
            f.get();
        }

        std::for_each( matchFilterData.begin(), matchFilterData.end(), [ & ]( const MatchFilterData& data ) {
            for ( int i = 0; i < NTHREADS; i++ )
            {
                free( OVL_IO_Buffer( data.alignSpec )[ i ].ovls );
                if ( OVL_IO_Buffer( data.alignSpec )[ i ].trace )
                {
                    free( OVL_IO_Buffer( data.alignSpec )[ i ].trace );
                }
            }
            free( OVL_IO_Buffer( data.alignSpec ) );

            Free_Align_Spec( data.alignSpec );

            if ( data.bBlock != ablock )
            {
                Close_DB( data.bBlock );
                free( data.bBlock );
            }
            if ( data.bIndex != aindex )
            {
                free( data.bIndex );
            }
            if ( strcmp( data.bRoot, aroot ) != 0 )
            {
                free( data.bRoot );
            }
        } );

        Close_DB( ablock );
        free( aindex );
        free( apath );
        free( aroot );

        exit( 0 );
    }
}
