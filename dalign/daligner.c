/************************************************************************************\
*                                                                                    *
 * Copyright (c) 2014, Dr. Eugene W. Myers (EWM). All rights reserved.                *
 *                                                                                    *
 * Redistribution and use in source and binary forms, with or without modification,   *
 * are permitted provided that the following conditions are met:                      *
 *                                                                                    *
 *  · Redistributions of source code must retain the above copyright notice, this     *
 *    list of conditions and the following disclaimer.                                *
 *                                                                                    *
 *  · Redistributions in binary form must reproduce the above copyright notice, this  *
 *    list of conditions and the following disclaimer in the documentation and/or     *
 *    other materials provided with the distribution.                                 *
 *                                                                                    *
 *  · The name of EWM may not be used to endorse or promote products derived from     *
 *    this software without specific prior written permission.                        *
 *                                                                                    *
 * THIS SOFTWARE IS PROVIDED BY EWM ”AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,    *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND       *
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL EWM BE LIABLE   *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS  *
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN  *
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                      *
 *                                                                                    *
 * For any issues regarding this software and its use, contact EWM at:                *
 *                                                                                    *
 *   Eugene W. Myers Jr.                                                              *
 *   Bautzner Str. 122e                                                               *
 *   01099 Dresden                                                                    *
 *   GERMANY                                                                          *
 *   Email: gene.myers@gmail.com                                                      *
 *                                                                                    *
 \************************************************************************************/

/*********************************************************************************************\
 *
 *  Find all local alignment between long, noisy DNA reads:
 *    Compare sequences in 'subject' database against those in the list of 'target' databases
 *    searching for local alignments of 1000bp or more (defined constant MIN_OVERLAP in
 *    filter.c).  Subject is compared in both orientations againt each target.  An output
 *    stream of 'Overlap' records (see align.h) is written in binary to the standard output,
 *    each encoding a given found local alignment between two of the sequences.  The -v
 *    option turns on a verbose reporting mode that gives statistics on each major stage.
 *
 *    There cannot be more than 65,535 reads in a given db, and each read must be less than
 *    66,535 characters long.
 *
 *    The filter operates by looking for a pair of diagonal bands of width 2^'s' that contain
 *    a collection of exact matching 'k'-mers between the two sequences, such that the total
 *    number of bases covered by 'k'-mer hits is 'h'.  k cannot be larger than 15 in the
 *    current implementation.
 *
 *    Some k-mers are significantly over-represented (e.g. homopolymer runs).  These are
 *    suppressed as seed hits, with the parameter 'm' -- any k-mer that occurs more than
 *    'm' times in either the subject or target is not counted as a seed hit.  If the -m
 *    option is absent then no k-mer is suppressed.
 *
 *    For each subject, target pair, say XXX and YYY, the program outputs a file containing
 *    overlaps of the form XXX.YYY.[C|N]#.las where C implies that the reads in XXX were
 *    complemented and N implies they were not (both comparisons are performed), and # is
 *    the thread that detected and wrote out the collection of overlaps.  For example, if
 *    NTHREAD in the program is 4, then 8 files are output for each subject, target pair.
 *
 *  Author:  Gene Myers
 *  Date  :  June 1, 2014
 *
 *********************************************************************************************/

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/param.h>
#if defined( BSD )
#include <sys/sysctl.h>
#endif

extern char* optarg;
extern int optind, opterr, optopt;

#include "align.h"
#include "db/DB.h"
#include "filter.h"
#include "lib/compression.h"
#include "lib/dmask.h"
#include "lib/tracks.h"

static void usage()
{
    fprintf( stderr, "usage:  \n" );
    fprintf( stderr, "daligner [-vbAIOT] [-k<int(14)>] [-w<int(6)>] [-h<int(35)>] [-t<int>] [-M<int>]\n" );
    fprintf( stderr, "         [-e<double(.70)] [-l<int(1000)>] [-s<int(100)>] [-H<int>] [-j<int>]\n" );
#ifdef DMASK
    fprintf( stderr, "         [-D<host:port>]\n" );
#endif
    fprintf( stderr, "         [-m<track>]+ <subject:db|dam> <target:db|dam> ...\n" );
    fprintf( stderr, "         [-r<int()1>]\n" );
    fprintf( stderr, "options: -v ... verbose\n" );
    fprintf( stderr, "         -b ... data has a strong compositional bias (e.g. >65%% AT rich)\n" );
    fprintf( stderr, "         -A ... asymmetric, for block X and Y the symmetric alignments for Y vs X are suppressed.\n" );
    fprintf( stderr, "         -I ... identity, overlaps of the same read will found and reported\n" );
    fprintf( stderr, "         -O ... only identity reads will be reported\n" );
    fprintf( stderr, "         -k ... kmer length (defaults: raw pacbio reads: 14, corrected reads: 25, mapping applications: 20)\n" );
    fprintf( stderr, "         -w ... diagonal band width (default: 6)\n" );
    fprintf( stderr, "         -h ... hit theshold (in bp.s, default: 35\n" );
    fprintf( stderr, "         -t ... tuple supression frequency, i.e. suppresses the use of any k-mer that occurs more than t times in either the subject or target block\n" );
    fprintf( stderr, "         -M ... specify the memory usage limit (in GB). Automatically adjust the -t parameter\n" );
    fprintf( stderr, "         -e ... average correlation rate for local alignments (default: 0.5). Must be in [.5,1.)\n" );
    fprintf( stderr, "         -l ... minimum alignment length (defaul: 1000)\n" );
    fprintf( stderr, "         -s ... trace spacing, i.e. report trace point every -s base pairs (default: raw pacbio reads: 100, corrected reads: 500)\n" );
    fprintf( stderr, "         -H ... report only overlaps where the a-read is over -H base pairs long\n" );
#ifdef DMASK
    fprintf( stderr, "         -D ... set up host and port where the dynamic mask server is running (default port: %d)\n", DMASK_DEFAULT_PORT );
#endif
    fprintf( stderr, "         -m ... specify an interval track that is to be softmasked\n" );
    fprintf( stderr, "         -r ... run identifier (default: 1). i.e. all overlap files of Block X a written to a subdirectory: dRUN-IDENTIFIER_X\n" );
    fprintf( stderr, "         -j ... number of threads (default: 4). Must be a power of 2!\n" );
    fprintf( stderr, "         -T ... disable trace points (default: enabled)\n" );
}

int VERBOSE; //   Globally visible to filter.c
int BIASED;
int MINOVER;
int HGAP_MIN;
int SYMMETRIC;
int IDENTITY;
int ONLY_IDENTITY;
int NTHREADS;
int NSHIFT;

uint64 MEM_LIMIT;
uint64 MEM_PHYSICAL;

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

int64 Merge_Size( HITS_DB* block, int mtop )
{
    Event ev[ mtop + 1 ];
    Event* heap[ mtop + 2 ];
    int r, mhalf;
    int64 nsize;

    {
        HITS_TRACK* track;
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
        HITS_TRACK* track;

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

HITS_TRACK* Merge_Tracks( HITS_DB* block, int mtop, int64 nsize )
{
    HITS_TRACK* ntrack;
    Event ev[ mtop + 1 ];
    Event* heap[ mtop + 2 ];
    int r, mhalf;
    int64* anno;
    int* data;

    ntrack = (HITS_TRACK*)Malloc( sizeof( HITS_TRACK ), "Allocating merged track" );
    if ( ntrack == NULL )
        exit( 1 );
    ntrack->name = Strdup( "merge", "Allocating merged track" );
    ntrack->anno = anno = (int64*)Malloc( sizeof( int64 ) * ( block->nreads + 1 ), "Allocating merged track" );
    ntrack->data = data = (int*)Malloc( sizeof( int ) * nsize, "Allocating merged track" );
    ntrack->size        = sizeof( int );
    ntrack->next        = NULL;
    if ( anno == NULL || data == NULL || ntrack->name == NULL )
        exit( 1 );

    {
        HITS_TRACK* track;
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
        HITS_TRACK* track;

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

#ifdef DMASK
static int read_DB( HITS_DB* block, char* name, char** mask, int* mstat, int mtop, int kmer, DynamicMask* dm )
#else
static int read_DB( HITS_DB* block, char* name, char** mask, int* mstat, int mtop, int kmer, void* dm )
#endif
{
    int i, isdam, stop;

    isdam = Open_DB( name, block );
    if ( isdam < 0 )
        exit( 1 );

    stop = 0;
    for ( i = 0; i < mtop; i++ )
    {
        HITS_TRACK* track;
        int64* anno;
        int j;

        /*
        int status = Check_Track(block, mask[i]);
        if (status > mstat[i])
          mstat[i] = status;

        if (status < 0)
          continue;
        */

        stop += 1;
        if ( dm )
        {
            track = dm_load_track( block, dm, mask[ i ] );
        }
        else
        {
            if ( block->part > 0 )
            {
                track = track_load_block( block, mask[ i ] );
            }
            else
            {
                track = track_load( block, mask[ i ] );
            }
        }

        if ( track == NULL )
        {
            printf( "[ERROR] - Unable to load track %s!\n", mask[ i ] );
            exit( 1 );
        }

        mstat[ i ] = 0;

        anno = (int64*)( track->anno );
        for ( j = 0; j <= block->nreads; j++ )
            anno[ j ] /= sizeof( track_data );
    }

    if ( stop > 1 )
    {
        int64 nsize;
        HITS_TRACK* track;

        nsize = Merge_Size( block, stop );
        track = Merge_Tracks( block, stop, nsize );

        while ( block->tracks != NULL )
            Close_Track( block, block->tracks->name );

        block->tracks = track;
    }

    for ( i = 0; i < block->nreads; i++ )
        if ( block->reads[ i ].rlen < kmer )
        {
            fprintf( stderr, "[ERROR] - daligner: Block %s contains reads < %dbp long !  Run DBsplit.\n", name, kmer );
            exit( 1 );
        }

    Read_All_Sequences( block, 0 );

    return ( isdam );
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

static unsigned int dlog2( unsigned int x )
{
    unsigned int ans = 0;
    while ( x >>= 1 )
        ans++;
    return ans;
}

static void createSubdir( HITS_DB* block, int RUN_ID )
{
    int blockID = block->part;
    struct stat s;
    char* out;

    out = getDir( RUN_ID, blockID );

    int err = stat( out, &s );

    if ( err == -1 )
    {
        if ( errno == ENOENT )
        {
            mkdir( out, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH );
        }
        else
        {
            fprintf( stderr, "Cannot create output directory: %s\n", out );
            exit( 1 );
        }
    }
    else
    {
        if ( !S_ISDIR( s.st_mode ) )
        {
            fprintf( stderr, "Output directory name: \"%s\" exist - but its not a directory\n", out );
            exit( 1 );
        }
    }
}

int main( int argc, char* argv[] )
{
    HITS_DB _ablock, _bblock;
    HITS_DB *ablock = &_ablock, *bblock = &_bblock;

    char *afile, *bfile;
    char *aroot, *broot;
    void *aindex, *bindex;
    int alen, blen;
    Align_Spec* asettings;

#ifdef DMASK
    DynamicMask* dm = NULL;
    char* dm_arg    = NULL;
#else
    void* dm = NULL;
#endif

    int isdam;
    int MMAX, MTOP, *MSTAT;
    char** MASK;

    int KMER_LEN        = 14;
    int HIT_MIN         = 35;
    int BIN_SHIFT       = 6;
    int MAX_REPS        = 0;
    int HGAP_MIN        = 0;
    double AVE_ERROR    = .70;
    int SPACING         = 100;
    int RUN_ID          = 0;
    int NO_TRACE_POINTS = 0;

    MINOVER       = 1000; //   Globally visible to filter.c
    RUN_ID        = 1;
    NTHREADS      = 4;
    NSHIFT        = 2;
    IDENTITY      = 0;
    ONLY_IDENTITY = 0;
    SYMMETRIC     = 1;

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
    if ( MASK == NULL || MSTAT == NULL )
        exit( 1 );

    // parse arguments
    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "vbOTAIk:w:h:t:M:e:l:s:H:D:m:r:j:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'v':
                VERBOSE = 1;
                break;
            case 'b':
                BIASED = 1;
                break;
            case 'T':
                NO_TRACE_POINTS = 1;
                break;
            case 'I':
                IDENTITY = 1;
                break;
            case 'O':
                IDENTITY      = 1;
                ONLY_IDENTITY = 1;
                break;
            case 'A':
                SYMMETRIC = 0;
                break;
            case 'k':
                KMER_LEN = atoi( optarg );
                break;
            case 'w':
                BIN_SHIFT = atoi( optarg );
                break;
            case 'h':
                HIT_MIN = atoi( optarg );
                break;
            case 't':
                MAX_REPS = atoi( optarg );
                break;
            case 'H':
                HGAP_MIN = atoi( optarg );
                break;
            case 'e':
                AVE_ERROR = atof( optarg );
                break;
            case 'l':
                MINOVER = atoi( optarg );
                break;
            case 's':
                SPACING = atoi( optarg );
                break;
            case 'j':
            {
                int tmp = atoi( optarg );
                if ( tmp < 1 )
                {
                    fprintf( stderr, "invalid number of threads: %d. Must be a positive number.\n", tmp );
                    exit( 1 );
                }

                int e = dlog2( (unsigned int)tmp );
                int i, test;
                test = 1;
                for ( i = 0; i < e; i++ )
                    test *= 2;

                if ( test == tmp )
                {
                    NTHREADS = tmp;
                    NSHIFT   = e;
                }
                else
                {
                    fprintf( stderr, "invalid number of threads: %d. Must be a power of 2.\n", tmp );
                    exit( 1 );
                }
                break;
            }
#ifdef DMASK
            case 'D':
                dm_arg = optarg;
                break;
#endif
            case 'M':
            {
                int tmp = atoi( optarg );
                if ( tmp < 0 )
                    fprintf( stderr, "invalid memory limit of (%d)\n", tmp );
                MEM_LIMIT = tmp * 0x40000000ll;
            }
            break;
            case 'm':
                if ( MTOP >= MMAX )
                {
                    MMAX  = 1.2 * MTOP + 10;
                    MASK  = (char**)Realloc( MASK, MMAX * sizeof( char* ), "Reallocating mask track array" );
                    MSTAT = (int*)Realloc( MSTAT, MMAX * sizeof( int ), "Reallocating mask status array" );
                    if ( MASK == NULL || MSTAT == NULL )
                        exit( 1 );
                }
                MSTAT[ MTOP ] = -2;
                MASK[ MTOP ]  = optarg;
                MTOP++;
                break;
            case 'r':
                RUN_ID = atoi( optarg );
                if ( RUN_ID < 0 )
                {
                    fprintf( stderr, "invalid run id of %d\n", RUN_ID );
                    exit( 1 );
                }
                break;
            default:
                fprintf( stderr, "Unsupported option: %s\n", argv[ optind - 1 ] );
                usage();
                exit( 1 );
        }
    }

    if ( KMER_LEN < 0 )
    {
        fprintf( stderr, "invalid kmer length of %d\n", KMER_LEN );
        exit( 1 );
    }
    if ( BIN_SHIFT < 0 )
    {
        fprintf( stderr, "invalid log of bin width of %d\n", BIN_SHIFT );
        exit( 1 );
    }
    if ( HIT_MIN < 0 )
    {
        fprintf( stderr, "invalid hit threshold of %d\n", HIT_MIN );
        exit( 1 );
    }
    if ( MAX_REPS < 0 )
    {
        fprintf( stderr, "invalid tuple supression frequency of %d\n", MAX_REPS );
        exit( 1 );
    }
    if ( HGAP_MIN < 0 )
    {
        fprintf( stderr, "invalid HGAP threshold of %d\n", HGAP_MIN );
        exit( 1 );
    }
    if ( AVE_ERROR < .5 || AVE_ERROR >= 1. )
    {
        fprintf( stderr, "Average correlation must be in [.5,1.) (%g)\n", AVE_ERROR );
        exit( 1 );
    }
    if ( MINOVER < 0 )
    {
        fprintf( stderr, "invalid minimum alignmnet length of (%d)\n", MINOVER );
        exit( 1 );
    }
    if ( SPACING < 0 )
    {
        fprintf( stderr, "invalid trace spacing of (%d)\n", SPACING );
        exit( 1 );
    }
    if ( optind + 2 > argc )
    {
        fprintf( stderr, "[ERROR] - at least one target and one subject block are required\n\n" );
        usage();
        exit( 1 );
    }

    MINOVER *= 2;
    if ( Set_Filter_Params( KMER_LEN, BIN_SHIFT, MAX_REPS, HIT_MIN ) )
    {
        fprintf( stderr, "Illegal combination of filter parameters\n" );
        exit( 1 );
    }

#ifdef DMASK
    if ( dm_arg != NULL )
    {
        char* dm_host = dm_arg;
        char* port;
        int dm_port;

        if ( ( port = strchr( dm_host, ':' ) ) == NULL )
        {
            dm_port = DMASK_DEFAULT_PORT;
        }
        else
        {
            *port   = '\0';
            dm_port = atoi( port + 1 );
        }

        dm = dm_init( dm_host, dm_port );

        if ( dm == NULL )
        {
            fprintf( stderr, "failed to initialise dynamic mask\n" );
            exit( 1 );
        }
    }
    else
    {
        dm = NULL;
    }
#endif

    /* Read in the reads in A */

    afile = argv[ optind ];
    isdam = read_DB( ablock, afile, MASK, MSTAT, MTOP, KMER_LEN, dm );

    if ( isdam )
        aroot = Root( afile, ".dam" );
    else
        aroot = Root( afile, ".db" );

    // check if b-blocks belong to a different DB, if so unset SYMMETRIC flag!!!!
    optind++;
    if ( SYMMETRIC )
    {
        int i;
        broot = NULL;
        for ( i = optind; i < argc; i++ )
        {
            bfile = argv[ i ];
            if ( strcmp( afile, bfile ) != 0 )
            {
                broot = Root( bfile, ".dam" );
                if ( broot == NULL )
                    broot = Root( bfile, ".db" );

                if ( broot )
                {
                    char* adot;
                    char* bdot;

                    adot = strrchr( aroot, '.' );
                    if ( adot == NULL )
                        adot = aroot + strlen( aroot ) - 1;
                    bdot = strrchr( broot, '.' );
                    if ( bdot == NULL )
                        bdot = broot + strlen( broot ) - 1;

                    if ( strncmp( aroot, broot, MAX( adot - aroot + 1, bdot - broot + 1 ) ) != 0 )
                    {
                        if ( VERBOSE )
                            printf( "[WARNING] - Daligner is performed on different databases (%s - %s). SYMMETRIC option is disabled!\n", aroot, broot );
                        SYMMETRIC = 0;
                        break;
                    }
                }
            }
        }
    }

    /* Create subdirectory */
    createSubdir( ablock, RUN_ID );

    asettings = New_Align_Spec( AVE_ERROR, SPACING, ablock->freq, NTHREADS, SYMMETRIC, ONLY_IDENTITY, NO_TRACE_POINTS );

    /* Compare against reads in B in both orientations */
    {
        int i, j;
        aindex = NULL;
        broot  = NULL;
        for ( i = optind; i < argc; i++ )
        {
            bfile = argv[ i ];
            if ( strcmp( afile, bfile ) != 0 )
            {
                isdam = read_DB( bblock, bfile, MASK, MSTAT, MTOP, KMER_LEN, dm );
                if ( isdam )
                    broot = Root( bfile, ".dam" );
                else
                    broot = Root( bfile, ".db" );
            }

            if ( i == optind )
            {
                for ( j = 0; j < MTOP; j++ )
                {
                    if ( MSTAT[ j ] == -2 )
                        printf( "[WARNING]: daligner -m%s option given but no track found.\n", MASK[ j ] );
                    else if ( MSTAT[ j ] == -1 )
                        printf( "[WARNING]: daligner %s track not sync'd with relevant db.\n", MASK[ j ] );
                }

                if ( VERBOSE )
                    printf( "\nBuilding index for %s\n", aroot );
                aindex = Sort_Kmers( ablock, &alen );
            }

            if ( strcmp( afile, bfile ) != 0 )
            {
                if ( SYMMETRIC )
                    createSubdir( bblock, RUN_ID );
                if ( dm )
                    dm_send_next( dm, RUN_ID, ablock, aroot, bblock, broot );

                if ( VERBOSE )
                    printf( "\nBuilding index for %s\n", broot );

                bindex = Sort_Kmers( bblock, &blen );
                Match_Filter( aroot, ablock, broot, bblock, aindex, alen, bindex, blen, 0, asettings );

                bblock = complement_DB( bblock, 1 );

                if ( VERBOSE )
                    printf( "\nBuilding index for c(%s)\n", broot );

                bindex = Sort_Kmers( bblock, &blen );
                Match_Filter( aroot, ablock, broot, bblock, aindex, alen, bindex, blen, 1, asettings );

                int lastRead;
                if ( bblock->part < ablock->part )
                    lastRead = bblock->ufirst + bblock->nreads - 1;
                else
                    lastRead = ablock->ufirst + ablock->nreads - 1;
                Write_Overlap_Buffer( asettings, RUN_ID, aroot, broot, lastRead );
                Reset_Overlap_Buffer( asettings );

                if ( dm )
                {
                    int sent = dm_send_block_done( dm, RUN_ID, ablock, aroot, bblock, broot );

                    if ( VERBOSE && !sent )
                        printf( "results not reported to dmask server\n" );
                }
                free( broot );
            }
            else
            {
                if ( dm )
                    dm_send_next( dm, RUN_ID, ablock, aroot, ablock, aroot );

                Match_Filter( aroot, ablock, aroot, ablock, aindex, alen, aindex, alen, 0, asettings );

                bblock = complement_DB( ablock, 0 );

                if ( VERBOSE )
                    printf( "\nBuilding index for c(%s)\n", aroot );

                bindex = Sort_Kmers( bblock, &blen );
                Match_Filter( aroot, ablock, aroot, bblock, aindex, alen, bindex, blen, 1, asettings );

                Write_Overlap_Buffer( asettings, RUN_ID, aroot, aroot, ablock->ufirst + ablock->nreads - 1 );
                Reset_Overlap_Buffer( asettings );

                if ( dm )
                {
                    int sent = dm_send_block_done( dm, RUN_ID, ablock, aroot, ablock, aroot );

                    if ( VERBOSE && !sent )
                        printf( "results not reported to dmask server\n" );
                }

                bblock->reads = NULL; //  ablock & bblock share "reads" vector, don't let Close_DB
                                      //     free it !
            }
            Close_DB( bblock );
        }
    }

    exit( 0 );
}
