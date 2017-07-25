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

/*******************************************************************************************
 *
 *  Split a .db into a set of sub-database blocks for use by the Dazzler:
 *     Divide the database <path>.db conceptually into a series of blocks referable to on the
 *     command line as <path>.1.db, <path>.2.db, ...  If the -x option is set then all reads
 *     less than the given length are ignored, and if the -a option is not set then secondary
 *     reads from a given well are also ignored.  The remaining reads are split amongst the
 *     blocks so that each block is of size -s * 1Mbp except for the last which necessarily
 *     contains a smaller residual.  The default value for -s is 400Mbp because blocks of this
 *     size can be compared by our "overlapper" dalign in roughly 16Gb of memory.  The blocks
 *     are very space efficient in that their sub-index of the master .idx is computed on the
 *     fly when loaded, and the .bps file of base pairs is shared with the master DB.  Any
 *     tracks associated with the DB are also computed on the fly when loading a database block.
 *
 *  Author:  Gene Myers
 *  Date  :  September 2013
 *  Mod   :  New splitting definition to support incrementality, and new stub file format
 *  Date  :  April 2014
 *
 ********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "DB.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

#define DEF_ARG_S 200
#define DEF_ARG_F 0

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    fprintf( stderr, "usage: [-s<int(%d)>] <path:db|dam>\n", DEF_ARG_S );
    fprintf( stderr, "         -s ... set block size of -s * 1Mbp (default: %dMBs)\n", DEF_ARG_S );
    fprintf( stderr, "         -f ... force Yes on all interactive queries\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db, dbs;
    int64 dbpos;
    FILE *dbfile, *ixfile;
    int status;

    int force = DEF_ARG_F;
    int SIZE  = DEF_ARG_S;

    // parse arguments
    {
        int c;
        opterr = 0;

        while ( ( c = getopt( argc, argv, "s:f" ) ) != -1 )
        {
            switch ( c )
            {
                case 'f':
                    force = 1;
                    break;

                case 's':
                {
                    SIZE = atoi( optarg );
                    if ( SIZE <= 0 )
                    {
                        fprintf( stderr, "invalid split size value of %d\n", SIZE );
                        exit( 1 );
                    }
                }
                break;
                default:
                    fprintf( stderr, "[ERROR] Unsupported option: %s\n", argv[ optind ] );
                    usage();
                    exit( 1 );
            }
        }

        if ( optind + 1 > argc )
        {
            fprintf( stderr, "[ERROR] - Database is required!\n" );
            usage();
            exit( 1 );
        }
    }

    //  Open db

    status = Open_DB( argv[ optind ], &db );
    if ( status < 0 )
    {
        fprintf( stderr, "[ERROR] Cannot open database %s\n", argv[ optind ] );
        exit( 1 );
    }
    if ( db.part > 0 )
    {
        fprintf( stderr, "[ERROR] Cannot be called on a block: %s\n", argv[ optind ] );
        exit( 1 );
    }

    {
        char *pwd, *root;
        char buffer[ 2 * MAX_NAME + 100 ];
        int nfiles;
        int i, nblocks;

        pwd = PathTo( argv[ optind ] );
        if ( status )
        {
            root   = Root( argv[ optind ], ".dam" );
            dbfile = Fopen( Catenate( pwd, "/", root, ".dam" ), "r+" );
        }
        else
        {
            root   = Root( argv[ optind ], ".db" );
            dbfile = Fopen( Catenate( pwd, "/", root, ".db" ), "r+" );
        }
        ixfile = Fopen( Catenate( pwd, PATHSEP, root, ".idx" ), "r+" );
        if ( dbfile == NULL || ixfile == NULL )
            exit( 1 );
        free( pwd );
        free( root );

        if ( fscanf( dbfile, DB_NFILE, &nfiles ) != 1 )
            SYSTEM_ERROR
        for ( i = 0; i < nfiles; i++ )
            if ( fgets( buffer, 2 * MAX_NAME + 100, dbfile ) == NULL )
                SYSTEM_ERROR

        if ( fread( &dbs, sizeof( HITS_DB ), 1, ixfile ) != 1 )
            SYSTEM_ERROR

        dbpos = ftello( dbfile );
        if ( fscanf( dbfile, DB_NBLOCK, &nblocks ) == 1 && !force )
        {
            printf( "You are about to overwrite the current partition settings. This\n" );
            printf( "will invalidate any tracks, overlaps, and other derivative files.\n" );
            printf( "Are you sure you want to proceed? [Y/N] " );
            fflush( stdout );

            if ( fgets( buffer, 100, stdin ) == NULL )
                SYSTEM_ERROR

            if ( index( buffer, 'n' ) != NULL || index( buffer, 'N' ) != NULL )
            {
                printf( "Aborted\n" );
                fflush( stdout );
                fclose( dbfile );
                exit( 1 );
            }
        }

        fseeko( dbfile, dbpos, SEEK_SET );
        fprintf( dbfile, DB_NBLOCK, 0 );
        fprintf( dbfile, DB_PARAMS, (int64)SIZE );
    }

    {
        HITS_READ* reads = db.reads;
        int nreads       = db.ureads;
        int64 size, totlen;
        int nblock, ireads, rlen, fno;
        int i;

        size = SIZE * 1000000ll;

        nblock = 0;
        totlen = 0;
        ireads = 0;
        fprintf( dbfile, DB_BDATA, 0 );
        for ( i = 0; i < nreads; i++ )
        {
            rlen = reads[ i ].rlen;
            ireads += 1;
            totlen += rlen;
            if ( totlen >= size )
            {
                fprintf( dbfile, DB_BDATA, i + 1 );
                totlen = 0;
                ireads = 0;
                nblock += 1;
            }
        }

        if ( ireads > 0 )
        {
            fprintf( dbfile, DB_BDATA, nreads );
            nblock += 1;
        }

        fno = fileno( dbfile );
        if ( ftruncate( fno, ftello( dbfile ) ) < 0 )
            SYSTEM_ERROR

        fseeko( dbfile, dbpos, SEEK_SET );

        fprintf( dbfile, DB_NBLOCK, nblock );

        rewind( ixfile );
        fwrite( &dbs, sizeof( HITS_DB ), 1, ixfile );
    }

    fclose( ixfile );
    fclose( dbfile );
    Close_DB( &db );

    return 0;
}
