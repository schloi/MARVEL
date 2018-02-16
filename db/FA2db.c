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
 *  Add .fasta files to a DB:
 *     Adds the given fasta files in the given order to <path>.db.  If the db does not exist
 *     then it is created.  All .fasta files added to a given data base must have the same
 *     header format and follow Pacbio's convention.  A file cannot be added twice and this
 *     is enforced.  The command either builds or appends to the .<path>.idx and .<path>.bps
 *     files, where the index file (.idx) contains information about each read and their offsets
 *     in the base-pair file (.bps) that holds the sequences where each base is compessed
 *     into 2-bits.  The two files are hidden by virtue of their names beginning with a '.'.
 *     <path>.db is effectively a stub file with given name that contains an ASCII listing
 *     of the files added to the DB and possibly the block partitioning for the DB if DBsplit
 *     has been called upon it.
 *
 *  Author:  Gene Myers
 *  Date  :  May 2013
 *  Modify:  DB upgrade: now *add to* or create a DB depending on whether it exists, read
 *             multiple .fasta files (no longer a stdin pipe).
 *  Date  :  April 2014
 *
 ********************************************************************************************/

#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "DB.h"
#include "fileUtils.h"
#include "lib/tracks.h"
#include "FA2x.h"
#include "lib/utils.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

#define DEF_OPT_X 1000

extern char* optarg;
extern int optind, opterr, optopt;

static char number[ 128 ] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};


static void parseOptions( int argc, char* argv[], CreateContext* ctx );
static void usage( const char* progname );


void initPacbioRead(pacbio_read* read, int createTracks)
{
    read->maxPrologLen = MAX_NAME;
    read->maxSequenceLen = MAX_NAME + 60000;

    if ((read->prolog = (char*) malloc(read->maxPrologLen)) == NULL)
    {
        fprintf(stderr, "Unable to allocate prolog buffer\n");
        exit(1);
    }

    if ((read->seq = malloc(read->maxSequenceLen)) == NULL)
    {
        fprintf(stderr, "Unable to allocate sequence buffer\n");
        exit(1);
    }

    if (createTracks)
    {
		read->maxtracks = 10;
		read->ntracks = 0;

		read->maxName = 20;
		int maxFields = 20;

		read->trackName = (char**) malloc(read->maxtracks * sizeof(char*));
		read->trackfields = (int**) malloc(read->maxtracks * sizeof(int*));

		int i;
		for (i = 0; i < read->maxtracks; i++)
		{
			if ((read->trackName[i] = (char*) malloc(read->maxName)) == NULL)
			{
				fprintf(stderr, "Unable to allocate track name buffer\n");
				exit(1);
			}

			if ((read->trackfields[i] = (int*) malloc((maxFields + 2) * sizeof(int))) == NULL)
			{
				fprintf(stderr, "Unable to allocate track fields buffer\n");
				exit(1);
			}

			bzero(read->trackfields[i], (maxFields + 2) * sizeof(int));
			read->trackfields[i][0] = maxFields;
			read->trackfields[i][1] = 2;
		}
    }
    else
    {
		read->maxtracks = 0;
		read->ntracks = 0;
		read->maxName = 0;
    }
}

void resetPacbioRead(pacbio_read* pr)
{
    pr->ntracks = 0;
    int i;

    for (i = 0; i < pr->maxtracks; i++)
    {
        pr->trackfields[i][1] = 2;
    }

    pr->well = -1;
    pr->beg = -1;
    pr->end = -1;
    pr->len = 0;
}


static void parse_header( CreateContext* ctx, char* header, pacbio_read* pr )
{
    // reset pr
    resetPacbioRead( pr );

    char* c     = header;
    char* name  = NULL;
    char* value = NULL;

    pr->hasPacbioHeader = isPacBioHeader( c );

    if ( pr->hasPacbioHeader )
    {
        char* pch;

        pch   = strchr( c, '/' );
        int x = sscanf( pch + 1, "%d/%d_%d\n", &( pr->well ), &( pr->beg ), &( pr->end ) );

        if ( x != 3 )
            pr->hasPacbioHeader = 0;
    }
    else
    {
        pr->well = -1;
        pr->beg  = -1;
        pr->end  = -1;
    }

    if ( ctx->t_create_n )
    {
        while ( *c != '\0' && *c != '\n' && *c != ' ' )
            c++;

        if ( *c == ' ' || *c == '\n' )
            c++;

        while ( *c != '\0' )
        {
            *( c - 1 ) = '\0';
            name       = c;

            while ( *c != '\n' && isalnum( *c ) && *c != '=' )
            {
                c++;
            }

            if ( *c != '=' )
            {
                fprintf( stderr, "malformed track name: '%s'\n", c );
                exit( 1 );
            }

            int i;
            int skip = 1;
            for ( i = 0 ; i < ctx->t_create_n ; i++ )
            {
                printf("header %s <-> %.*s\n", ctx->t_create[i], (int)(c-name), name);

                if ( strlen(ctx->t_create[i]) == (size_t)(c - name) && strncmp(ctx->t_create[i], name, c - name) == 0 )
                {
                    skip = 0;
                    break;
                }
            }

            if (skip)
            {
                c++;

                while ( *c != '\0' && *c != '\n' && *c != ' ' )
                    c++;

                int cont = ( *c == ' ' );

                *c = '\0';

                if (cont)
                {
                    c += 1;
                }

                continue;
            }

            if ( pr->ntracks >= pr->maxtracks )
            {
                int newSize     = pr->ntracks * 1.2 + 10;
                pr->maxtracks   = newSize;
                pr->trackName   = (char**)realloc( pr->trackName, newSize * sizeof( char* ) );
                pr->trackfields = (int**)realloc( pr->trackfields, newSize * sizeof( int* ) );

                int i;
                int maxFields = 20;
                for ( i = pr->ntracks; i < newSize; i++ )
                {
                    if ( ( pr->trackName[ i ] = (char*)malloc( pr->maxName ) ) == NULL )
                    {
                        fprintf( stderr, "Unable to allocate track name buffer\n" );
                        exit( 1 );
                    }

                    if ( ( pr->trackfields[ i ] = (int*)malloc( ( maxFields + 2 ) * sizeof( int ) ) ) == NULL )
                    {
                        fprintf( stderr, "Unable to allocate track fields buffer\n" );
                        exit( 1 );
                    }

                    bzero( pr->trackfields[ i ], ( maxFields + 2 ) * sizeof( int ) );
                    pr->trackfields[ i ][ 0 ] = maxFields;
                }
            }

            int len = strlen( name );
            int cur = pr->ntracks;
            if ( len > pr->maxName )
            {
                pr->maxName = len + 10;
                int i;
                for ( i                = 0; i < cur; i++ )
                {
                    pr->trackName[ i ] = (char*)realloc( pr->trackName[ i ], pr->maxName );
                }
            }

            *c = '\0';

            int mval = 0;
            int cont = 0;
            char* endptr;
            long val;

            do
            {
                c++;
                // add assumption when RQ is parsed
                if ( strcmp( name, "RQ" ) == 0 && strncasecmp( c, "0.", 2 ) == 0 )
                    c = c + 2;

                value = c;

                while ( *c != '\0' && *c != '\n' && *c != ' ' && *c != ',' )
                    c++;

                mval = ( *c == ',' );
                cont = ( *c == ' ' );

                *c = '\0';

                if ( strcmp( name, TRACK_PACBIO_CHEM ) == 0 )
                {
                    int i         = 0;
                    int numOfChar = strlen( value );

                    int curf = pr->trackfields[ cur ][ 1 ];
                    if ( curf + numOfChar >= pr->trackfields[ cur ][ 0 ] )
                    {
                        int newFields               = curf * 1.2 + numOfChar;
                        pr->trackfields[ cur ]      = (int*)realloc( pr->trackfields[ cur ], ( newFields + 2 ) * sizeof( int ) );
                        pr->trackfields[ cur ][ 0 ] = newFields;
                    }

                    while ( i < numOfChar )
                    {
                        pr->trackfields[ cur ][ curf++ ] = (char)value[ i++ ];
                        pr->trackfields[ cur ][ 1 ] += 1;
                    }
                    break;
                }

                //            printf("'%s' -> '%s'\n", name, value);

                val = strtol( value, &endptr, 10 );

                if ( endptr != NULL && *endptr != '\0' )
                {
                    printf( "non-numeric value %s\n", value );
                    exit( 1 );
                }

                int curf = pr->trackfields[ cur ][ 1 ];
                if ( curf + 1 == pr->trackfields[ cur ][ 0 ] )
                {
                    uint64 newFields            = curf * 1.2 + 10;
                    pr->trackfields[ cur ]      = (int*)realloc( pr->trackfields[ cur ], ( newFields + 2 ) * sizeof( int ) );
                    pr->trackfields[ cur ][ 0 ] = newFields;
                }

                pr->trackfields[ cur ][ curf ] = val;
                pr->trackfields[ cur ][ 1 ] += 1;

            } while ( mval );

            //        printf("memcpy: cur: %d, name: %s, len: %d\n", cur,name, strlen(name)+1);
            memcpy( pr->trackName[ cur ], name, strlen( name ) + 1 );
            pr->ntracks++;

            if ( cont )
            {
                c++;
            }
        }
    }
}

void errorExit( CreateContext* ctx )
{

    //  Error exit:  Either truncate or remove the .idx and .bps files as appropriate.
    //               Remove the new image file <pwd>/<root>.dbx

    if ( ctx->ioff != 0 )
    {
        fseeko( ctx->indx, 0, SEEK_SET );
        if ( ftruncate( fileno( ctx->indx ), ctx->ioff ) < 0 )
            SYSTEM_ERROR
    }
    if ( ctx->boff != 0 )
    {
        fseeko( ctx->bases, 0, SEEK_SET );
        if ( ftruncate( fileno( ctx->bases ), ctx->boff ) < 0 )
            SYSTEM_ERROR
    }
    fclose( ctx->indx );
    fclose( ctx->bases );
    if ( ctx->ioff == 0 )
        unlink( Catenate( ctx->pwd, PATHSEP, ctx->root, ".idx" ) );
    if ( ctx->boff == 0 )
        unlink( Catenate( ctx->pwd, PATHSEP, ctx->root, ".bps" ) );

    if ( ctx->istub != NULL )
        fclose( ctx->istub );
    fclose( ctx->ostub );
    unlink( Catenate( ctx->pwd, "/", ctx->root, ".dbx" ) );

    exit( 1 );
}

static void findAndAddAvailableTracks( CreateContext* ctx )
{
    int rlen, dlen;
    char *root, *pwd, *name;
    int isdam;
    DIR* dirp;
    struct dirent* dp;

    root = ctx->root;
    pwd  = ctx->pwd;
    rlen = strlen( root );

    if ( root == NULL || pwd == NULL )
    {
        fprintf( stderr, "[ERROR] findAndAddAvailableTracks - database name not available\n" );
        exit( 1 );
    }

    if ( ( dirp = opendir( pwd ) ) == NULL )
    {
        fprintf( stderr, "[ERROR] findAndAddAvailableTracks - Cannot open directory %s\n", pwd );
    }

    isdam = 0;
    while ( ( dp = readdir( dirp ) ) != NULL ) //   Get case dependent root name (if necessary)
    {
        name = dp->d_name;
        if ( strcmp( name, Catenate( "", "", root, ".db" ) ) == 0 )
            break;
        if ( strcmp( name, Catenate( "", "", root, ".dam" ) ) == 0 )
        {
            isdam = 1;
            break;
        }
        if ( strcasecmp( name, Catenate( "", "", root, ".db" ) ) == 0 )
        {
            strncpy( root, name, rlen );
            break;
        }
        if ( strcasecmp( name, Catenate( "", "", root, ".dam" ) ) == 0 )
        {
            strncpy( root, name, rlen );
            isdam = 1;
            break;
        }
    }
    if ( dp == NULL )
    {
        fprintf( stderr, "findAndAddAvailableTracks - Cannot find %s (List_DB_Files)\n", pwd );
        closedir( dirp );
        exit( 1 );
    }

    if ( isdam )
        printf( "%s\n", Catenate( pwd, "/", root, ".dam" ) );
    else
        printf( "%s\n", Catenate( pwd, "/", root, ".db" ) );

    rewinddir( dirp ); //   Report each auxiliary file
    while ( ( dp = readdir( dirp ) ) != NULL )
    {
        name = dp->d_name;
        dlen = strlen( name );
#ifdef HIDE_FILES
        if ( name[ 0 ] != '.' )
            continue;
        dlen -= 1;
        name += 1;
#endif
        if ( dlen < rlen + 1 )
            continue;
        if ( name[ rlen ] != '.' )
            continue;
        if ( strncmp( name, root, rlen ) != 0 )
            continue;

        if ( strcasecmp( name + ( dlen - 4 ), "anno" ) == 0 )
        {
            name[ dlen - 5 ] = '\0';
            char* ptr        = strrchr( name, '.' );
            if ( ptr != NULL )
                find_track( ctx, ptr + 1 );
        }
    }
    closedir( dirp );
}

void initDB( int argc, char** argv, CreateContext* ctx )
{
    // parse program options
    parseOptions( argc, argv, ctx );

    {
        int i;

        ctx->root   = Root( argv[ ctx->lastParameterIdx ], ".db" );
        ctx->pwd    = PathTo( argv[ ctx->lastParameterIdx ] );
        ctx->dbname = Strdup( Catenate( ctx->pwd, "/", ctx->root, ".db" ), "Allocating db name" );
        if ( ctx->dbname == NULL )
            exit( 1 );

        if ( ctx->IFILE == NULL )
            ctx->ifiles = argc - ctx->lastParameterIdx - 1;
        else
        {
            File_Iterator* ng;

            ctx->ifiles = 0;
            ng          = init_file_iterator( argc, argv, ctx->IFILE, ctx->lastParameterIdx + 1 );
            while ( next_file( ng ) )
                ctx->ifiles += 1;
            free( ng );
        }

        ctx->istub = fopen( ctx->dbname, "r" );

        if ( ctx->istub == NULL )
        {
            ctx->ofiles = 0;

            ctx->bases = Fopen( Catenate( ctx->pwd, PATHSEP, ctx->root, ".bps" ), "w+" );
            ctx->indx  = Fopen( Catenate( ctx->pwd, PATHSEP, ctx->root, ".idx" ), "w+" );
            if ( ctx->bases == NULL || ctx->indx == NULL )
                exit( 1 );

            fwrite( ctx->db, sizeof( HITS_DB ), 1, ctx->indx );

            ctx->ureads        = 0;
            ctx->offset        = 0;
            ctx->boff          = 0;
            ctx->ioff          = 0;
            ctx->initialUreads = 0;
        }
        else
        {
            if ( fscanf( ctx->istub, DB_NFILE, &ctx->ofiles ) != 1 )
                SYSTEM_ERROR

            ctx->bases = Fopen( Catenate( ctx->pwd, PATHSEP, ctx->root, ".bps" ), "r+" );
            ctx->indx  = Fopen( Catenate( ctx->pwd, PATHSEP, ctx->root, ".idx" ), "r+" );
            if ( ctx->bases == NULL || ctx->indx == NULL )
                exit( 1 );

            if ( fread( ctx->db, sizeof( HITS_DB ), 1, ctx->indx ) != 1 )
                SYSTEM_ERROR
            fseeko( ctx->bases, 0, SEEK_END );
            fseeko( ctx->indx, 0, SEEK_END );

            ctx->initialUreads = ctx->db->ureads;
            ctx->ureads        = ctx->db->ureads;
            ctx->offset        = ftello( ctx->bases );
            ctx->boff          = ctx->offset;
            ctx->ioff          = ftello( ctx->indx );

            if ( ctx->t_create_n > 0 )
            {
                findAndAddAvailableTracks( ctx );
            }
        }

        ctx->flist = (char**)Malloc( sizeof( char* ) * ( ctx->ofiles + ctx->ifiles ), "Allocating file list" );
        ctx->ostub = Fopen( Catenate( ctx->pwd, "/", ctx->root, ".dbx" ), "w+" );
        if ( ctx->ostub == NULL || ctx->flist == NULL )
            exit( 1 );

        fprintf( ctx->ostub, DB_NFILE, ctx->ofiles + ctx->ifiles );
        for ( i = 0; i < ctx->ofiles; i++ )
        {
            int last;
            char prolog[ MAX_NAME ], fname[ MAX_NAME ];

            if ( fscanf( ctx->istub, DB_FDATA, &last, fname, prolog ) != 3 )
                errorExit( ctx );
            if ( ( ctx->flist[ i ] = Strdup( fname, "Adding to file list" ) ) == NULL )
                errorExit( ctx );
            fprintf( ctx->ostub, DB_FDATA, last, fname, prolog );
        }
    }

    ctx->pr1 = (pacbio_read*)malloc( sizeof( pacbio_read ) );
    ctx->pr2 = (pacbio_read*)malloc( sizeof( pacbio_read ) );

    initPacbioRead( ctx->pr1, ctx->t_create_n );
    initPacbioRead( ctx->pr2, ctx->t_create_n );

    ctx->rmax = MAX_NAME + 60000;
    ctx->read = (char*)malloc( ctx->rmax + 1 );
    if ( ctx->read == NULL )
    {
        fprintf( stderr, "[Error] Cannot allocate read buffer\n" );
        exit( 1 );
    }
}

static void addReadToDB( CreateContext* ctx, pacbio_read* prBest )
{
    int i, x;
    for ( i = 0; i < prBest->len; i++ )
    {
        x = number[ (int)prBest->seq[ i ] ];
        ctx->count[ x ] += 1;
        prBest->seq[ i ] = (char)x;
    }

    HITS_READ hr;
    hr.boff  = ctx->offset;
    hr.rlen  = prBest->len;
    hr.coff  = -1;
    hr.flags = DB_BEST;

    Compress_Read( prBest->len, prBest->seq );
    size_t clen = COMPRESSED_LEN( prBest->len );

    if ( fwrite( prBest->seq, 1, clen, ctx->bases ) != clen )
    {
        fprintf( stderr, "[ERROR] - Unable to write compressed sequence (%s, %d) to database\n", prBest->prolog, prBest->seqIDinFasta );
        exit( 1 );
    }
    if ( fwrite( &hr, sizeof( HITS_READ ), 1, ctx->indx ) != 1 )
    {
        fprintf( stderr, "[ERROR] - Unable to write HITS_READ (%s, %d) to database\n", prBest->prolog, prBest->seqIDinFasta );
        exit( 1 );
    }

    add_to_track( ctx, find_track( ctx, TRACK_SEQID ), ctx->ureads, prBest->seqIDinFasta );
    if ( prBest->hasPacbioHeader )
    {
        add_to_track( ctx, find_track( ctx, TRACK_PACBIO_HEADER ), ctx->ureads, prBest->well );
        add_to_track( ctx, find_track( ctx, TRACK_PACBIO_HEADER ), ctx->ureads, prBest->beg );
        add_to_track( ctx, find_track( ctx, TRACK_PACBIO_HEADER ), ctx->ureads, prBest->end );
    }

    for ( i = 0; i < prBest->ntracks; i++ )
    {
        int j;
        for ( j = 2; j < prBest->trackfields[ i ][ 1 ]; j++ )
            add_to_track( ctx, find_track( ctx, prBest->trackName[ i ] ), ctx->ureads, prBest->trackfields[ i ][ j ] );
    }

    ctx->offset += clen;
    ctx->ureads += 1;
    ctx->totlen += prBest->len;
    if ( prBest->len > ctx->maxlen )
        ctx->maxlen = prBest->len;
}

static void readFastaFile( CreateContext* ctx, char* name )
{
    FILE* input;
    char *path, *core, *prolog;
    int nline, eof, rlen;

    //  Open it: <path>/<core>.fasta, check that core is not too long,
    //           and checking that it is not already in flist.

    path = PathTo( name );
    core = Root( name, ".fasta" );
    if ( ( input = fopen( Catenate( path, "/", core, ".fasta" ), "r" ) ) == NULL )
    {
        core = Root( name, ".fa" );
        if ( ( input = fopen( Catenate( path, "/", core, ".fa" ), "r" ) ) == NULL )
            errorExit( ctx );
    }
    free( path );

    if ( strlen( core ) >= MAX_NAME )
    {
        fprintf( stderr, "File name over %d chars: '%.200s'\n",
                 MAX_NAME, core );
        errorExit( ctx );
    }

    {
        int j;

        for ( j = 0; j < ctx->ofiles; j++ )
            if ( strcmp( core, ctx->flist[ j ] ) == 0 )
            {
                fprintf( stderr, "File %s.fasta is already in database %s.db\n", core, Root( ctx->dbname, ".db" ) );
                errorExit( ctx );
            }
    }

    //  Get the header of the first line.  If the file is empty skip.

    rlen  = 0;
    nline = 1;
    eof   = ( fgets( ctx->read, MAX_NAME, input ) == NULL );
    if ( eof || strlen( ctx->read ) < 1 )
    {
        fprintf( stderr, "Skipping '%s', file is empty!\n", core );
        fclose( input );
        free( core );
        return;
    }

    //   Add the file name to flist

    if ( ctx->VERBOSE )
    {
        fprintf( stderr, "Adding '%s' ...\n", core );
        fflush( stderr );
    }
    ctx->flist[ ctx->ofiles++ ] = core;

    // Check that the first line has PACBIO format, and record prolog in 'prolog'.

    if ( ctx->read[ strlen( ctx->read ) - 1 ] != '\n' )
    {
        fprintf( stderr, "File %s.fasta, Line 1: Fasta line is too long (> %d chars)\n", core, MAX_NAME - 2 );
        errorExit( ctx );
    }

    if ( !eof && ctx->read[ 0 ] != '>' )
    {
        fprintf( stderr, "File %s.fasta, Line 1: First header in fasta file is missing\n", core );
        errorExit( ctx );
    }

    if ( isPacBioHeader( ctx->read + 1 ) )
    {
        char* find;
        find   = index( ctx->read + 1, '/' );
        *find  = '\0';
        prolog = Strdup( ctx->read + 1, "Extracting prolog" );
        *find  = '/';
    }
    else
    {
        prolog = Strdup( "DAZZ_READ", "Extracting prolog" );
    }

    //  Read in all the sequences until end-of-file

    pacbio_read *prBest, *prNext;
    prNext = ctx->pr1;
    prBest = ( ctx->BEST ) ? NULL : ctx->pr1;

    {
        ctx->pr1->seqIDinFasta = -1;
        ctx->pr2->seqIDinFasta = -1;
        while ( !eof )
        {
            // parse header
            parse_header( ctx, ctx->read + ( rlen + 1 ), prNext );
            prNext->seqIDinFasta += 1;

            rlen = 0;
            while ( 1 )
            {
                char* line = NULL;
                size_t linelen;
                line = fgetln(input, &linelen);

                if (!line)
                {
                    eof = 1;
                    break;
                }

                if ( rlen + linelen > (size_t)ctx->rmax )
                {
                    ctx->rmax = ( (int)( 1.2 * ctx->rmax ) ) + 1000 + linelen;
                    ctx->read = (char*)realloc( ctx->read, ctx->rmax + 1 );
                    if ( ctx->read == NULL )
                    {
                        fprintf( stderr, "File %s.fasta, Line %d:", core, nline );
                        fprintf( stderr, "Out of memory (Allocating line buffer)\n" );
                        errorExit( ctx );
                    }
                }

                if ( line[0] == '>' )
                {
                    memcpy(ctx->read + rlen, line, linelen + 1);
                }
                else if ( line[linelen - 1] == '\n' )
                {
                    line[linelen - 1] = '\0';
                    linelen -= 1;

                    memcpy(ctx->read + rlen, line, linelen);
                }

                nline += 1;

                if (line[0] == '>')
                {
                    break;
                }

                rlen += linelen;
            }

            if ( rlen < ctx->opt_min_length )
            {
                if ( ctx->VERBOSE > 1 )
                {
                    fprintf( stderr, "Warning: skipping read of length %d\n", rlen );
                }
                continue;
            }

            if ( ctx->t_create_n && ctx->useFullHqReadsOnly )
            {
                int i;

                for ( i = 0; i < prNext->ntracks; i++ )
                {
                    if ( strcmp( prNext->trackName[ i ], "readType" ) == 0 )
                    {
                        break;
                    }
                }

                // ignore reads that do not have the readType attribute
                if ( i == prNext->ntracks )
                    continue;

                // ignore reads that have not the proper format of readType argument
                if ( prNext->trackfields[ i ][ 1 ] != 3 )
                    continue;

                // TODO use enum from dextractUtils
                // typedef enum { type_Empty = 0, type_FullHqRead0 = 1, type_FullHqRead1 = 2, type_PartialHqRead0 = 3, type_PartialHqRead1 = 4, type_PartialHqRead2 = 5, type_Multiload = 6, type_Indeterminate = 7, type_NotDefined = 255} readType ;
                if ( prNext->trackfields[ i ][ 2 ] != 1 && prNext->trackfields[ i ][ 2 ] != 2 )
                    continue;
            }

            prNext->len       = rlen;
            ctx->read[ rlen ] = '\0';

            if ( rlen >= prNext->maxSequenceLen )
            {
                prNext->maxSequenceLen = ( (int)( 1.2 * rlen ) ) + 1000 + MAX_NAME;
                prNext->seq            = (char*)realloc( prNext->seq, prNext->maxSequenceLen + 1 );
                if ( prNext->seq == NULL )
                {
                    fprintf( stderr, "File %s.fasta, Line %d:", core, nline );
                    fprintf( stderr, " Out of memory (Allocating line buffer)\n" );
                    errorExit( ctx );
                }
            }

            memcpy( prNext->seq, ctx->read, rlen );

            if ( ctx->BEST )
            {
                if ( prBest == NULL )
                {
                    if ( prNext == ctx->pr1 )
                    {
                        prBest = ctx->pr1;
                        prNext = ctx->pr2;
                    }
                    else
                    {
                        prBest = ctx->pr2;
                        prNext = ctx->pr1;
                    }
                    continue;
                }
                else if ( prBest->well == prNext->well )
                {
                    if ( prBest->len < prNext->len )
                    {
                        prBest = prNext;
                        prNext = ( prBest == ctx->pr1 ) ? ctx->pr2 : ctx->pr1;
                    }
                    continue;
                }
            }

            addReadToDB( ctx, prBest );

            if ( ctx->BEST )
            {
                prBest = prNext;
                prNext = ( prBest == ctx->pr1 ) ? ctx->pr2 : ctx->pr1;
            }
        }

        //  Complete processing of .fasta file: flush last well group, write file line
        //      in db image, free prolog, and close file

        if ( ctx->BEST )
        {
            addReadToDB( ctx, prBest );
        }

        fprintf( ctx->ostub, DB_FDATA, ctx->ureads, core, prolog );
    }

    free( prolog );
    fclose( input );
}

static void updateBlockDBOffsets( CreateContext* ctx )
{
    //  If db has been previously partitioned then calculate additional partition points and
    //    write to new db file image

    int nblock;
    if ( ctx->istub && fscanf( ctx->istub, DB_NBLOCK, &nblock ) == 1 )
    {
        int64 totlen, dbpos, size;
        int ireads, rlen;
        int ufirst;
        HITS_READ record;
        int i;

        if ( ctx->VERBOSE )
        {
            fprintf( stderr, "Updating block partition ...\n" );
            fflush( stderr );
        }

        //  Read the block portion of the existing db image getting the indices of the first
        //    read in the last block of the exisiting db as well as the partition parameters.
        //    Copy the old image block information to the new block information (except for
        //    the indices of the last partial block)

        dbpos = ftello( ctx->ostub );
        fprintf( ctx->ostub, DB_NBLOCK, 0 );
        if ( fscanf( ctx->istub, DB_PARAMS, &size ) != 1 )
            SYSTEM_ERROR
        fprintf( ctx->ostub, DB_PARAMS, size );

        size *= 1000000ll;

        if ( !ctx->appendReadsToNewBlock )
            nblock -= 1;
        for ( i = 0; i <= nblock; i++ )
        {
            if ( fscanf( ctx->istub, DB_BDATA, &ufirst ) != 1 )
                SYSTEM_ERROR
            fprintf( ctx->ostub, DB_BDATA, ufirst );
        }

        //  Seek the first record of the last block of the existing db in .idx, and then
        //    compute and record partition indices for the rest of the db from this point
        //    forward.

        fseeko( ctx->indx, sizeof( HITS_DB ) + sizeof( HITS_READ ) * ufirst, SEEK_SET );
        totlen = 0;
        ireads = 0;
        for ( i = ufirst; i < ctx->ureads; i++ )
        {
            if ( fread( &record, sizeof( HITS_READ ), 1, ctx->indx ) != 1 )
                SYSTEM_ERROR
            rlen = record.rlen;
            ireads += 1;
            totlen += rlen;
            if ( totlen >= size )
            {
                fprintf( ctx->ostub, " %9d\n", i + 1 );
                totlen = 0;
                ireads = 0;
                nblock += 1;
            }
        }

        if ( ireads > 0 )
        {
            fprintf( ctx->ostub, DB_BDATA, ctx->ureads );
            nblock += 1;
        }

        fseeko( ctx->ostub, dbpos, SEEK_SET );
        fprintf( ctx->ostub, DB_NBLOCK, nblock ); //  Rewind and record the new number of blocks
    }
}

static void usage(const char* progname)
{
    fprintf( stderr, "usage: %s [-vabQ] [-c <track>] [-x <int>] <path:db> (-f file  | <path:string> <input:fasta> ...)\n", progname );
    fprintf( stderr, "options: -v ... verbose\n" );
    fprintf( stderr, "         -Q ... only use pacbio reads with readtype FullHqRead\n" );
    fprintf( stderr, "         -a ... append new reads to new block\n" );
    fprintf( stderr, "         -b ... only the longest/best read from a pacbio well is incorporated into DB\n" );
    fprintf( stderr, "         -x ... min read length (%d)\n", DEF_OPT_X );
    fprintf( stderr, "         -c ... convert fasta header arguments (NAME=x,y) into database tracks\n" );
}

static void parseOptions( int argc, char* argv[], CreateContext* ctx )
{

    // set default values
    ctx->VERBOSE               = 0;
    ctx->useFullHqReadsOnly    = 0;
    // ctx->createTracks          = 0;
    ctx->BEST                  = 0;
    ctx->appendReadsToNewBlock = 0;
    ctx->opt_min_length        = DEF_OPT_X;

    // parse arguments

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "vc:abQx:f:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'v':
                ctx->VERBOSE += 1;
                break;

            case 'Q':
                ctx->useFullHqReadsOnly = 1;
                break;

            case 'c':
                // ctx->createTracks = 1;

                if ( ctx->t_create_n + 1 >= ctx->t_create_max )
                {
                    ctx->t_create_max += 10;
                    ctx->t_create = realloc( ctx->t_create, sizeof(char*) * ctx->t_create_max );
                }

                ctx->t_create[ ctx->t_create_n ] = optarg;
                ctx->t_create_n += 1;

                break;

            case 'x':
                ctx->opt_min_length = atoi( optarg );
                break;

            case 'b':
                ctx->BEST = 1;
                break;

            case 'a':
                ctx->appendReadsToNewBlock = 1;
                break;
            case 'f':
            {
                ctx->IFILE = fopen( optarg, "r" );
                if ( ctx->IFILE == NULL )
                {
                    fprintf( stderr, "Cannot open file of inputs '%s'\n", optarg );
                    exit( 1 );
                }
                break;
            }

            default:
                usage( argv[ 0 ] );
                exit( 1 );
        }
    }

    if ( ctx->opt_min_length < 0 )
    {
        fprintf( stderr, "invalid min read length of %d\n", ctx->opt_min_length );
        exit( 1 );
    }

    if ( ( ctx->IFILE == NULL ) && argc - optind < 2 )
    {
        usage( argv[ 0 ] );
        exit( 1 );
    }

    ctx->lastParameterIdx = optind;
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    CreateContext ctx;
    bzero( &ctx, sizeof( CreateContext ) );

    ctx.db = &db;

    // parse options and init db

    initDB( argc, argv, &ctx );

    int c;
    File_Iterator* ng;

    ctx.totlen = 0;                      //  total # of bases in new .fasta files
    ctx.maxlen = 0;                      //  longest read in new .fasta files
    for ( c            = 0; c < 4; c++ ) //  count of acgt in new .fasta files
        ctx.count[ c ] = 0;

    //  For each new .fasta file do:

    ng = init_file_iterator( argc, argv, ctx.IFILE, ctx.lastParameterIdx + 1 );
    while ( next_file( ng ) )
    {
        if ( ng->name == NULL )
            errorExit( &ctx );

        readFastaFile( &ctx, ng->name );
    }

    //  Finished loading all sequences: update relevant fields in db record

    db.ureads = ctx.ureads;

    if ( ctx.istub == NULL )
    {
        for ( c = 0; c < 4; c++ )
        {
            db.freq[ c ] = (float)( ( 1. * ctx.count[ c ] ) / ctx.totlen );
        }
        db.totlen = ctx.totlen;
        db.maxlen = ctx.maxlen;
    }
    else
    {
        for ( c          = 0; c < 4; c++ )
            db.freq[ c ] = (float)( ( db.freq[ c ] * db.totlen + ( 1. * ctx.count[ c ] ) ) / ( db.totlen + ctx.totlen ) );
        db.totlen += ctx.totlen;
        if ( ctx.maxlen > db.maxlen )
            db.maxlen = ctx.maxlen;
    }

    updateBlockDBOffsets( &ctx );

    rewind( ctx.indx );
    fwrite( &db, sizeof( HITS_DB ), 1, ctx.indx ); //  Write the finalized db record into .idx

    rewind( ctx.ostub ); //  Rewrite the number of files actually added
    fprintf( ctx.ostub, DB_NFILE, ctx.ofiles );

    if ( ctx.istub != NULL )
        fclose( ctx.istub );
    fclose( ctx.ostub );
    fclose( ctx.indx );
    fclose( ctx.bases );

    rename( Catenate( ctx.pwd, "/", ctx.root, ".dbx" ), ctx.dbname ); //  New image replaces old image

    write_tracks( &ctx, ctx.dbname );

    // clean up
    free_tracks( &ctx );

    free( ng );
    free( ctx.root );
    free( ctx.pwd );
    free( ctx.dbname );

    return 0;
}
