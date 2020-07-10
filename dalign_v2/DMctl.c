#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../lib/dmask.h"
#include "../lib/dmask_proto.h"

#define DEF_ARG_H "localhost"        // host
#define DEF_ARG_P DMASK_DEFAULT_PORT // listen port

static void usage( FILE* fout, const char* app )
{

    fprintf( fout, "usage:  %s [-h host] [-p port] <command>\n\n", app );

    fprintf( fout, "Interact with the dynamic masking server.\n\n" );

    fprintf( fout, "options: -h  masking server host (%s)\n", DEF_ARG_H );
    fprintf( fout, "         -p  masking server port (%d)\n\n", DEF_ARG_P );

    fprintf( fout, "commands: shutdown        initiate shutdown of the dynamic masking server\n" );
    fprintf( fout, "          lock            disable coverage statistics updates\n" );
    fprintf( fout, "          unlock          enable coverage statistics updates\n" );
    fprintf( fout, "          track           write track(s)\n" );
    fprintf( fout, "          done [dir|input.las ...]  send done signal for all overlap files in dir or a list of las file\n" );
}

int main( int argc, char* argv[] )
{
    int port   = DEF_ARG_P;
    char* host = DEF_ARG_H;
    char* app  = argv[ 0 ];

    // process arguments

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "h:p:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'p':
                port = atoi( optarg );
                break;

            case 'h':
                host = optarg;
                break;

            default:
                usage(stdout, app);
                exit( 1 );
        }
    }

    if ( port == 0 )
    {
        fprintf( stderr, "invalid listen port %d\n", port );
        exit( 1 );
    }

    if ( optind + 1 > argc )
    {
        usage(stdout, app);
        exit( 1 );
    }

    char* command   = argv[ optind++ ];
    DynamicMask* dm = dm_init( host, port );

    if ( dm == NULL )
    {
        fprintf( stderr, "failed to initialise dynamic dust\n" );
        exit( 1 );
    }

    if ( strcasecmp( command, "shutdown" ) == 0 )
    {
        dm_shutdown( dm );
    }
    else if ( strcasecmp( command, "lock" ) == 0 )
    {
        dm_lock( dm );
    }
    else if ( strcasecmp( command, "unlock" ) == 0 )
    {
        dm_unlock( dm );
    }
    else if ( strcasecmp( command, "intervals" ) == 0 )
    {
        dm_intervals( dm );
    }
    else if ( strcasecmp( command, "track" ) == 0 )
    {
        dm_write_track( dm );
    }
    else if ( strcasecmp( command, "done" ) == 0 )
    {
        char** files;
        int maxf = 100;
        int curf = 0;
        int i, len, dirLen;

        files = (char**)malloc( sizeof( char* ) * maxf );

        while ( argc - optind > 0 )
        {
            struct stat sb;
            stat( argv[ optind ], &sb );

            if ( S_ISDIR( sb.st_mode ) )
            {
                DIR* dp;
                struct dirent* ep;
                dp = opendir( argv[ optind ] );

                if ( dp != NULL )
                {
                    dirLen = strlen( argv[ optind ] );
                    while ( ( ep = readdir( dp ) ) )
                    {
                        if ( curf == maxf )
                        {
                            maxf  = maxf * 1.2 + 10;
                            files = (char**)realloc( files, sizeof( char* ) * maxf );
                        }
                        len = strlen( ep->d_name );
                        // check for las file name extension
                        if ( ( len < 4 ) || ( ( strcasecmp( ep->d_name + ( len - 4 ), ".las" ) != 0 ) ) )
                            continue;

                        files[ curf ] = (char*)malloc( dirLen + len + 20 );
                        sprintf( files[ curf ], "%s/%s", argv[ optind ], ep->d_name );
                        curf++;
                    }
                    (void)closedir( dp );
                }
                else
                    perror( "Couldn't open the directory" );
            }
            else if ( S_ISREG( sb.st_mode ) )
            {
                if ( curf == maxf )
                {
                    maxf  = maxf * 1.2 + 10;
                    files = (char**)realloc( files, sizeof( char* ) * maxf );
                }
                len = strlen( argv[ optind ] );
                if ( ( len > 4 ) && ( ( strcasecmp( argv[ optind ] + ( len - 4 ), ".las" ) == 0 ) ) )
                {
                    files[ curf ] = (char*)malloc( len + 10 );
                    memcpy( files[ curf ], argv[ optind ], len );
                    files[ curf ][ len ] = '\0';
                    curf++;
                }
            }
            else
            {
                fprintf( stderr, "warning: file %s is not accepted!\n", argv[ optind ] );
            }
            optind++;
        }

        files[ curf ] = NULL;

        //        i = 0;
        //        while (files[i] != NULL)
        //          {
        //            printf("file[%d]: %s\n", i, files[i]);
        //            i++;
        //          }
        //        exit(1);

        if ( !dm_done( dm, files ) )
            fprintf( stderr, "server did not get results\n" );

        for ( i = 0; i < curf; i++ )
            free( files[ i ] );
        free( files );
    }
    else
    {
        usage(stdout, app);
    }

    dm_free( dm );

    return 0;
}
