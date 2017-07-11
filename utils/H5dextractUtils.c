#include "H5dextractUtils.h"

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void printUsage( char* prog, FILE* out )
{
    fprintf( out, "\nUsage: [options] -F file | <input:bax_h5> ... \n\n" );
    fprintf( out, "Extract sequence and quality of the reads contained in a (set of) bax.h5 files\n\n" );

    fprintf( out, " -h, --help\n" );
    fprintf( out, " -c, --cum                report cumulative statistics for all input files\n\n" );

    fprintf( out, " -s file, --stat file   output statistics to file (defaults to stdout)\n" );
    fprintf( out, " -f file, --fasta file  output fasta file name\n" );
    fprintf( out, " -q file, --fastq file  output fastq file name\n" );
    fprintf( out, " -i file, --quiva file  output quiva file name\n\n" );

    fprintf( out, " -X n, --Len n   maximum read length\n" );
    fprintf( out, " -x n, --len n   minimum read length\n" );
    fprintf( out, " -Q n, --qual n  minimum read quality (based on the RQ field)\n" );

    fprintf( out, " -S mode, --subread mode   which subreads from the ZMWs should be extracted\n"
                  "    a ... all subreads (default)\n"
                  "    l ... longest subread\n"
                  "    s ... shortest subread\n"
                  "    b ... best subread (based on quality and length)\n\n" );

    fprintf( out, " -m n, --movMin n       ignore bases captured prior the amount of seconds specified\n" );
    fprintf( out, " -M n, --movMax n       ignore bases captured after the amount of seconds specified\n" );

    fprintf( out, " -t n, --timeBinSize n  sequencing quality histogram bin size, in seconds. (default 600 = 10 min)\n" );
    fprintf( out, " -l n, --lenBinSize n 	 subread length histogram bin size (default 1000)\n" );
    fprintf( out, " -w file                restrict extraction to a specific set of wells. one line per input bax.h5 file. Format: n1 n2 n3 n4-n5.\n\n" );
    fprintf( out, " -F file                file with list of input bax.h5 files (one file per line) \n" );
    fprintf( out, " -z n                   extract subreads only from ZMW's with a specific number of subreads (default: -1)\n" );
}

BAX_OPT* parseBaxOptions( int argc, char** argv )
{
    BAX_OPT* bopt = (BAX_OPT*)malloc( sizeof( BAX_OPT ) );
    initBaxOptions( bopt );

    int c;
    while ( 1 )
    {
        static struct option long_options[] =
            {
                {"help", no_argument, 0, 'h'},
                {"cum", no_argument, 0, 'c'},
                {"verbose", no_argument, 0, 'v'},
                {"fasta", required_argument, 0, 'f'},
                {"fastq", required_argument, 0, 'q'},
                {"quiva", required_argument, 0, 'i'},
                {"stat", required_argument, 0, 's'},
                {"len", required_argument, 0, 'x'},
                {"Len", required_argument, 0, 'X'},
                {"qual", required_argument, 0, 'Q'},
                {"movMin", required_argument, 0, 'm'},
                {"movMax", required_argument, 0, 'M'},
                {"timeBinSize", required_argument, 0, 't'},
                {"lenBinSize", required_argument, 0, 'l'},
                {"subread", required_argument, 0, 'S'},
                {"wellNumbers", required_argument, 0, 'w'},
                {"zmw", required_argument, 0, 'z'},
                {"file", required_argument, 0, 'F'}};
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long( argc, argv, "vchf:q:i:s:x:X:Q:m:M:t:l:S:w:F:z:", long_options, &option_index );

        /* Detect the end of the options. */
        if ( c == -1 )
            break;

        switch ( c )
        {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if ( long_options[ option_index ].flag != 0 )
                    break;
                break;

            case 'h':
                printUsage( argv[ 0 ], stderr );
                exit( 1 );
            case 'c':
                bopt->CUMULATIVE = 1;
                break;
            case 'v':
                bopt->VERBOSE++;
                break;
            case 'f':
                bopt->fastaOut = optarg;
                break;
            case 'q':
                bopt->fastqOut = optarg;
                break;
            case 'i':
                bopt->quivaOut = optarg;
                break;
            case 's':
                bopt->statOut = optarg;
                break;
            case 'F':
                bopt->baxInFileName = optarg;
                break;
            case 'w':
                bopt->wellNumbersInFileName = optarg;
                break;
            case 'x':
                bopt->MIN_LEN = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument of minimum read length (-x ARG)! Must be a positive number.\n" );
                    exit( 1 );
                }
                break;
            case 'X':
                bopt->MAX_LEN = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument of maximum read length (-X ARG)! Must be a positive number\n" );
                    exit( 1 );
                }
                break;
            case 'Q':
                bopt->MIN_QV = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument of minimum read quality (-q ARG)! Must be an integer in [0, 1000]\n" );
                    exit( 1 );
                }
                break;
            case 'm':
                bopt->MIN_MOVIE_TIME = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument for movie start time (-m ARG)! Must be an integer in [0, 36000]\n" );
                    exit( 1 );
                }
                break;
            case 'M':
                bopt->MAX_MOVIE_TIME = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument for movie end time (-M ARG)! Must be an integer in [0, 36000]\n" );
                    exit( 1 );
                }
                break;
            case 'z':
                bopt->zmw_minNrOfSubReads = (int)strtol( optarg, NULL, 10 );
                if ( errno || bopt->zmw_minNrOfSubReads < 1 )
                {
                    fprintf( stderr, "Cannot parse argument for zmw (-z ARG)! Must be a positive number\n" );
                    exit( 1 );
                }
                break;
            case 't':
                bopt->TIME_BIN_SIZE = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument of time histogram bin size (-t ARG)! Must an integer in [300, 3600]\n" );
                    exit( 1 );
                }
                break;
            case 'l':
                bopt->READLEN_BIN_SIZE = (int)strtol( optarg, NULL, 10 );
                if ( errno )
                {
                    fprintf( stderr, "Cannot parse argument of length histogram bin size (-l ARG)! Must be an integer in [1000, 10000]\n" );
                    exit( 1 );
                }
                break;
            case 'S':
                switch ( *optarg )
                {
                    case 'a':
                        bopt->subreadSel = all;
                        break;
                    case 'b':
                        bopt->subreadSel = best;
                        break;
                    case 'l':
                        bopt->subreadSel = longest;
                        break;
                    case 's':
                        bopt->subreadSel = shortest;
                        break;
                    default:
                        fprintf( stderr, "Cannot parse argument subread selection (-S ARG)! Possible values: [ a (all), b (best), l (longest), s (shortest) ]\n" );
                        exit( 1 );
                }
                break;
            case '?':
                printUsage( argv[ 0 ], stderr );
                exit( 1 );

            default:
                printUsage( argv[ 0 ], stderr );
                exit( 1 );
        }
    }

    if ( bopt->statOut != NULL )
    {
        if ( ( bopt->statFile = fopen( bopt->statOut, "w" ) ) == NULL )
        {
            fprintf( stderr, "Cannot open statistic file: %s!\n", bopt->statOut );
            exit( 1 );
        }
    }
    else
        bopt->statFile = stdout;

    if ( bopt->fastaOut != NULL )
    {
        if ( ( bopt->fastaFile = fopen( bopt->fastaOut, "w" ) ) == NULL )
        {
            fprintf( stderr, "Cannot open fasta file: %s!\n", bopt->fastaOut );
            exit( 1 );
        }
    }

    if ( bopt->fastqOut != NULL )
    {
        if ( ( bopt->fastqFile = fopen( bopt->fastqOut, "w" ) ) == NULL )
        {
            fprintf( stderr, "Cannot open fastq file: %s!\n", bopt->fastqOut );
            exit( 1 );
        }
    }

    if ( bopt->quivaOut != NULL )
    {
        if ( ( bopt->quivaFile = fopen( bopt->quivaOut, "w" ) ) == NULL )
        {
            fprintf( stderr, "Cannot open fasta file: %s!\n", bopt->quivaOut );
            exit( 1 );
        }
    }

    //parse file with list of bax.h5 files if present
    if ( bopt->baxInFileName != NULL )
    {
        readInputBaxFromFile( bopt );
        if ( optind < argc )
        {
            fprintf( stderr, "[WARNING] : if -F option is set, then further bax.h5 files given in command line are ignored!!\n" );
        }
    }

    else
    {
        if ( optind == argc )
        {
            fprintf( stderr, "At least one bax.h5 file is needed!\n" );
            printUsage( argv[ 0 ], stderr );
            exit( 1 );
        }

        while ( optind < argc )
        {
            if ( bopt->nBax == bopt->nMaxBax )
            {
                bopt->nMaxBax = ( (int)( 1.2 * bopt->nMaxBax ) ) + 10;
                bopt->baxIn   = (char**)realloc( bopt->baxIn, sizeof( char* ) * bopt->nMaxBax );
            }

            FILE* in;
            if ( ( in = fopen( argv[ optind ], "r" ) ) == NULL )
            {
                fprintf( stderr, "%s: Cannot find file %s !\n", argv[ 0 ], argv[ optind ] );
                optind++;
                continue;
            }
            else
                fclose( in );
            bopt->baxIn[ bopt->nBax ] = (char*)malloc( strlen( argv[ optind ] ) + 10 );
            strcpy( bopt->baxIn[ bopt->nBax ], argv[ optind ] );
            optind++;
            bopt->nBax++;
        }
    }

    if ( bopt->wellNumbersInFileName != NULL )
        readWellNumbersFromFile( bopt );

    return bopt;
}

char* trimwhitespace( char* str )
{
    char* end;

    // Trim leading space
    while ( isspace( *str ) )
        str++;

    if ( *str == 0 ) // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen( str ) - 1;
    while ( end > str && isspace( *end ) )
        end--;

    // Write new null terminator
    *( end + 1 ) = 0;

    return str;
}

int parse_ranges( char* line, int* _reps, int** _pts )
{
    int num = 10;
    int b, e;

    int* pts = (int*)malloc( sizeof( int ) * 2 * ( 2 + num ) );
    int reps = 0;

    char* s    = line;
    char* eptr = line;
    while ( eptr[ 0 ] != '\0' )
    {
        if ( reps + 4 >= num )
        {
            num = ( reps + 4 ) * 1.2 + 10;
            pts = (int*)realloc( pts, sizeof( int ) * num );
        }
        b = strtol( s, &eptr, 10 );

        if ( eptr[ 0 ] == '\0' || eptr[ 0 ] == ' ' )
        {
            pts[ reps++ ] = b;
            pts[ reps++ ] = b;
            e             = b;

            if ( eptr[ 0 ] == ' ' )
                s = eptr++;
        }

        if ( eptr[ 0 ] == '-' )
        {
            s             = eptr;
            e             = strtol( s + 1, &eptr, 10 );
            pts[ reps++ ] = b;
            pts[ reps++ ] = e;

            s = eptr;
        }
    }

    if ( reps > 0 )
    {
        qsort( pts, reps / 2, sizeof( int64 ), cmp_range );

        int c;
        b = 0;

        for ( c = 0; c < reps; c += 2 )
        {
            if ( b > 0 && pts[ b - 1 ] >= pts[ c ] - 1 )
            {
                if ( pts[ c + 1 ] > pts[ b - 1 ] )
                {
                    pts[ b - 1 ] = pts[ c + 1 ];
                }
            }
            else
            {
                pts[ b++ ] = pts[ c ];
                pts[ b++ ] = pts[ c + 1 ];
            }
        }

        pts[ b++ ] = INT32_MAX;
        reps       = b;
    }
    else
    {
        pts[ reps++ ] = 1;
        pts[ reps++ ] = INT32_MAX;
    }

    *_reps = reps;
    *_pts  = pts;

    return 1;
}

int cmp_range( const void* l, const void* r )
{
    int x = *( (int32*)l );
    int y = *( (int32*)r );
    return ( x - y );
}

void readWellNumbersFromFile( BAX_OPT* bopt )
{
    if ( bopt->wellNumbersInFileName == NULL )
    {
        fprintf( stderr, "[WARNING] - readWellNumbersFromFile: Input file name not available!\n" );
        return;
    }

    FILE* in;
    if ( ( in = fopen( bopt->wellNumbersInFileName, "r" ) ) == NULL )
    {
        fprintf( stderr, "[WARNING] - readWellNumbersFromFile: Cannot open file \"%s\"!\n", bopt->wellNumbersInFileName );
        return;
    }

    // read file line by line
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    int lineNumber = 0;

    while ( ( read = getline( &line, &len, in ) ) != -1 )
    {
        if ( bopt->VERBOSE > 1 )
        {
            printf( "Retrieved line of length %zu :\n", read );
            printf( "%s", line );
        }
        // get rid of newline
        if ( line[ read - 1 ] == '\n' )
            line[ read - 1 ] = '\0';

        // trim leading and trailing white spaces
        line = trimwhitespace( line );

        if ( lineNumber >= bopt->nBax )
        {
            fprintf( stderr, "Number of wellNumber lines greater than number of bax.h5 files!\n" );
            exit( 1 );
        }

        // parse lines
        int reps = 0;
        int* pts = NULL;

        parse_ranges( line, &reps, &pts );
        if ( bopt->VERBOSE > 2 )
        {
            int i;
            printf( "reps: %d\n", reps );
            for ( i = 1; i < reps; i += 2 )
                printf( "%d, %d; \n", pts[ i - 1 ], pts[ i ] );
        }
        bopt->numWellNumbers[ lineNumber ] = reps;
        bopt->wellNumbers[ lineNumber ]    = pts;
        lineNumber++;
    }
    if ( line )
        free( line );
    fclose( in );
}

void readInputBaxFromFile( BAX_OPT* bopt )
{
    if ( bopt->baxInFileName == NULL )
    {
        fprintf( stderr, "[WARNING] - readInputBaxFromFile: Input file name not available!\n" );
        return;
    }

    FILE *in, *baxTest;
    if ( ( in = fopen( bopt->baxInFileName, "r" ) ) == NULL )
    {
        fprintf( stderr, "[WARNING] - readInputBaxFromFile: Cannot open file \"%s\"!\n", bopt->baxInFileName );
        return;
    }

    // read file line by line
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    int lineNumber = 0;

    while ( ( read = getline( &line, &len, in ) ) != -1 )
    {
        lineNumber++;
        if ( bopt->VERBOSE > 1 )
        {
            printf( "Retrieved line of length %zu :\n", read );
            printf( "%s", line );
        }
        // trim leading and trailing white spaces
        trimwhitespace( line );

        // sanity check file name extension: .bax.h5
        size_t blen = strlen( line );
        if ( strcmp( line + ( blen - 7 ), ".bax.h5" ) != 0 )
        {
            fprintf( stderr, "[Warning] - readInputBaxFromFile: ignore line %d, file name extension .bax.h5 is missing (%s)\n", lineNumber, line );
            continue;
        }

        if ( bopt->nBax == bopt->nMaxBax )
        {
            bopt->nMaxBax        = ( (int)( 1.2 * bopt->nMaxBax ) ) + 10;
            bopt->baxIn          = (char**)realloc( bopt->baxIn, sizeof( char* ) * bopt->nMaxBax );
            bopt->numWellNumbers = (int*)realloc( bopt->numWellNumbers, sizeof( int ) * bopt->nMaxBax );
            bopt->wellNumbers    = (int**)realloc( bopt->wellNumbers, sizeof( int* ) * bopt->nMaxBax );
        }

        // sanity check file is accessible
        if ( ( baxTest = fopen( line, "r" ) ) == NULL )
        {
            fprintf( stderr, "[WARNING] - readInputBaxFromFile: Cannot find file %s !\n", line );
            continue;
        }
        else
            fclose( baxTest );

        // add bax file name to list
        bopt->baxIn[ bopt->nBax ] = (char*)malloc( blen + 10 );
        strcpy( bopt->baxIn[ bopt->nBax ], line );
        bopt->numWellNumbers[ bopt->nBax ] = 0;
        bopt->nBax++;
    }
    if ( line )
        free( line );
    fclose( in );
}

void initBaxOptions( BAX_OPT* bopt )
{
    bopt->statOut               = NULL;
    bopt->fastaOut              = NULL;
    bopt->fastqOut              = NULL;
    bopt->quivaOut              = NULL;
    bopt->wellNumbersInFileName = NULL;
    bopt->baxInFileName         = NULL;

    bopt->statFile  = NULL;
    bopt->fastaFile = NULL;
    bopt->fastqFile = NULL;
    bopt->quivaFile = NULL;

    bopt->nBax           = 0;
    bopt->nMaxBax        = 10;
    bopt->baxIn          = (char**)malloc( sizeof( char* ) * bopt->nMaxBax );
    bopt->wellNumbers    = (int**)malloc( sizeof( int* ) * bopt->nMaxBax );
    bopt->numWellNumbers = (int*)malloc( sizeof( int ) * bopt->nMaxBax );

    bopt->VERBOSE             = 0;
    bopt->MAX_LEN             = INT_MAX;
    bopt->MIN_LEN             = 500;
    bopt->MIN_QV              = 750;
    bopt->CUMULATIVE          = 0;
    bopt->READLEN_BIN_SIZE    = 1000;
    bopt->TIME_BIN_SIZE       = 600;
    bopt->MIN_MOVIE_TIME      = 0;
    bopt->MAX_MOVIE_TIME      = MAX_TIME_LIMIT;
    bopt->subreadSel          = all;
    bopt->zmw_minNrOfSubReads = -1;
}

void printBaxOptions( BAX_OPT* bopt )
{
    int i = 0;
    if ( bopt->statOut )
        printf( "statout:         \t%s\n", bopt->statOut );
    else
        printf( "statout:         \tstdout\n" );

    if ( bopt->fastaOut )
        printf( "fastaout:        \t%s\n", bopt->fastaOut );
    if ( bopt->fastqOut )
        printf( "fastqout:        \t%s\n", bopt->fastqOut );
    if ( bopt->quivaOut )
        printf( "quivaOut:        \t%s\n", bopt->quivaOut );

    printf( "#BaxFiles:       \t%d\n", bopt->nBax );
    for ( i = 0; i < bopt->nBax; i++ )
    {
        printf( "file%5d:       \t%s\n", i, bopt->baxIn[ i ] );
        if ( bopt->numWellNumbers[ i ] > 0 )
        {
            int j;
            printf( "selected wells:  \t" );
            for ( j = 1; j < bopt->numWellNumbers[ i ]; j += 2 )
            {
                if ( bopt->wellNumbers[ i ][ j - 1 ] == bopt->wellNumbers[ i ][ j ] )
                    printf( "%d ", bopt->wellNumbers[ i ][ j - 1 ] );
                else
                    printf( "%d-%d ", bopt->wellNumbers[ i ][ j - 1 ], bopt->wellNumbers[ i ][ j ] );
            }

            printf( "\n" );
        }
    }
    if ( bopt->CUMULATIVE )
        printf( "Cumulative:        \tYES\n" );
    else
        printf( "Cumulative:        \tNO\n" );
    printf( "MinLen:            \t%d\n", bopt->MIN_LEN );
    printf( "MinQV:             \t%d\n", bopt->MIN_QV );
    printf( "LenBinSize:        \t%d\n", bopt->READLEN_BIN_SIZE );
    printf( "TimeBinSize:       \t%d\n", bopt->TIME_BIN_SIZE );
    if ( bopt->MIN_MOVIE_TIME < 0 )
        printf( "movieStartTime:  \tNOT SET\n" );
    else
        printf( "movieStartTime:  \t%d\n", bopt->MIN_MOVIE_TIME );
    if ( bopt->MAX_MOVIE_TIME < 0 )
        printf( "movieEndTime:    \tNOT SET\n" );
    else
        printf( "movieEndTime:    \t%d\n", bopt->MAX_MOVIE_TIME );
    switch ( bopt->subreadSel )
    {
        case all:
        default:
            printf( "subreadSelection:\tall\n" );
            break;
        case best:
            printf( "subreadSelection:\tbest\n" );
            break;
        case shortest:
            printf( "subreadSelection:\tshortest\n" );
            break;
        case longest:
            printf( "subreadSelection:\tlongest\n" );
            break;
    }
}

void freeBaxOptions( BAX_OPT* bopt )
{
    if ( bopt->statFile )
        fclose( bopt->statFile );

    if ( bopt->fastaFile )
        fclose( bopt->fastaFile );

    if ( bopt->fastqFile )
        fclose( bopt->fastqFile );

    if ( bopt->quivaFile )
        fclose( bopt->quivaFile );

    int i;
    for ( i = 0; i < bopt->nBax; i++ )
        free( bopt->baxIn[ i ] );

    if ( bopt->wellNumbersInFileName )
    {
        free( bopt->numWellNumbers );
        for ( i = 0; i < bopt->nBax; i++ )
            free( bopt->wellNumbers[ i ] );
        free( bopt->wellNumbers );
    }

    free( bopt->baxIn );
}

void initBaxData( BaxData* b )
{
    b->fullName     = NULL;
    b->shortNameBeg = 0;
    b->shortNameEnd = 0;

    b->baseCall = NULL;
    b->delQV    = NULL;
    b->delTag   = NULL;
    b->insQV    = NULL;
    b->mergeQV  = NULL;
    b->subQV    = NULL;
    b->fastQV   = NULL;

    b->holeStatus = NULL;
    b->numEvent   = NULL;
    b->region     = NULL;

    b->numBase   = 0;
    b->numZMW    = 0;
    b->numRegion = 0;

    b->widthInFrames   = NULL;
    b->preBaseFrames   = NULL;
    b->hqRegionBegTime = NULL;
    b->hqRegionEndTime = NULL;
    b->pausiness       = NULL;
    b->productivity    = NULL;
    b->readType        = NULL;
}

void freeBaxData( BaxData* b )
{
    free( b->baseCall );
    free( b->widthInFrames );
    free( b->numEvent );
    free( b->holeStatus );
    free( b->hqRegionBegTime );
    free( b->region );
    free( b->pausiness );
    free( b->productivity );
    free( b->bindingKit );
    free( b->sequencingKit );
    free( b->softwareVersion );
    if ( b->sequencingChemistry != NULL )
        free( b->sequencingChemistry );
}

void initZMW( ZMW* z )
{
    z->number      = 0;
    z->index       = 0;
    z->regionRow   = 0;
    z->status      = UNKNOWN;
    z->hqBeg       = 0;
    z->hqEnd       = 0;
    z->regionScore = -1;

    z->roff = 0;

    z->maxFrag    = 10;
    z->numFrag    = 0;
    z->insBeg     = (int*)malloc( z->maxFrag * sizeof( int ) );
    z->insEnd     = (int*)malloc( z->maxFrag * sizeof( int ) );
    z->insTimeBeg = (int*)malloc( z->maxFrag * sizeof( int ) );
    z->insTimeEnd = (int*)malloc( z->maxFrag * sizeof( int ) );
    z->toReport   = (char*)malloc( z->maxFrag );

    z->fragQual      = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );
    z->fragSequ      = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );
    z->widthInFrames = (unsigned short**)malloc( z->maxFrag * sizeof( unsigned short* ) );
    z->preBaseFrames = (unsigned short**)malloc( z->maxFrag * sizeof( unsigned short* ) );
    z->delQV         = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );
    z->delTag        = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );
    z->insQV         = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );
    z->mergeQV       = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );

    z->subQV = (unsigned char**)malloc( z->maxFrag * sizeof( unsigned char* ) );
    z->len   = (int*)malloc( z->maxFrag * sizeof( int ) );
    z->avgQV = (float*)malloc( z->maxFrag * sizeof( float ) );

    z->spr = (slowPolymeraseRegions*)malloc( z->maxFrag * sizeof( slowPolymeraseRegions ) );
    int i;
    for ( i = 0; i < z->maxFrag; i++ )
        initSlowPolymeraseRegions( z->spr + i, 100, 50 );
}

void resetZMW( ZMW* z )
{
    z->status      = UNKNOWN;
    z->prod        = prod_NotDefined;
    z->type        = type_NotDefined;
    z->pausiness   = .0;
    z->numFrag     = 0;
    z->hqBeg       = 0;
    z->hqEnd       = 0;
    z->regionScore = -1;
    int i;
    for ( i = 0; i < z->maxFrag; i++ )
        resetSlowPolymeraseRegions( z->spr + i );
}

void ensureZMWCapacity( ZMW* z )
{
    if ( z->numFrag + 1 == z->maxFrag )
    {
        z->maxFrag       = (int)( z->maxFrag * 1.2 ) + 5;
        z->insBeg        = (int*)realloc( z->insBeg, z->maxFrag * sizeof( int ) );
        z->insEnd        = (int*)realloc( z->insEnd, z->maxFrag * sizeof( int ) );
        z->insTimeBeg    = (int*)realloc( z->insTimeBeg, z->maxFrag * sizeof( int ) );
        z->insTimeEnd    = (int*)realloc( z->insTimeEnd, z->maxFrag * sizeof( int ) );
        z->toReport      = (char*)realloc( z->toReport, z->maxFrag );
        z->len           = (int*)realloc( z->len, z->maxFrag * sizeof( int ) );
        z->fragQual      = (unsigned char**)realloc( z->fragQual, z->maxFrag * sizeof( unsigned char* ) );
        z->fragSequ      = (unsigned char**)realloc( z->fragSequ, z->maxFrag * sizeof( unsigned char* ) );
        z->widthInFrames = (unsigned short**)realloc( z->widthInFrames, z->maxFrag * sizeof( unsigned short* ) );
        z->preBaseFrames = (unsigned short**)realloc( z->preBaseFrames, z->maxFrag * sizeof( unsigned short* ) );
        z->delQV         = (unsigned char**)realloc( z->delQV, z->maxFrag * sizeof( unsigned char* ) );
        z->delTag        = (unsigned char**)realloc( z->delTag, z->maxFrag * sizeof( unsigned char* ) );
        z->insQV         = (unsigned char**)realloc( z->insQV, z->maxFrag * sizeof( unsigned char* ) );
        z->mergeQV       = (unsigned char**)realloc( z->mergeQV, z->maxFrag * sizeof( unsigned char* ) );
        z->subQV         = (unsigned char**)realloc( z->subQV, z->maxFrag * sizeof( unsigned char* ) );

        z->avgQV = (float*)realloc( z->avgQV, z->maxFrag * sizeof( float ) );
        z->spr   = (slowPolymeraseRegions*)realloc( z->spr, z->maxFrag * sizeof( slowPolymeraseRegions ) );
        int i;
        for ( i = z->numFrag + 1; i < z->maxFrag; i++ )
            initSlowPolymeraseRegions( z->spr + i, 100, 50 );
    }
}

void deleteZMW( ZMW* z )
{
    z->numFrag = 0;
    z->maxFrag = 0;
    deleteSlowPolymeraseRegions( z->spr );
    free( z->insBeg );
    free( z->insEnd );
    free( z->insTimeBeg );
    free( z->insTimeEnd );
    free( z->toReport );
    free( z->len );
    free( z->fragSequ );
    free( z->fragQual );
    free( z->avgQV );
    free( z->widthInFrames );
    free( z->preBaseFrames );
    free( z->delQV );
    free( z->delTag );
    free( z->insQV );
    free( z->mergeQV );
    free( z->subQV );
    free( z );
}

void printZMW( ZMW* z )
{
    printf( "--------- ZMW ---------\n" );
    printf( "hole number: %d (index: %d) (regionRow: %d) (roff: %d)\n", z->number, z->index, z->regionRow, z->roff );
    printf( "status: %d\n", z->status );
    printf( "pausiness: %.5f\n", z->pausiness );
    printf( "readType: %d\n", z->type );
    printf( "product: %d\n", z->prod );
    printf( "hqBeg: %d\n", z->hqBeg );
    printf( "hqEnd: %d\n", z->hqEnd );
    printf( "score: %d\n", z->regionScore );
    printf( "fragments: %d\n", z->numFrag );
    int i;
    for ( i = 0; i < z->numFrag; i++ )
    {
        printf( "frag %d: report: %d, ins: %d_%d, time: %d-%d len: %d\n", i, z->toReport[ i ], z->insBeg[ i ], z->insEnd[ i ], (int)( z->insTimeBeg[ i ] / FRAME_RATE / 60 ), (int)( z->insTimeEnd[ i ] / FRAME_RATE / 60 ), z->len[ i ] );
    }
}

void initBaxNames( BaxData* b, char* fname )
{
    // set pointer from fname to b->fullname
    b->fullName = fname;

    char* c;
    int epos;

    c = strrchr( fname, '/' );
    if ( c != NULL )
        b->shortNameBeg = c - fname + 1;
    else
        b->shortNameBeg = 0;

    epos = strlen( fname );

    if ( ( epos >= 9 ) && ( ( strcasecmp( fname + ( epos - 9 ), ".1.bax.h5" ) == 0 ) || ( strcasecmp( fname + ( epos - 9 ), ".2.bax.h5" ) == 0 ) || ( strcasecmp( fname + ( epos - 9 ), ".3.bax.h5" ) == 0 ) ) )
        b->shortNameEnd = epos - 9;
    else
        b->shortNameEnd = epos;
}

//  Check if memory needed is above high water mark, and if so allocate

void ensureCapacity( BaxData* b, hsize_t numBaseCalls, hsize_t numHoles, hsize_t numHQReads )
{
    static hsize_t bmax = 0; // base call streams + metrics (number of bases)
    static hsize_t hmax = 0; // streams corresponding to a single ZMW (number of ZMW)
    static hsize_t rmax = 0; // streams correspiding to singles hq subreads (number of all HQ subreads)

    if ( bmax < numBaseCalls )
    {
        bmax = 1.2 * numBaseCalls + 10000;

        if ( ( b->baseCall = (unsigned char*)realloc( b->baseCall, 8ll * bmax ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate basecall buffer\n" );
            exit( 1 );
        }
        b->fastQV  = b->baseCall + bmax;
        b->delQV   = b->fastQV + bmax;
        b->delTag  = b->delQV + bmax;
        b->insQV   = b->delTag + bmax;
        b->mergeQV = b->insQV + bmax;
        b->subQV   = b->mergeQV + bmax;

        if ( ( b->widthInFrames = (unsigned short*)realloc( b->widthInFrames, 2ll * bmax * sizeof( unsigned short ) ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate frame buffer\n" );
            exit( 1 );
        }
        b->preBaseFrames = b->widthInFrames + bmax;
    }
    if ( hmax < numHoles )
    {
        hmax = 1.2 * numHoles + 1000;
        if ( ( b->numEvent = (int*)realloc( b->numEvent, hmax * sizeof( int ) ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate event buffer\n" );
            exit( 1 );
        }

        if ( ( b->holeStatus = (char*)realloc( b->holeStatus, hmax ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate status buffer\n" );
            exit( 1 );
        }

        if ( ( b->hqRegionBegTime = (float*)realloc( b->hqRegionEndTime, 2ll * hmax * sizeof( float ) ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate HQregion buffer\n" );
            exit( 1 );
        }
        b->hqRegionEndTime = b->hqRegionBegTime + hmax;

        if ( ( b->productivity = (unsigned char*)realloc( b->productivity, 2ll * hmax ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate productivity buffer\n" );
            exit( 1 );
        }
        b->readType = b->productivity + hmax;

        if ( ( b->pausiness = (float*)realloc( b->pausiness, hmax * sizeof( float ) ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate pausiness buffer\n" );
            exit( 1 );
        }
    }
    if ( rmax < numHQReads )
    {
        rmax = 1.1 * numHQReads + 1000;
        if ( ( b->region = (int*)realloc( b->region, 5ll * rmax * sizeof( int ) ) ) == NULL )
        {
            fprintf( stderr, "Cannot allocate HQregion buffer\n" );
            exit( 1 );
        }
    }
}

void initBaxStatistic( BaxStatistic* s, BAX_OPT* bopt )
{
    s->nFiles = 0;
    s->nZMWs  = 0;
    memset( s->readTypeHist, 0, ( type_NotDefined + 1 ) * sizeof( s->readTypeHist[ 0 ] ) );
    memset( s->productiveHist, 0, ( prod_NotDefined + 1 ) * sizeof( s->productiveHist[ 0 ] ) );
    memset( s->stateHist, 0, ( UNKNOWN + 1 ) * sizeof( s->stateHist[ 0 ] ) );
    s->cumPausiness = .0f;

    s->minLen         = bopt->MIN_LEN;
    s->minScore       = bopt->MIN_QV;
    s->cumulative     = bopt->CUMULATIVE;
    s->readLenBinSize = bopt->READLEN_BIN_SIZE;
    s->timeLenBinSize = bopt->TIME_BIN_SIZE;
    s->minMovieTime   = bopt->MIN_MOVIE_TIME;
    s->maxMovieTime   = bopt->MAX_MOVIE_TIME;

    s->numSubreadBases = 0;
    s->numSubreads     = 0;

    s->nLenBins = ( MAX_READ_LEN / bopt->READLEN_BIN_SIZE ) + 1;

    s->readLengthHist = NULL;
    if ( ( s->readLengthHist = (uint64*)malloc( 3ll * sizeof( uint64 ) * s->nLenBins ) ) == NULL )
    {
        fprintf( stderr, "Cannot allocate read length histogram buffer\n" );
        exit( 1 );
    }
    s->readLengthBasesHist = s->readLengthHist + s->nLenBins;
    s->readLengthTimeHist  = s->readLengthBasesHist + s->nLenBins;

    s->nTimBins = ( MAX_TIME_LIMIT / bopt->TIME_BIN_SIZE ) + 1;

    s->cumTimeDepQVs = (uint64**)malloc( ( SUB_SUM + 1 ) * sizeof( uint64* ) );
    int i;
    for ( i = NUC_COUNT; i <= SUB_SUM; i++ )
    {
        s->cumTimeDepQVs[ i ] = (uint64*)malloc( s->nTimBins * sizeof( uint64 ) );
        memset( s->cumTimeDepQVs[ i ], 0, s->nTimBins * sizeof( uint64 ) );
    }

    if ( ( s->baseDistributionHist = (uint64**)malloc( BASE_N + 1 * sizeof( uint64* ) * s->nTimBins ) ) == NULL )
    {
        fprintf( stderr, "Cannot allocate base distribution histogram buffer\n" );
        exit( 1 );
    }

    for ( i = BASE_A; i <= BASE_N; i++ )
    {
        s->baseDistributionHist[ i ] = (uint64*)malloc( s->nTimBins * sizeof( uint64 ) );
        memset( s->baseDistributionHist[ i ], 0, s->nTimBins * sizeof( uint64 ) );
    }

    memset( s->readLengthHist, 0, 3 * s->nLenBins * sizeof( s->readLengthHist[ 0 ] ) );
    memset( s->subreadHist, 0, ( MAX_SUBREADS + 1 ) * sizeof( s->subreadHist[ 0 ] ) );

    s->cumSlowPolymeraseRegionLenHist = (uint64*)malloc( 2ll * s->nLenBins * sizeof( uint64 ) );
    s->nSlowPolymeraseRegionLenHist   = s->cumSlowPolymeraseRegionLenHist + s->nLenBins;
    memset( s->cumSlowPolymeraseRegionLenHist, 0, 2ll * s->nLenBins * sizeof( uint64 ) );

    s->cumSlowPolymeraseRegionTimeHist = (uint64*)malloc( s->nTimBins * sizeof( uint64 ) );
    memset( s->cumSlowPolymeraseRegionTimeHist, 0, s->nTimBins * sizeof( uint64 ) );
}

void resetBaxStatistic( BaxStatistic* s )
{
    s->numSubreadBases = 0;
    s->numSubreads     = 0;
    s->nFiles          = 0;
    s->nZMWs           = 0;
    memset( s->readTypeHist, 0, ( type_NotDefined + 1 ) * sizeof( s->readTypeHist[ 0 ] ) );
    memset( s->productiveHist, 0, ( prod_NotDefined + 1 ) * sizeof( s->productiveHist[ 0 ] ) );
    memset( s->stateHist, 0, ( UNKNOWN + 1 ) * sizeof( s->stateHist[ 0 ] ) );
    s->cumPausiness = .0f;

    memset( s->readLengthHist, 0, 3 * s->nLenBins * sizeof( s->readLengthHist[ 0 ] ) );
    int i;
    for ( i = BASE_A; i <= BASE_N; i++ )
        memset( s->baseDistributionHist[ i ], 0, s->nTimBins * sizeof( uint64 ) );

    memset( s->subreadHist, 0, ( MAX_SUBREADS + 1 ) * sizeof( s->subreadHist[ 0 ] ) );

    for ( i = NUC_COUNT; i <= SUB_SUM; i++ )
        memset( s->cumTimeDepQVs[ i ], 0, s->nTimBins * sizeof( uint64 ) );

    memset( s->cumSlowPolymeraseRegionLenHist, 0, 2ll * s->nLenBins * sizeof( uint64 ) );
    memset( s->cumSlowPolymeraseRegionTimeHist, 0, s->nTimBins * sizeof( uint64 ) );
}

void freeBaxStatistic( BaxStatistic* s )
{
    free( s->readLengthHist );

    int i;
    for ( i = NUC_COUNT; i <= SUB_SUM; i++ )
        if ( s->cumTimeDepQVs[ i ] )
            free( s->cumTimeDepQVs[ i ] );

    for ( i = BASE_A; i <= BASE_N; i++ )
        if ( s->baseDistributionHist[ i ] )
            free( s->baseDistributionHist[ i ] );

    free( s->cumTimeDepQVs );
    free( s->baseDistributionHist );
    free( s->cumSlowPolymeraseRegionLenHist );
    free( s->cumSlowPolymeraseRegionTimeHist );
}

void ln_estimate2( unsigned short* data1, unsigned short* data2, int beg, int end, double* mu, double* sig )
{
    double _mu  = 0.0;
    double _sig = 0.0;

    int i;
    for ( i = beg; i < end; i++ )
    {
        _mu += log( data1[ i ] + data2[ i ] );
    }

    _mu = _mu / ( end - beg );

    for ( i = beg; i < end; i++ )
    {
        _sig += SQR( log( data1[ i ] + data2[ i ] ) - _mu );
    }

    _sig = sqrt( _sig / ( end - beg ) );

    *mu  = _mu;
    *sig = _sig;
}

void n_estimate2( unsigned short* data1, unsigned short* data2, int beg, int end, double* mu, double* sig )
{
    double _mu  = 0.0;
    double _sig = 0.0;

    int i;
    for ( i = beg; i < end; i++ )
    {
        _mu += data1[ i ] + data2[ i ];
    }

    _mu = _mu / ( end - beg );

    for ( i = beg; i < end; i++ )
    {
        _sig += SQR( ( data1[ i ] + data2[ i ] ) - _mu );
    }

    _sig = sqrt( _sig / ( end - beg ) );

    *mu  = _mu;
    *sig = _sig;
}

void printBaxError( int errorCode )
{
    fprintf( stderr, "  *** Warning ***: " );
    switch ( errorCode )
    {
        case CANNOT_OPEN_BAX_FILE:
            fprintf( stderr, "Cannot open bax file:\n" );
            break;
        case BAX_BASECALL_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/Basecall from file:\n" );
            break;
        case BAX_DELETIONQV_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/DeletionQV from file:\n" );
            break;
        case BAX_DELETIONTAG_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/DeletionTag from file:\n" );
            break;
        case BAX_INSERTIONQV_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/InsertionQV from file:\n" );
            break;
        case BAX_MERGEQV_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/MergeQV from file:\n" );
            break;
        case BAX_SUBSTITUTIONQV_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/SubstitutionQV from file:\n" );
            break;
        case BAX_QV_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/QualityValue from file:\n" );
            break;
        case BAX_NR_EVENTS_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMW/NumEvent from file:\n" );
            break;
        case BAX_REGION_ERR:
            fprintf( stderr, "Cannot parse /PulseData/Regions from file:\n" );
            break;
        case BAX_HOLESTATUS_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMW/HoleStatus from file:\n" );
            break;
        case BAX_WIDTHINFRAMES_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/WidthInFrames from file:\n" );
            break;
        case BAX_PREBASEFRAMES_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/PreBaseFrames from file:\n" );
            break;
        case BAX_HQREGIONSTARTTIME_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMWMetrics/HQRegionStartTime from file:\n" );
            break;
        case BAX_HQREGIONENDTIME_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMWMetrics/HQRegionEndTime from file:\n" );
            break;
        case BAX_PAUSINESS_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMWMetrics/Pausiness from file:\n" );
            break;
        case BAX_PRODUCTIVITY_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMWMetrics/Productivity from file:\n" );
            break;
        case BAX_READTYPE_ERR:
            fprintf( stderr, "Cannot parse /PulseData/BaseCalls/ZMWMetrics/ReadType from file:\n" );
            break;
        case IGNORE_BAX:
            fprintf( stderr, "Ignore bax file\n" );
            break;

        default:
            fprintf( stderr, "Cannot parse bax file:\n" );
            break;
    }
    fflush( stderr );
}

void initSlowPolymeraseRegions( slowPolymeraseRegions* spr, int segmentWidth, int shift )
{
    spr->segmentWidth = segmentWidth;
    spr->shift        = shift;
    spr->nRegions     = 0;
    spr->nmax         = 10;
    spr->beg          = (int*)malloc( spr->nmax * sizeof( int ) );
    spr->end          = (int*)malloc( spr->nmax * sizeof( int ) );
    spr->numSlowBases = 0;
}

void resetSlowPolymeraseRegions( slowPolymeraseRegions* spr )
{
    spr->nRegions     = 0;
    spr->numSlowBases = 0;
}

void ensureSlowPolymeraseRegionsCapacity( slowPolymeraseRegions* spr )
{
    if ( spr->nRegions + 1 == spr->nmax )
    {
        spr->nmax = (int)( spr->nmax * 1.2 ) + 5;
        spr->beg  = realloc( spr->beg, spr->nmax * sizeof( int ) );
        spr->end  = realloc( spr->end, spr->nmax * sizeof( int ) );
    }
}

void deleteSlowPolymeraseRegions( slowPolymeraseRegions* spr )
{
    free( spr->beg );
    free( spr->end );
    free( spr );
}

int isBaseInSlowPolymeraseRegion( slowPolymeraseRegions* spr, int baseIdx )
{
    if ( spr == NULL )
        return 0;

    int i;
    for ( i = 0; i < spr->nRegions; i++ )
        if ( baseIdx >= spr->beg[ i ] && baseIdx <= spr->end[ i ] )
            return 1;

    return 0;
}

void Print_Number( FILE* out, int64 num, int width )
{
    if ( width == 0 )
    {
        if ( num < 1000ll )
            fprintf( out, "%lld", num );
        else if ( num < 1000000ll )
            fprintf( out, "%lld%c%03lld", num / 1000ll, COMMA, num % 1000ll );
        else if ( num < 1000000000ll )
            fprintf( out, "%lld%c%03lld%c%03lld", num / 1000000ll, COMMA, ( num % 1000000ll ) / 1000ll, COMMA, num % 1000ll );
        else
            fprintf( out, "%lld%c%03lld%c%03lld%c%03lld", num / 1000000000ll,
                     COMMA, ( num % 1000000000ll ) / 1000000ll, COMMA, ( num % 1000000ll ) / 1000ll, COMMA, num % 1000ll );
    }
    else
    {
        if ( num < 1000ll )
            fprintf( out, "%*lld", width, num );
        else if ( num < 1000000ll )
            fprintf( out, "%*lld%c%03lld", width - 4, num / 1000ll, COMMA, num % 1000ll );
        else if ( num < 1000000000ll )
            fprintf( out, "%*lld%c%03lld%c%03lld", width - 8, num / 1000000ll,
                     COMMA, ( num % 1000000ll ) / 1000ll, COMMA, num % 1000ll );
        else
            fprintf( out, "%*lld%c%03lld%c%03lld%c%03lld", width - 12, num / 1000000000ll, COMMA, ( num % 1000000000ll ) / 1000000ll,
                     COMMA, ( num % 1000000ll ) / 1000ll, COMMA, num % 1000ll );
    }
}
