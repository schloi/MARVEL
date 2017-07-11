/*******************************************************************************************
 *
 *  BAXstat	: pulls out information from .bax.h5 files produced by Pacbio
 *  		  - statistics i.e. time-dependent quality values,
 *  		  - fasta/fastq and quiva files
 *
 *
 *  Author  : MARVEL Team
 *  Date  	:  Jul 20, 2014
 *
 *
 *  Date  	:  Jul 29, 2014 - added average movie time in respect to subread length
 *                        	- fixed bug: percentage of fragments
 *
 *
 ********************************************************************************************/

#include "lib/stats.h"
#include "limits.h"
#include <ctype.h>
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

#define DEBUG 0
#define VERBOSE_SPR 0
#define MEASURE_TIME 0

#define MAX_NAME 1024

#include "H5dextractUtils.h"

// Fetch the relevant contents of the current bax.h5 file and return the H5 file id.

static int getAttribute( hid_t file_id, char* groupName, char* attrName, char** buffer )
{
    hid_t attr;
    hid_t grp;
    hid_t atype;

    char buf[ MAX_NAME ];

    grp = H5Gopen( file_id, groupName, H5P_DEFAULT );
    if ( grp < 0 )
    {
        printf( "Cannot open %s\n", groupName );
        return -1;
    }

    int na = H5Aget_num_attrs( grp );
    int i, found;
    found = 0;
    for ( i = 0; i < na; i++ )
    {
        attr = H5Aopen_idx( grp, (unsigned int)i );
        // len = H5Aget_name(attr, MAX_NAME, buf);
        //        printf("    Attribute Name : %s\n", buf);

        if ( strcmp( buf, attrName ) == 0 )
        {
            atype = H5Tcopy( H5T_C_S1 );
            H5Tset_size( atype, H5T_VARIABLE );
            H5Aread( attr, atype, buffer );
            //            fprintf(stderr, "Attribute string read was '%s'\n", *buffer);
            H5Tclose( atype );
            found = 1;
        }
        H5Aclose( attr );
    }

    if ( found == 0 )
        *buffer = NULL;

    H5Gclose( grp );
    return 0;
}

static int getBaxData( BaxData* b, BAX_OPT* bopt )
{
    hid_t field_space;
    hid_t field_set;
    hsize_t field_len[ 1 ];
    hid_t file_id;
    herr_t stat;

    H5Eset_auto( H5E_DEFAULT, 0, 0 ); // silence hdf5 error stack

    file_id = H5Fopen( b->fullName, H5F_ACC_RDONLY, H5P_DEFAULT );
    if ( file_id < 0 )
        return ( CANNOT_OPEN_BAX_FILE );

    // ensure capacity
    {
        field_set   = H5Dopen2( file_id, "/PulseData/BaseCalls/Basecall", H5P_DEFAULT );
        field_space = H5Dget_space( field_set );
        if ( field_set < 0 || field_space < 0 )
        {
            H5Fclose( file_id );
            return ( BAX_BASECALL_ERR );
        }
        H5Sget_simple_extent_dims( field_space, field_len, NULL );
        b->numBase = field_len[ 0 ];

        field_set   = H5Dopen2( file_id, "/PulseData/BaseCalls/ZMW/NumEvent", H5P_DEFAULT );
        field_space = H5Dget_space( field_set );
        if ( field_set < 0 || field_space < 0 )
        {
            H5Fclose( file_id );
            return ( BAX_NR_EVENTS_ERR );
        }
        H5Sget_simple_extent_dims( field_space, field_len, NULL );
        b->numZMW = field_len[ 0 ];

        field_set   = H5Dopen2( file_id, "/PulseData/Regions", H5P_DEFAULT );
        field_space = H5Dget_space( field_set );
        if ( field_set < 0 || field_space < 0 )
        {
            H5Fclose( file_id );
            return ( BAX_REGION_ERR );
        }
        H5Sget_simple_extent_dims( field_space, field_len, NULL );
        b->numRegion = field_len[ 0 ];

        if ( getAttribute( file_id, "/ScanData/RunInfo", "SequencingKit", &( b->sequencingKit ) ) < 0 )
            fprintf( stderr, "Cannot read attribute \"SequencingKit\" from group \"/ScanData/RunInfo\" from file %s. \n", b->fullName );
        if ( getAttribute( file_id, "/ScanData/RunInfo", "BindingKit", &( b->bindingKit ) ) < 0 )
            fprintf( stderr, "Cannot read attribute \"BindingKit\" from group \"/ScanData/RunInfo\" from file %s. \n", b->fullName );
        if ( getAttribute( file_id, "/PulseData/BaseCalls", "ChangeListID", &( b->softwareVersion ) ) < 0 )
            fprintf( stderr, "Cannot read attribute \"ChangeListID\" from group \"/PulseData/BaseCalls\" from file %s. \n", b->fullName );
        // optional
        getAttribute( file_id, "/ScanData/RunInfo", "SequencingChemistry", &( b->sequencingChemistry ) );
    }

    ensureCapacity( b, b->numBase, b->numZMW, b->numRegion );

// type is an "enum" :
// 0 -- unsigned char
// 1 -- char
// 2 -- uInt16
// 3 -- int16
// 4 -- uInt32
// 5 -- int32
// 6 -- float
// 7 -- double
#define FETCH( field, path, error, type )                                                                \
    {                                                                                                    \
        field_set   = H5Dopen2( file_id, path, H5P_DEFAULT );                                            \
        field_space = H5Dget_space( field_set );                                                         \
        if ( field_set < 0 || field_space < 0 )                                                          \
        {                                                                                                \
            H5Fclose( file_id );                                                                         \
            return ( error );                                                                            \
        }                                                                                                \
        switch ( type )                                                                                  \
        {                                                                                                \
            case 0:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field );  \
                break;                                                                                   \
            case 1:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field );   \
                break;                                                                                   \
            case 2:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field ); \
                break;                                                                                   \
            case 3:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field );  \
                break;                                                                                   \
            case 4:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field );   \
                break;                                                                                   \
            case 5:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field );    \
                break;                                                                                   \
            case 6:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field );  \
                break;                                                                                   \
            case 7:                                                                                      \
                stat = H5Dread( field_set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, b->field ); \
                break;                                                                                   \
            default:                                                                                     \
                return -11;                                                                              \
        }                                                                                                \
        if ( stat < 0 )                                                                                  \
            return ( error );                                                                            \
        H5Sclose( field_space );                                                                         \
        H5Dclose( field_set );                                                                           \
    }

    // Get read lengths out of the bax file
    FETCH( numEvent, "/PulseData/BaseCalls/ZMW/NumEvent", BAX_NR_EVENTS_ERR, 5 );

    // Get the region annotations out of the bax file
    FETCH( region, "/PulseData/Regions", BAX_REGION_ERR, 5 );
    // check if set of hole numbers is specified, if none of them is present in current file, then skip the reading from bax file
    if ( bopt->wellNumbersInFileName != NULL && bopt->numWellNumbers[ bopt->curBaxFile ] )
    {
        if ( b->region[ 0 ] > bopt->wellNumbers[ bopt->curBaxFile ][ bopt->numWellNumbers[ bopt->curBaxFile ] - 2 ] )
        {
            printf( "first zmw of %d > %d (last zmw in selection range)\n", b->region[ 0 ], bopt->wellNumbers[ bopt->curBaxFile ][ bopt->numWellNumbers[ bopt->curBaxFile ] - 2 ] );
            H5Fclose( file_id );
            return IGNORE_BAX;
        }
        if ( b->region[ ( b->numRegion - 1 ) * 5 ] < bopt->wellNumbers[ bopt->curBaxFile ][ 0 ] )
        {
            printf( "last zmw of %d < %d (first zmw in selection range)\n", b->region[ ( b->numRegion - 1 ) * 5 ], bopt->wellNumbers[ bopt->curBaxFile ][ 0 ] );
            H5Fclose( file_id );
            return IGNORE_BAX;
        }
    }

    FETCH( holeStatus, "/PulseData/BaseCalls/ZMW/HoleStatus", BAX_HOLESTATUS_ERR, 0 );
    FETCH( hqRegionBegTime, "/PulseData/BaseCalls/ZMWMetrics/HQRegionStartTime", BAX_HQREGIONSTARTTIME_ERR, 6 );
    FETCH( hqRegionEndTime, "/PulseData/BaseCalls/ZMWMetrics/HQRegionEndTime", BAX_HQREGIONENDTIME_ERR, 6 );

    // Get all quality streams
    FETCH( baseCall, "/PulseData/BaseCalls/Basecall", BAX_BASECALL_ERR, 0 )
    FETCH( fastQV, "/PulseData/BaseCalls/QualityValue", BAX_QV_ERR, 0 )
    FETCH( delQV, "/PulseData/BaseCalls/DeletionQV", BAX_DELETIONQV_ERR, 0 )
    FETCH( delTag, "/PulseData/BaseCalls/DeletionTag", BAX_DELETIONTAG_ERR, 0 )
    FETCH( insQV, "/PulseData/BaseCalls/InsertionQV", BAX_INSERTIONQV_ERR, 0 )
    FETCH( mergeQV, "/PulseData/BaseCalls/MergeQV", BAX_MERGEQV_ERR, 0 )
    FETCH( subQV, "/PulseData/BaseCalls/SubstitutionQV", BAX_SUBSTITUTIONQV_ERR, 0 )

    // Get times for single base calls
    FETCH( widthInFrames, "/PulseData/BaseCalls/WidthInFrames", BAX_WIDTHINFRAMES_ERR, 2 );
    FETCH( preBaseFrames, "/PulseData/BaseCalls/PreBaseFrames", BAX_PREBASEFRAMES_ERR, 2 );

    // Get additional ZMW statistics
    FETCH( pausiness, "/PulseData/BaseCalls/ZMWMetrics/Pausiness", BAX_PAUSINESS_ERR, 6 );
    FETCH( productivity, "/PulseData/BaseCalls/ZMWMetrics/Productivity", BAX_PRODUCTIVITY_ERR, 0 );
    FETCH( readType, "/PulseData/BaseCalls/ZMWMetrics/ReadType", BAX_READTYPE_ERR, 0 );

    H5Fclose( file_id );
    return ( 0 );
}

static void printBaxStatisticHeader( BaxStatistic* s, BAX_OPT* bopt )
{
    FILE* out = bopt->statFile;
    fprintf( out, "################ BAX statistic setup ##########\n" );
    fprintf( out, "# minLen:          \t" );
    Print_Number( out, (int64)bopt->MIN_LEN, 15 );
    if ( bopt->MAX_LEN < INT_MAX )
    {
        fprintf( out, "# maxLen:          \t" );
        Print_Number( out, (int64)bopt->MAX_LEN, 15 );
    }
    fprintf( out, "\n# minScore:      \t" );
    Print_Number( out, (int64)s->minScore, 15 );
    fprintf( out, "\n# cumulative:    \t" );
    if ( s->cumulative )
        fprintf( out, "            yes" );
    else
        fprintf( out, "             no" );
    fprintf( out, "\n# readLenBinSize:\t" );
    Print_Number( out, (int64)s->readLenBinSize, 15 );
    fprintf( out, "\n# timeLenBinSize:\t" );
    Print_Number( out, (int64)s->timeLenBinSize, 15 );
    fprintf( out, "\n# minMovieTime:   \t" );
    Print_Number( out, (int64)s->minMovieTime, 15 );
    if ( s->minMovieTime == 0 )
        fprintf( out, " (default, not set)" );
    fprintf( out, "\n# maxMovieTime:   \t" );
    Print_Number( out, (int64)s->maxMovieTime, 15 );
    if ( s->maxMovieTime == MAX_TIME_LIMIT )
        fprintf( out, " (default, not set)" );
    switch ( bopt->subreadSel )
    {
        case all:
        default:
            fprintf( out, "\n# subreadSelection:                 all" );
            break;
        case best:
            fprintf( out, "\n# subreadSelection:                best" );
            break;
        case shortest:
            fprintf( out, "\n# subreadSelection:            shortest" );
            break;
        case longest:
            fprintf( out, "\n# subreadSelection:             longest" );
            break;
    }
    fprintf( out, "\n###############################################\n" );
    fflush( out );
}

static void printBaxStatistic( BaxStatistic* s, FILE* out )
{
    ///////////////////// general output
    {
        fprintf( out, "#general ##########################\n" );
        fprintf( out, "\nnumFiles:   \t" );
        Print_Number( out, (int64)s->nFiles, 15 );
        fprintf( out, "\nnumZMWs:   \t" );
        Print_Number( out, (int64)s->nZMWs, 15 );
        fprintf( out, "\nZMWstates:\n" );
        int i;
        for ( i = SEQUENCING; i <= UNKNOWN; i++ )
            if ( s->stateHist[ i ] != 0 )
            {
                switch ( i )
                {
                    case SEQUENCING:
                        fprintf( out, "  SEQUENCING\t" );
                        break;
                    case ANTIHOLE:
                        fprintf( out, "  ANTIHOLE\t" );
                        break;
                    case FIDUCIAL:
                        fprintf( out, "  FIDUCIAL\t" );
                        break;
                    case SUSPECT:
                        fprintf( out, "  SUSPECT\t" );
                        break;
                    case ANTIMIRROR:
                        fprintf( out, "  ANTIMIRROR\t" );
                        break;
                    case FDZMW:
                        fprintf( out, "  FDZMW\t" );
                        break;
                    case FBZMW:
                        fprintf( out, "  FBZMW\t" );
                        break;
                    case ANTIBEAMLET:
                        fprintf( out, "  ANTIBEAMLET\t" );
                        break;
                    case OUTSIDEFOV:
                        fprintf( out, "  OUTSIDEFOV\t" );
                        break;
                    case UNKNOWN:
                        fprintf( out, "  UNKNOWN\t" );
                        break;
                    default:
                        break;
                }
                Print_Number( out, (int64)s->stateHist[ i ], 15 );
                fprintf( out, "\n" );
            }
        fprintf( out, "ZMWproductivity:\n" );
        for ( i = prod_Empty; i <= prod_NotDefined; i++ )
            if ( s->productiveHist[ i ] != 0 )
            {
                switch ( i )
                {
                    case prod_Empty:
                        fprintf( out, "  EMPTY\t" );
                        break;
                    case prod_Productive:
                        fprintf( out, "  PRODUCTIVE\t" );
                        break;
                    case prod_Other:
                        fprintf( out, "  OTHER\t" );
                        break;
                    case prod_NotDefined:
                        fprintf( out, "  NOT_DEFINED\t" );
                        break;
                    default:
                        break;
                }
                Print_Number( out, (int64)s->productiveHist[ i ], 15 );
                fprintf( out, "\n" );
            }
        fprintf( out, "ZMWReadType:\n" );
        for ( i = type_Empty; i <= type_NotDefined; i++ )
            if ( s->readTypeHist[ i ] != 0 )
            {
                switch ( i )
                {
                    case type_Empty:
                        fprintf( out, "  Empty         \t" );
                        break;
                    case type_FullHqRead0:
                        fprintf( out, "  FullHqRead0   \t" );
                        break;
                    case type_FullHqRead1:
                        fprintf( out, "  FullHqRead1   \t" );
                        break;
                    case type_PartialHqRead0:
                        fprintf( out, "  PartialHqRead0\t" );
                        break;
                    case type_PartialHqRead1:
                        fprintf( out, "  PartialHqRead1\t" );
                        break;
                    case type_PartialHqRead2:
                        fprintf( out, "  PartialHqRead2\t" );
                        break;
                    case type_Multiload:
                        fprintf( out, "  Multiload     \t" );
                        break;
                    case type_Indeterminate:
                        fprintf( out, "  Indeterminate \t" );
                        break;
                    case type_NotDefined:
                        fprintf( out, "  NotDefined    \t" );
                        break;
                    default:
                        break;
                }
                Print_Number( out, (int64)s->readTypeHist[ i ], 15 );
                fprintf( out, "\n" );
            }
        fprintf( out, "avgPausiness:     \t%.4f", s->cumPausiness * 1. / s->nZMWs );
        fprintf( out, "\nnumSubreadBases:  " );
        Print_Number( out, (int64)s->numSubreadBases, 15 );
        fprintf( out, "\nnumSubReads:   \t" );
        Print_Number( out, (int64)s->numSubreads, 15 );
        fprintf( out, "\navgReadLen:   \t" );
        Print_Number( out, s->numSubreads > 0 ? (int64)s->numSubreadBases / s->numSubreads : 0, 15 );

        int sprCount, sprBaseCount;
        sprCount = sprBaseCount = 0;
        for ( i = 0; i < s->nLenBins; i++ )
        {
            sprCount += s->nSlowPolymeraseRegionLenHist[ i ];
            sprBaseCount += s->cumSlowPolymeraseRegionLenHist[ i ];
        }
        fprintf( out, "\nnumSubreadsWithSPR:   \t" );
        Print_Number( out, sprCount, 15 );
        fprintf( out, " (%5.2f)", sprCount * 100. / s->numSubreads );
        fprintf( out, "\nnumBasesInSPR:   \t" );
        Print_Number( out, sprBaseCount, 15 );
        fprintf( out, " (%5.2f)", sprBaseCount * 100. / s->numSubreadBases );

        fprintf( out, "\n" );
    }
    ////////////////////////// report subread fragmentation
    {
        int a;
        float cumNum = 0;
        fprintf( out, "\n#Subread fragmentation: #########################\n" );
        for ( a = 0; a <= MAX_SUBREADS; a++ )
        {
            if ( s->subreadHist[ a ] == 0 )
                continue;

            if ( a == MAX_SUBREADS )
                fprintf( out, "\n>" );
            else
                fprintf( out, "\n " );
            Print_Number( out, (int64)a, 4 );
            fprintf( out, "-frag: \t" );
            Print_Number( out, (int64)s->subreadHist[ a ], 10 );
            cumNum += a * s->subreadHist[ a ];
            fprintf( out, "\t(%5.3f%%)\t(%5.3f%%)", a * s->subreadHist[ a ] * 100. / s->numSubreads, cumNum * 100. / s->numSubreads );
        }
        fprintf( out, "\n" );
    }
    /////////////////////// report read length histogram
    {
        int a, c;
        uint64 rcount, bcount;
        fprintf( out, "\n#Distribution of Read Lengths (Bin size = " );
        Print_Number( out, (int64)s->readLenBinSize, 0 );
        fprintf( out, ") #######################\n\n    Bin:      Count  %% Reads  %% Bases   Average     RunTimeAvg (min)     SPR (reads/bases)\n" );
        rcount = bcount = 0;
        for ( a = s->nLenBins - 1; a >= 0; a-- )
            if ( s->readLengthHist[ a ] > 0 )
                break;
        for ( c = 0; c < s->nLenBins - 1; c++ )
            if ( s->readLengthHist[ c ] > 0 )
                break;
        if ( a < 0 )
            fprintf( out, "--- not any high quality read ---" );
        for ( ; a >= c; a-- )
        {
            rcount += s->readLengthHist[ a ];
            bcount += s->readLengthBasesHist[ a ];
            Print_Number( out, ( int64 )( a * s->readLenBinSize ), 7 );
            fprintf( out, ":" );
            Print_Number( out, (int64)s->readLengthHist[ a ], 11 );
            if ( s->readLengthHist[ a ] == 0 )
                fprintf( out, "    %5.1f    %5.1f    %5lld      -                      -  ( - ) /       - ( - )\n", ( 100. * rcount ) / s->numSubreads, ( 100. * bcount ) / s->numSubreadBases, bcount / rcount );
            else
                fprintf( out, "    %5.1f    %5.1f    %5lld      %5.2f           %7llu (%4.2f) / %7llu (%4.2f)\n", ( 100. * rcount ) / s->numSubreads, ( 100. * bcount ) / s->numSubreadBases, bcount / rcount, ( s->readLengthTimeHist[ a ] / FRAME_RATE / s->readLengthHist[ a ] ) / 60., s->nSlowPolymeraseRegionLenHist[ a ], s->nSlowPolymeraseRegionLenHist[ a ] * 100. / s->readLengthHist[ a ], s->cumSlowPolymeraseRegionLenHist[ a ], s->cumSlowPolymeraseRegionLenHist[ a ] * 100. / s->readLengthBasesHist[ a ] );
            if ( rcount == s->numSubreads )
                break;
        }
        fprintf( out, "\n" );
    }
    ///////////////// report time-dependent Base distribution
    {
        int a, c;
        int64 tmp;
        int64 cumBaseA, cumBaseC, cumBaseG, cumBaseT, cumBaseN;
        cumBaseA = cumBaseC = cumBaseG = cumBaseT = cumBaseN = 0;
        fprintf( out, "#movie time dependent nucleotide distribution #####################################\n\n" );
        fprintf( out, "minutes    A      C      G      T      N      AT     GC \n" );
        for ( a = s->nTimBins - 1; a >= 0; a-- )
            if ( ( s->baseDistributionHist[ BASE_A ][ a ] > 0 ) || ( s->baseDistributionHist[ BASE_C ][ a ] > 0 ) || ( s->baseDistributionHist[ BASE_G ][ a ] > 0 ) || ( s->baseDistributionHist[ BASE_T ][ a ] > 0 ) || ( s->baseDistributionHist[ BASE_N ][ a ] > 0 ) )
                break;
        for ( c = 0; c < s->nTimBins - 1; c++ )
            if ( ( s->baseDistributionHist[ BASE_A ][ c ] > 0 ) || ( s->baseDistributionHist[ BASE_C ][ c ] > 0 ) || ( s->baseDistributionHist[ BASE_G ][ c ] > 0 ) || ( s->baseDistributionHist[ BASE_T ][ c ] > 0 ) || ( s->baseDistributionHist[ BASE_N ][ c ] > 0 ) )
                break;
        if ( a < 0 )
            fprintf( out, "--- not any high quality read ---" );
        for ( ; a >= c; a-- )
        {
            tmp = s->baseDistributionHist[ BASE_A ][ a ] + s->baseDistributionHist[ BASE_C ][ a ] + s->baseDistributionHist[ BASE_G ][ a ] + s->baseDistributionHist[ BASE_T ][ a ] + s->baseDistributionHist[ BASE_N ][ a ];
            Print_Number( out, ( int64 )( a * s->timeLenBinSize / 60 ), 7 );
            fprintf( out, "  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f\n", s->baseDistributionHist[ BASE_A ][ a ] * 1. / tmp, s->baseDistributionHist[ BASE_C ][ a ] * 1. / tmp, s->baseDistributionHist[ BASE_G ][ a ] * 1. / tmp, s->baseDistributionHist[ BASE_T ][ a ] * 1. / tmp, s->baseDistributionHist[ BASE_N ][ a ] * 1. / tmp, ( s->baseDistributionHist[ BASE_A ][ a ] + s->baseDistributionHist[ BASE_T ][ a ] ) * 1. / tmp, ( s->baseDistributionHist[ BASE_G ][ a ] + s->baseDistributionHist[ BASE_C ][ a ] ) * 1. / tmp );
            cumBaseA += s->baseDistributionHist[ BASE_A ][ a ];
            cumBaseC += s->baseDistributionHist[ BASE_C ][ a ];
            cumBaseG += s->baseDistributionHist[ BASE_G ][ a ];
            cumBaseT += s->baseDistributionHist[ BASE_T ][ a ];
            cumBaseN += s->baseDistributionHist[ BASE_N ][ a ];
        }
        tmp = cumBaseA + cumBaseC + cumBaseG + cumBaseT + cumBaseN;
        fprintf( out, "\n" );
        fprintf( out, "overall: %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f  %1.3f\n\n", cumBaseA * 1. / tmp, cumBaseC * 1. / tmp, cumBaseG * 1. / tmp, cumBaseT * 1. / tmp, cumBaseN * 1. / tmp, ( cumBaseA + cumBaseT ) * 1. / tmp, ( cumBaseG + cumBaseC ) * 1. / tmp );
    }
    ///////////////// report qualities
    {
        int a, i, c;
        int64 tmp;
        int64 cumBaseQV, cumDelQV, cumInsQV, cumMerQV, cumSubQV;
        cumBaseQV = cumDelQV = cumInsQV = cumMerQV = cumSubQV = 0;
        // cumTimeDepQVs
        fprintf( out, "#movie time dependent average quality distribution #############################\n\n" );
        fprintf( out, "minutes  BASEQV  DELQV    INSQV    MERQV    SUBQV       CONTENT   %%BASES    SPR (%%BASES)\n" );
        for ( a = s->nTimBins - 1; a >= 0; a-- )
            if ( s->cumTimeDepQVs[ NUC_COUNT ][ a ] > 0 )
                break;
        for ( c = 0; c < s->nTimBins - 1; c++ )
            if ( s->cumTimeDepQVs[ NUC_COUNT ][ c ] > 0 )
                break;
        if ( a < 0 )
            fprintf( out, "--- not any high quality read ---" );
        for ( ; a >= c; a-- )
        {
            cumBaseQV += s->cumTimeDepQVs[ QV_SUM ][ a ];
            cumDelQV += s->cumTimeDepQVs[ DEL_SUM ][ a ];
            cumInsQV += s->cumTimeDepQVs[ INS_SUM ][ a ];
            cumMerQV += s->cumTimeDepQVs[ MER_SUM ][ a ];
            cumSubQV += s->cumTimeDepQVs[ SUB_SUM ][ a ];
            Print_Number( out, ( int64 )( a * s->timeLenBinSize / 60 ), 7 );
            fprintf( out, "%7.3f  %7.3f  %7.3f  %7.3f  %7.3f  ", s->cumTimeDepQVs[ QV_SUM ][ a ] * 1. / s->cumTimeDepQVs[ NUC_COUNT ][ a ], s->cumTimeDepQVs[ DEL_SUM ][ a ] * 1. / s->cumTimeDepQVs[ NUC_COUNT ][ a ], s->cumTimeDepQVs[ INS_SUM ][ a ] * 1. / s->cumTimeDepQVs[ NUC_COUNT ][ a ], s->cumTimeDepQVs[ MER_SUM ][ a ] * 1. / s->cumTimeDepQVs[ NUC_COUNT ][ a ], s->cumTimeDepQVs[ SUB_SUM ][ a ] * 1. / s->cumTimeDepQVs[ NUC_COUNT ][ a ] );
            Print_Number( out, (int64)s->cumTimeDepQVs[ NUC_COUNT ][ a ], 12 );
            tmp = 0;
            for ( i = 0; i <= a; ++i )
                tmp += s->cumTimeDepQVs[ NUC_COUNT ][ i ];
            fprintf( out, "  %5.2f", tmp * 100. / s->numSubreadBases );
            fprintf( out, "  %5.2f", 100. * s->cumSlowPolymeraseRegionTimeHist[ a ] / s->cumTimeDepQVs[ NUC_COUNT ][ a ] );
            fprintf( out, "\n" );
        }
        fprintf( out, "\noverall: %5.3f  %7.3f  %7.3f  %7.3f  %7.3f\n\n", cumBaseQV * 1. / s->numSubreadBases, cumDelQV * 1. / s->numSubreadBases, cumInsQV * 1. / s->numSubreadBases, cumMerQV * 1. / s->numSubreadBases, cumSubQV * 1. / s->numSubreadBases );
    }
}

static void getSlowPolymeraseRegions( ZMW* zmw )
{
#if MEASURE_TIME
    clock_t begin, end;
    begin = clock();
#endif
    int a, d, i;
    float numSigma = 2.0;
    int slen;

    slowPolymeraseRegions* spr;
    double lmu, lsig;
    static double help[ 100000 ];
    int segW                 = zmw->spr->segmentWidth;
    int segS                 = zmw->spr->shift;
    int numSuspBaseThreshold = ( segW * 0.5 ) + 1;

    // loop over all selected subreads(fragments) of the current ZMW
    for ( i = 0; i < zmw->numFrag; i++ )
    {
        if ( zmw->toReport[ i ] == 0 )
            continue;

        spr  = zmw->spr + i;
        slen = zmw->len[ i ];

        for ( a       = 0; a < slen; a++ )
            help[ a ] = log( zmw->widthInFrames[ i ][ a ] + zmw->preBaseFrames[ i ][ a ] );

        // evaluate mu and sigma for normal- and log-normal-distribution --> check which works better
        n_estimate_double( help, slen, &lmu, &lsig );
        int suspBase = lmu + ( numSigma * lsig );

#if VERBOSE_SPR
        fprintf( stdout, "SPR %d/%d_%d len%d lmu: %.2f lsig: %.2f, mu: %.2f sig: %.2f\n", zmw->number, zmw->insBeg[ i ], zmw->insEnd[ i ], zmw->len[ i ], lmu, lsig, mu, sig );
#endif
        // iterate over segments
        int numSuspBases;
        int stopA = slen - segW;
        int stopD;
        for ( a = 0; a < stopA; a += segS )
        {
            numSuspBases = 0;
            stopD        = a + segW;
            for ( d = a; d < stopD; d++ )
            {
                if ( help[ d ] > suspBase )
                    numSuspBases++;
            }

            if ( numSuspBases >= numSuspBaseThreshold )
            {
                if ( spr->nRegions == 0 ) // add region directly to spr
                {
                    spr->beg[ 0 ] = a;
                    spr->end[ 0 ] = d;
                    spr->nRegions = 1;
                    spr->numSlowBases += ( d - a + 1 );
#if VERBOSE_SPR
                    fprintf( stdout, "add first interval: %d %d num: %d (#suspBases: %d/%d)\n", spr->beg[ 0 ], spr->end[ 0 ], spr->nRegions, numSuspBases, spr->segmentWidth );
#endif
                }
                else
                {
                    // try to merge intervals
                    if ( a < spr->end[ spr->nRegions - 1 ] )
                    {
#if VERBOSE_SPR
                        fprintf( stdout, "merge interval: %d %d num %d --> %d %d num: %d (#suspBases: %d/%d)\n", spr->beg[ spr->nRegions - 1 ], spr->end[ spr->nRegions - 1 ],
                                 spr->nRegions, spr->beg[ spr->nRegions - 1 ], d, spr->nRegions, numSuspBases, spr->segmentWidth );
#endif
                        spr->numSlowBases += d - spr->end[ spr->nRegions - 1 ];
                        spr->end[ spr->nRegions - 1 ] = d;
                    }
                    // add new interval
                    else
                    {
                        ensureSlowPolymeraseRegionsCapacity( spr );
                        spr->beg[ spr->nRegions ] = a;
                        spr->end[ spr->nRegions ] = d;
                        spr->numSlowBases += ( d - a + 1 );
#if VERBOSE_SPR
                        fprintf( stdout, "add new interval: %d %d num: %d (#suspBases: %d/%d)\n", spr->beg[ spr->nRegions ], spr->beg[ spr->nRegions ], spr->nRegions + 1,
                                 numSuspBases, spr->segmentWidth );
#endif
                        spr->nRegions++;
                    }
                }
            }
        }
    }
#if MEASURE_TIME
    end = clock();
    printf( "SPR TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif
}

static int getNextZMW( BaxData* b, ZMW* zmw )
{
#if MEASURE_TIME
    clock_t begin, end;
    begin = clock();
#endif
    resetZMW( zmw );

#define HOLE 0
#define TYPE 1
#define ADAPTER_REGION 0
#define INSERT_REGION 1
#define HQV_REGION 2
#define START 2
#define FINISH 3
#define SCORE 4

    int* region = ( b->region ) + ( zmw->regionRow * 5 );

    if ( zmw->regionRow == b->numRegion )
        return 0;

    zmw->number    = region[ HOLE ];
    zmw->status    = b->holeStatus[ zmw->index ];
    zmw->pausiness = b->pausiness[ zmw->index ];
    zmw->prod      = b->productivity[ zmw->index ];
    zmw->type      = b->readType[ zmw->index ];

    if ( zmw->index > 0 )
        zmw->roff += b->numEvent[ zmw->index - 1 ];

    // parse line wise all information from Regions that belong to the same ZMW
    while ( region[ HOLE ] == zmw->number && zmw->regionRow < b->numRegion )
    {
        switch ( region[ TYPE ] )
        {
            case ADAPTER_REGION:
                break;
            case INSERT_REGION:
                ensureZMWCapacity( zmw );
                zmw->insBeg[ zmw->numFrag ] = region[ START ];
                zmw->insEnd[ zmw->numFrag ] = region[ FINISH ];
                zmw->numFrag++;
                break;
            case HQV_REGION:
                zmw->hqBeg       = region[ START ];
                zmw->hqEnd       = region[ FINISH ];
                zmw->regionScore = region[ SCORE ];
                break;
            default:
                fprintf( stderr, "unknown region type!\n" );
                exit( 1 );
        }
        region += 5;
        zmw->regionRow++;
    }
    zmw->index++;
#if MEASURE_TIME
    end = clock();
    printf( "ZMW TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif
    return 1;
}

// depends on subreadSelection (all, best, longest, shortest)
// 1. adjusts insert [ start, end ] interval to high quality [ start, end ] interval and if given Min/Max movie times
// 2. checks minimal subread length, (skips subreads that violate the specified minimum length)
// 3. which subreads are pulled from BaxData streams:
// 	all: 			all subreads are pulled from basecall stream
//  best: 		first: for each subread all quality values are reported
//		--> best := 0.2*(subreadLen/HQregionLength)+0.4*(#subreadSegments - #slowPolymeraseSegments/#subreadSegments)+0.4(avgQV/max(QVsFromWholeZMW))
//  longest:	only longest basecall stream is fetched
//  shortest: only shortest basecall stream is fetched
// set toReport flag to appropriate subreads
// returns true, if ZMW contains a valid subread that should be reported, false otherwise
static int getSubreads( BaxData* b, ZMW* zmw, BAX_OPT* bopt )
{
    // adjust insert start and end positions
    int i, j, slen, shortIdx = -1, longIdx = -1, noSubread = 1;

    memset( zmw->toReport, 0, zmw->numFrag );
    for ( i = 0; i < zmw->numFrag; i++ )
    {
        // adjust insBeg/insEnd to HQ region
        {
            if ( zmw->insBeg[ i ] < zmw->hqBeg )
                zmw->insBeg[ i ] = zmw->hqBeg;

            if ( zmw->insEnd[ i ] > zmw->hqEnd )
                zmw->insEnd[ i ] = zmw->hqEnd;

            slen          = zmw->insEnd[ i ] - zmw->insBeg[ i ];
            zmw->len[ i ] = slen;

            if ( slen < bopt->MIN_LEN )
                continue;

            if ( slen > bopt->MAX_LEN )
                continue;
        }
        // set number of frames for insTimeBeg/insTimeEnd (includes also MIN/MAX_MOVIE_TIME)
        {
            int tmpSubreadFrames = 0;
            int curBase          = 0;
            unsigned short *pPBF, *pWIF;
            pPBF                 = b->preBaseFrames + zmw->roff;
            pWIF                 = b->widthInFrames + zmw->roff;
            zmw->insTimeBeg[ i ] = zmw->insTimeEnd[ i ] = -1;

            for ( j = 0; j < zmw->insEnd[ i ]; j++ )
            {
                curBase = *( pPBF + j ) + *( pWIF + j );
                if ( bopt->MIN_MOVIE_TIME < ( tmpSubreadFrames + curBase ) / FRAME_RATE )
                {
                    if ( j > zmw->insBeg[ i ] )
                    {
                        zmw->insBeg[ i ]     = j;
                        zmw->len[ i ]        = zmw->insEnd[ i ] - j;
                        zmw->insTimeBeg[ i ] = tmpSubreadFrames;
                        break;
                    }
                }

                // adjust insTimeBeg, if minimum movie threshold starts before subread starts (default case)
                if ( j >= zmw->insBeg[ i ] )
                {
                    zmw->insTimeBeg[ i ] = tmpSubreadFrames;
                    break;
                }
                tmpSubreadFrames += curBase;
            }

            if ( zmw->insTimeBeg[ i ] == -1 )
                continue; // subread ends before MIN_MOVIE_TIME

            for ( ; j < zmw->insEnd[ i ]; j++ )
            {
                curBase = *( pPBF + j ) + *( pWIF + j );
                if ( bopt->MAX_MOVIE_TIME < ( tmpSubreadFrames + curBase ) / FRAME_RATE )
                {
                    if ( j < zmw->insEnd[ i ] )
                    {
                        zmw->insEnd[ i ] = j;
                        zmw->len[ i ]    = j - zmw->insBeg[ i ];
                    }
                    zmw->insTimeEnd[ i ] = tmpSubreadFrames;
                    break;
                }
                tmpSubreadFrames += *( pPBF + j ) + *( pWIF + j );
            }

            if ( zmw->len[ i ] < bopt->MIN_LEN )
                continue;

            if ( zmw->len[ i ] > bopt->MAX_LEN )
                continue;

            if ( zmw->insTimeEnd[ i ] == -1 )
                zmw->insTimeEnd[ i ] = tmpSubreadFrames;
        }

        noSubread = 0;

        if ( shortIdx < 0 )
            shortIdx = i;
        else if ( slen < zmw->len[ shortIdx ] && slen > bopt->MIN_LEN && slen < bopt->MAX_LEN )
            shortIdx = i;

        if ( longIdx < 0 )
            longIdx = i;
        else if ( slen > zmw->len[ longIdx ] && slen < bopt->MAX_LEN )
            longIdx = i;

        zmw->toReport[ i ] = 1;
    }

    if ( noSubread )
        return 0;

    switch ( bopt->subreadSel )
    {
        case longest:
            memset( zmw->toReport, 0, zmw->numFrag );
            zmw->toReport[ longIdx ] = 1;
            break;
        case shortest:
            memset( zmw->toReport, 0, zmw->numFrag );
            zmw->toReport[ shortIdx ] = 1;
            break;
        case all:
        case best:
        default:
            break;
    }

    int tmpOff;
    for ( i = 0; i < zmw->numFrag; i++ )
    {
        if ( zmw->toReport == 0 )
            continue;

        tmpOff = zmw->roff + zmw->insBeg[ i ];

        // set subread stream pointer
        zmw->fragSequ[ i ]      = b->baseCall + tmpOff;
        zmw->fragQual[ i ]      = b->fastQV + tmpOff;
        zmw->preBaseFrames[ i ] = b->preBaseFrames + tmpOff;
        zmw->widthInFrames[ i ] = b->widthInFrames + tmpOff;
        zmw->delQV[ i ]         = b->delQV + tmpOff;
        zmw->delTag[ i ]        = b->delTag + tmpOff;
        zmw->insQV[ i ]         = b->insQV + tmpOff;
        zmw->mergeQV[ i ]       = b->mergeQV + tmpOff;
        zmw->subQV[ i ]         = b->subQV + tmpOff;

        // calculate average QV
        j         = 0;
        int tmpQV = 0;

        while ( j < zmw->len[ i ] )
        {
            tmpQV += (int)( zmw->fragQual[ i ][ j ] );
            j++;
        }
        zmw->avgQV[ i ] = ( tmpQV * 1.0 ) / zmw->len[ i ];
    }

    // determine slow polymerase chunks within subreads
    getSlowPolymeraseRegions( zmw );

    // if best was selected, then set report flag to best avgQV
    // eval: 0.2*(subreadLen/HQregionLength)+0.4*(subreadLen/#slowPolymeraseBaseCalls)+0.4(avgQV/max(QVsFromWholeZMW))
    if ( bopt->subreadSel == best && zmw->numFrag > 1 )
    {
        // get maximum QV value from all subreads of current ZMW
        int max = 0;
        for ( i = 0; i < zmw->numFrag; i++ )
            for ( j = 0; j < zmw->len[ i ]; j++ )
                if ( zmw->fragQual[ i ][ j ] > max )
                    max = zmw->fragQual[ i ][ j ];

        if ( max == 0 ) // should never happen, otherwise all basecalls have qv of 0!!!
            max = 1;

        int bestIdx   = 0;
        float bestVal = .0, tmpBestVal = .0;

        for ( i = 0; i < zmw->numFrag; i++ )
        {
            if ( zmw->toReport == 0 )
                continue;

            tmpBestVal = .2 * ( 1.0 * zmw->len[ i ] / ( zmw->hqEnd - zmw->hqBeg ) ) + .4 * ( 1.0 * zmw->avgQV[ i ] / max ) + .4 * ( 1.0 * ( zmw->len[ i ] - zmw->spr[ i ].numSlowBases ) / zmw->len[ i ] );

#if DEBUG
            printf( "i: %d, len: %d tmpBestVal: %f, numFrag: %d (%f, %f, %f) \n", i, zmw->len[ i ], tmpBestVal, zmw->numFrag,
                    ( 1.0 * zmw->len[ i ] / ( zmw->hqEnd - zmw->hqBeg ) ),
                    ( 1.0 * zmw->avgQV[ i ] / max ),
                    ( 1.0 * ( zmw->len[ i ] - zmw->spr[ i ].numSlowBases ) / zmw->len[ i ] ) );
#endif
            if ( tmpBestVal > bestVal && zmw->len[ i ] >= bopt->MIN_LEN && zmw->len[ i ] <= bopt->MAX_LEN )
            {
                bestVal = tmpBestVal;
                bestIdx = i;
            }
        }

#if DEBUG
        printf( "set best index: %d\n", bestIdx );
#endif
        memset( zmw->toReport, 0, zmw->numFrag );
        zmw->toReport[ bestIdx ] = 1;
    }

    return 1;
}

static int isInSelectedWellRange( ZMW* zmw, BAX_OPT* bopt )
{
    if ( bopt->wellNumbersInFileName != NULL || !bopt->numWellNumbers[ bopt->curBaxFile ] )
        return 1;

    int* range = bopt->wellNumbers[ bopt->curBaxFile ];
    int well   = zmw->number;

    int i;
    int reps = bopt->numWellNumbers[ bopt->curBaxFile ];

    for ( i = 1; i < reps; i += 2 )
    {
        if ( range[ i - 1 ] <= well && well <= range[ i ] )
            return 1;
    }
    return 0;
}

static void getBaxStats( BaxData* b, BaxStatistic* s, BAX_OPT* bopt )
{
    ZMW zmw;
    initZMW( &zmw );

    int i = 0, j = 0;

    int hqLen;
#if MEASURE_TIME
    clock_t begin, end;
#endif
    while ( getNextZMW( b, &zmw ) )
    {
        // check for sequencing hole
        if ( zmw.status != SEQUENCING )
            continue;

        // check if hq region violates the minimum subread length precondition
        hqLen = zmw.hqEnd - zmw.hqBeg;
        if ( bopt->MIN_LEN > hqLen )
            continue;

        // check if hq quality violates the minimum subread quality precondition
        if ( bopt->MIN_QV > zmw.regionScore )
            continue;

        //
        if ( !isInSelectedWellRange( &zmw, bopt ) )
            continue;

        if ( zmw.numFrag < bopt->zmw_minNrOfSubReads )
            continue;

// determine which subreads should be reported

#if MEASURE_TIME
        begin = clock();
#endif
        if ( !getSubreads( b, &zmw, bopt ) )
            continue;
#if MEASURE_TIME
        end = clock();
        printf( "SUBREAD TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif

        if ( bopt->VERBOSE > 1 )
            printZMW( &zmw );
        // derive statistics
        {
#if MEASURE_TIME
            begin = clock();
#endif
            int bin, numSub;

            s->nZMWs++;
            s->cumPausiness += zmw.pausiness;
            s->productiveHist[ zmw.prod ]++;
            s->readTypeHist[ zmw.type ]++;
            s->stateHist[ zmw.status ]++;

            for ( i = 0, numSub = 0; i < zmw.numFrag; i++ )
            {
                if ( !zmw.toReport[ i ] )
                    continue;

                numSub++;

                if ( zmw.len[ i ] > MAX_READ_LEN )
                    bin = s->nLenBins - 1;
                else
                    bin = zmw.len[ i ] / s->readLenBinSize;

                s->numSubreads++;
                s->numSubreadBases += zmw.len[ i ];
                s->readLengthHist[ bin ]++;
                s->readLengthBasesHist[ bin ] += zmw.len[ i ];
                s->readLengthTimeHist[ bin ] += zmw.insTimeEnd[ i ] - zmw.insTimeBeg[ i ];

                if ( zmw.spr[ i ].nRegions )
                {
                    slowPolymeraseRegions* spr = zmw.spr + i;
                    s->nSlowPolymeraseRegionLenHist[ bin ]++;
                    for ( j = 0; j < spr->nRegions; j++ )
                        s->cumSlowPolymeraseRegionLenHist[ bin ] += ( spr->end[ j ] - spr->beg[ j ] + 1 );
                }

                int tmpBaseFrames = zmw.insTimeBeg[ i ];
                for ( j = 0; j < zmw.len[ i ]; j++ )
                {
                    tmpBaseFrames += ( zmw.preBaseFrames[ i ][ j ] + zmw.widthInFrames[ i ][ j ] );
                    bin = ( tmpBaseFrames / FRAME_RATE ) / s->timeLenBinSize;

                    // add base calling times to appropriate histogram bins
                    if ( bin > s->nTimBins )
                        bin = s->nTimBins - 1;

                    // add base to appropriate histogram bins
                    switch ( zmw.fragSequ[ i ][ j ] )
                    {
                        case 'A':
                            s->baseDistributionHist[ BASE_A ][ bin ]++;
                            break;
                        case 'C':
                            s->baseDistributionHist[ BASE_C ][ bin ]++;
                            break;
                        case 'G':
                            s->baseDistributionHist[ BASE_G ][ bin ]++;
                            break;
                        case 'T':
                            s->baseDistributionHist[ BASE_T ][ bin ]++;
                            break;
                        default:
                            s->baseDistributionHist[ BASE_N ][ bin ]++;
                            break;
                    }

                    // add quality values to appropriate histogram bins
                    s->cumTimeDepQVs[ NUC_COUNT ][ bin ]++;
                    s->cumTimeDepQVs[ QV_SUM ][ bin ] += (int)zmw.fragQual[ i ][ j ];
                    s->cumTimeDepQVs[ DEL_SUM ][ bin ] += (int)zmw.delQV[ i ][ j ];
                    s->cumTimeDepQVs[ INS_SUM ][ bin ] += (int)zmw.insQV[ i ][ j ];
                    s->cumTimeDepQVs[ MER_SUM ][ bin ] += (int)zmw.mergeQV[ i ][ j ];
                    s->cumTimeDepQVs[ SUB_SUM ][ bin ] += (int)zmw.subQV[ i ][ j ];

                    // check time dependent slow polymerase region
                    if ( isBaseInSlowPolymeraseRegion( zmw.spr, j + zmw.insBeg[ i ] ) )
                        s->cumSlowPolymeraseRegionTimeHist[ bin ]++;
                }
            }
            bin = numSub;
            if ( bin >= MAX_SUBREADS )
                bin = MAX_SUBREADS;

            s->subreadHist[ bin ]++;
#if MEASURE_TIME
            end = clock();
            printf( "DERIVE TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif
        }

// fasta output
#if MEASURE_TIME
        begin = clock();
#endif
        if ( bopt->fastaOut )
        {
            for ( i = 0; i < zmw.numFrag; i++ )
            {
                if ( zmw.toReport[ i ] == 0 )
                    continue;

                fprintf( bopt->fastaFile, ">%.*s/%d/%d_%d RQ=0.%d readType=%d", b->shortNameEnd - b->shortNameBeg, b->fullName + b->shortNameBeg, zmw.number, zmw.insBeg[ i ], zmw.insEnd[ i ], zmw.regionScore, zmw.type );
                if ( zmw.spr[ i ].nRegions )
                {
                    slowPolymeraseRegions* spr = zmw.spr + i;
                    fprintf( bopt->fastaFile, " spr=" );
                    for ( j = 0; j < spr->nRegions; j++ )
                    {
                        fprintf( bopt->fastaFile, "%d,%d", spr->beg[ j ], spr->end[ j ] );
                        if ( j + 1 < spr->nRegions )
                            fprintf( bopt->fastaFile, "," );
                    }
                }

                fprintf( bopt->fastaFile, " avgQV=%d Len=%d", (int)( zmw.avgQV[ i ] * 1000 ), zmw.len[ i ] );
                if ( b->bindingKit != NULL && b->sequencingKit != NULL && b->softwareVersion != NULL )
                    fprintf( bopt->fastaFile, " chemistry=%s|%s|%s", b->bindingKit, b->sequencingKit, b->softwareVersion );
                if ( b->sequencingChemistry != NULL )
                    fprintf( bopt->fastaFile, "|%s", b->sequencingChemistry );

                fprintf( bopt->fastaFile, "\n" );
                fprintf( bopt->fastaFile, "%.*s\n", zmw.len[ i ], zmw.fragSequ[ i ] );
            }
        }
        // fastq output
        if ( bopt->fastqOut )
        {
            for ( i = 0; i < zmw.numFrag; i++ )
            {
                if ( zmw.toReport[ i ] == 0 )
                    continue;

                fprintf( bopt->fastqFile, "@%.*s/%d/%d_%d RQ=0.%d readType=%d", b->shortNameEnd - b->shortNameBeg, b->fullName + b->shortNameBeg, zmw.number, zmw.insBeg[ i ], zmw.insEnd[ i ], zmw.regionScore, zmw.type );
                if ( zmw.spr[ i ].nRegions )
                {
                    slowPolymeraseRegions* spr = zmw.spr + i;
                    fprintf( bopt->fastqFile, " spr=" );
                    for ( j = 0; j < spr->nRegions; j++ )
                    {
                        fprintf( bopt->fastqFile, "%d,%d", spr->beg[ j ], spr->end[ j ] );
                        if ( j + 1 < spr->nRegions )
                            fprintf( bopt->fastqFile, "," );
                    }
                }
                fprintf( bopt->fastqFile, "\n" );
                fprintf( bopt->fastqFile, "%.*s\n+\n", zmw.len[ i ], zmw.fragSequ[ i ] );
                for ( j = 0; j < zmw.len[ i ]; j++ )
                    fputc( zmw.fragQual[ i ][ j ] + PHRED_OFFSET, bopt->fastqFile );
                fputc( '\n', bopt->fastqFile );
            }
        }
        // quiva output

        if ( bopt->quivaOut )
        {
            for ( i = 0; i < zmw.numFrag; i++ )
            {
                if ( zmw.toReport[ i ] == 0 )
                    continue;

                fprintf( bopt->quivaFile, "@%.*s/%d/%d_%d RQ=0.%d\n", b->shortNameEnd - b->shortNameBeg, b->fullName + b->shortNameBeg, zmw.number, zmw.insBeg[ i ], zmw.insEnd[ i ], zmw.regionScore );

                for ( j = 0; j < zmw.len[ i ]; j++ )
                    fputc( zmw.delQV[ i ][ j ] + PHRED_OFFSET, bopt->quivaFile );
                fputc( '\n', bopt->quivaFile );

                fprintf( bopt->quivaFile, "%.*s\n", zmw.len[ i ], zmw.delTag[ i ] );

                for ( j = 0; j < zmw.len[ i ]; j++ )
                    fputc( zmw.insQV[ i ][ j ] + PHRED_OFFSET, bopt->quivaFile );
                fputc( '\n', bopt->quivaFile );

                for ( j = 0; j < zmw.len[ i ]; j++ )
                    fputc( zmw.mergeQV[ i ][ j ] + PHRED_OFFSET, bopt->quivaFile );
                fputc( '\n', bopt->quivaFile );

                for ( j = 0; j < zmw.len[ i ]; j++ )
                    fputc( zmw.subQV[ i ][ j ] + PHRED_OFFSET, bopt->quivaFile );
                fputc( '\n', bopt->quivaFile );
            }
        }
#if MEASURE_TIME
        end = clock();
        printf( "OUT TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif

#if DEBUG
        printf( "\n zmw %d (idx: %d) hq: %d %d %d roff %d\n ", zmw.number, zmw.index, zmw.hqBeg, zmw.hqEnd, zmw.regionScore, zmw.roff );
        for ( i = 0; i < zmw.numFrag; i++ )
            printf( "ins %d: %d %d -> report? %d\n ", i, zmw.insBeg[ i ], zmw.insEnd[ i ], zmw.toReport[ i ] );
#endif
    }
}

int main( int argc, char* argv[] )
{
    //  Check that zlib library is present
    if ( !H5Zfilter_avail( H5Z_FILTER_DEFLATE ) )
    {
        fprintf( stderr, "%s: zlib library is not present, check build/installation\n", argv[ 0 ] );
        exit( 1 );
    }
#if MEASURE_TIME
    clock_t begin, end;
    begin = clock();
#endif

    BAX_OPT* bopt = parseBaxOptions( argc, argv );

    if ( bopt->VERBOSE > 1 )
        printBaxOptions( bopt );

    /* here, do your time-consuming job */

    BaxData b;

    initBaxData( &b );

    BaxStatistic s;
    initBaxStatistic( &s, bopt );
    printBaxStatisticHeader( &s, bopt );

#if MEASURE_TIME
    end = clock();
    printf( "INIT TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif
    {
        int i, ecode;
        for ( i = 0; i < bopt->nBax; i++ )
        {
            bopt->curBaxFile = i;
            if ( !bopt->CUMULATIVE )
                resetBaxStatistic( &s );

            initBaxNames( &b, bopt->baxIn[ i ] );
#if MEASURE_TIME
            begin = clock();
#endif

            ecode = getBaxData( &b, bopt ); // parse bax.h5 file

#if MEASURE_TIME
            end = clock();
            printf( "FETCH TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif
            if ( ecode >= 0 )
            {
                s.nFiles++;
#if MEASURE_TIME
                begin = clock();
#endif
                getBaxStats( &b, &s, bopt );
#if MEASURE_TIME
                end = clock();
                printf( "EXTRACT TIME: %f\n", (double)( end - begin ) / CLOCKS_PER_SEC );
#endif
                if ( !bopt->CUMULATIVE )
                    printBaxStatistic( &s, bopt->statFile );
                fflush( stdout );
            }
            else
            {
                fprintf( stderr, " Skipping %s due to failure\n", b.fullName );
                printBaxError( ecode );
            }
        }

        if ( bopt->CUMULATIVE )
            printBaxStatistic( &s, bopt->statFile );
    }

    freeBaxData( &b );
    freeBaxStatistic( &s );
    freeBaxOptions( bopt );

    return 0;
}
