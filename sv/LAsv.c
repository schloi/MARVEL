
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/param.h>

#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include "db/DB.h"
#include "dalign/align.h"

// defaults

#define DEF_ARG_R "repeats"
#define DEF_ARG_RR "repeats"
#define DEF_ARG_N 300

// switches

#define VERBOSE

// structs

typedef struct
{
    HITS_DB* db_ref;
    HITS_DB* db_asm;
    HITS_TRACK* repeats_ref;
    HITS_TRACK* repeats_asm;

    // command line args

    int min_non_repeat_bases;

} LaSvContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static uint64_t count_repeat_bases(uint64_t b, uint64_t e,
                                   uint64_t repanno_b, uint64_t repanno_e,
                                   track_data* rep_data)
{
    uint64_t repeat_bases = 0;

    while ( repanno_b < repanno_e )
    {
        uint64_t rb = rep_data[ repanno_b ];
        uint64_t re = rep_data[ repanno_b + 1 ];
        repanno_b += 2;


        if ( re < b )
        {
            continue;
        }

        if ( rb > e )
        {
            break;
        }

        repeat_bases += intersect( b, e, rb, re );
    }

    return repeat_bases;
}

static void sv_pre( PassContext* pctx, LaSvContext* svctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS sv detection\n" ANSI_COLOR_RESET );
#endif

    UNUSED(pctx);
    UNUSED(svctx);
}

static void sv_post( LaSvContext* svctx )
{
    UNUSED(svctx);
}

static int sv_handler( void* _ctx, Overlap* ovls, int novl )
{
    LaSvContext* svctx = (LaSvContext*)_ctx;
    HITS_DB* db_ref = svctx->db_ref;
    int aread = ovls->aread;
    HITS_TRACK* repeats_ref = svctx->repeats_ref;
    HITS_TRACK* repeats_asm = svctx->repeats_asm;
    track_anno* repref_anno = repeats_ref->anno;
    track_data* repref_data = repeats_ref->data;
    track_anno* repasm_anno = repeats_asm->anno;
    track_data* repasm_data = repeats_asm->data;
    int min_non_repeat_bases = svctx->min_non_repeat_bases;

    printf("processing read %d novl %d\n", aread, novl);

    // a reads -> assembly
    // b reads -> reference

    // tag repeat induced alignments

    int stats_repeat_alignments = 0;
    int i;
    for ( i = 0 ; i < novl ; i++ )
    {
        Overlap* ovl = ovls + i;
        int bread = ovl->bread;
        int b      = repref_anno[ bread ] / sizeof( track_data );
        int e      = repref_anno[ bread + 1 ] / sizeof( track_data );
        int ovllen = ovl->path.bepos - ovl->path.bbpos;
        int blen = DB_READ_LEN( db_ref, bread );

        int bbpos, bepos;

        if ( ovl->flags & OVL_COMP )
        {
            bbpos = blen - ovl->path.bepos;
            bepos = blen - ovl->path.bbpos;
        }
        else
        {
            bbpos = ovl->path.bbpos;
            bepos = ovl->path.bepos;
        }

        int repeat_bases_ref = count_repeat_bases(bbpos, bepos, b, e, repref_data);

        b = repasm_anno[ aread ] / sizeof(track_data);
        e = repasm_anno[ aread + 1 ] / sizeof(track_data);
        int repeat_bases_asm = count_repeat_bases(ovl->path.abpos, ovl->path.aepos,
                    b, e, repasm_data);

        //printf("alnlen %6d ... repeat_bases asm/ref %6d %6d ... left %6d\n",
        //        ovllen, repeat_bases_asm, repeat_bases_ref, ovllen - repeat_bases_ref);

        if ( (repeat_bases_ref > 0 && ovllen - repeat_bases_ref < min_non_repeat_bases) ||
             (repeat_bases_asm > 0 && ovllen - repeat_bases_asm < min_non_repeat_bases) )
        {
            // printf( "overlap %d -> %d: drop due to repeat in b\n", aread, bread );

            stats_repeat_alignments += 1;
            ovl->flags |= OVL_REPEAT;
        }

        printf("%c %c %6d x %6d %7d...%7d x %7d...%7d\n",
            ovl->flags & OVL_REPEAT ? ' ' : '*',
            ovl->flags & OVL_COMP ? 'c' : 'n',
            aread, bread,
            ovl->path.abpos, ovl->path.aepos,
            ovl->path.bbpos, ovl->path.bepos);
    }

    printf("%d alignments tagged as repeat induced\n", stats_repeat_alignments);

    return 1;
}


static void usage()
{
    printf("LAsv [-r <track>] <db> <input.las>\n");
    printf("options: -r track  repeat track (defaults %s)\n", DEF_ARG_R);
};

int main(int argc, char* argv[])
{
    HITS_DB db_ref, db_asm;
    LaSvContext svctx;
    char* track_repeats_ref = DEF_ARG_RR;
    char* track_repeats_asm = DEF_ARG_R;

    bzero(&svctx, sizeof(LaSvContext));

    svctx.min_non_repeat_bases = DEF_ARG_N;

    // process arguments

    opterr = 0;

    int c;
    while ((c = getopt(argc, argv, "n:r:R:")) != -1)
    {
        switch (c)
        {
            case 'n':
                      svctx.min_non_repeat_bases = strtol(optarg, NULL, 10);
                      break;

            case 'r':
                      track_repeats_asm = optarg;
                      break;

            case 'R':
                      track_repeats_ref = optarg;
                      break;

            default:
                      usage();
                      exit(1);
        }
    }

    if (argc - optind < 2)
    {
        usage();
        exit(1);
    }

    char* pcPathReadsAsmIn = argv[optind++];
    char* pcPathReadsRefIn = argv[optind++];
    char* pcPathOverlapsIn = argv[optind++];
    FILE* fileOvlIn = NULL;

    if (Open_DB(pcPathReadsAsmIn, &db_asm))
    {
        fprintf(stderr, "could not open '%s'\n", pcPathReadsAsmIn);
        exit(1);
    }
    svctx.db_asm = &db_asm;

    if (Open_DB(pcPathReadsRefIn, &db_ref))
    {
        fprintf(stderr, "could not open '%s'\n", pcPathReadsRefIn);
        exit(1);
    }
    svctx.db_ref = &db_ref;

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathOverlapsIn );
        exit( 1 );
    }

    if ( ( svctx.repeats_asm = track_load(svctx.db_asm, track_repeats_asm) ) == NULL )
    {
        fprintf( stderr, "failed to open track '%s' for database '%s'\n", track_repeats_asm, pcPathReadsAsmIn);
        exit( 1 );
    }

    if ( ( svctx.repeats_ref = track_load(svctx.db_ref, track_repeats_ref) ) == NULL )
    {
        fprintf( stderr, "failed to open track '%s' for database '%s'\n", track_repeats_ref, pcPathReadsRefIn);
        exit( 1 );
    }

    // init
    PassContext* pctx;

    pctx = pass_init( fileOvlIn, NULL );

    pctx->split_b         = 0;
    pctx->load_trace      = 1;
    pctx->unpack_trace    = 1;
    pctx->data            = &svctx;

    sv_pre( pctx, &svctx );
    pass( pctx, sv_handler );
    sv_post( &svctx );


    // cleanup

    pass_free( pctx );
    Close_DB(svctx.db_asm);
    Close_DB(svctx.db_ref);

    return 0;
}
