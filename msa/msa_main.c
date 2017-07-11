
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <assert.h>
#include <unistd.h>

#include "lib/oflags.h"
#include "lib/tracks.h"
#include "lib/pass.h"
#include "lib/colors.h"
#include "lib/utils.h"
#include "msa.h"

#include "db/DB.h"
#include "dalign/align.h"


typedef struct
{
    msa* m;

    int rid;
    int a_from;
    int a_to;

    int twidth;

    int nreads;
    char** reads;

    char* base_out;

    FILE* fileOutMsa;
    FILE* fileOutConsensus;
    FILE* fileOutIds;

    HITS_DB* db;

    // re-alignment

    Work_Data* align_work_data;
    Align_Spec* align_spec;

} MsaContext;


extern char *optarg;
extern int optind, opterr, optopt;

static void load_reads(MsaContext* mctx, Overlap* pOvls, int nOvls)
{
    if (nOvls >= mctx->nreads)
    {
        int nreads = mctx->nreads * 1.2 + nOvls + 1;
        mctx->reads = (char**)realloc(mctx->reads, sizeof(char*)*nreads);

        for (;mctx->nreads < nreads; mctx->nreads++)
        {
            mctx->reads[ mctx->nreads ] = New_Read_Buffer(mctx->db);
        }
    }

    Load_Read(mctx->db, pOvls->aread, mctx->reads[0], 0);

    int i;
    for (i = 0; i < nOvls; i++)
    {
        int bread = pOvls[i].bread;
        Load_Read(mctx->db, bread, mctx->reads[i+1], 0);

        if (pOvls[i].flags & OVL_COMP)
        {
            int len = DB_READ_LEN(mctx->db, bread);

            Complement_Seq( mctx->reads[i+1], len );
        }
    }
}

static void write_seq(FILE* file, char* seq)
{
    const int width = 80;
    int len = strlen(seq);
    int j;

    for (j = 0; j + width < len; j += width)
    {
        fprintf(file, "%.*s\n", width, seq + j);
    }

    if (j < len)
    {
        fprintf(file, "%s\n", seq + j);
    }
}

static void pre_msa(PassContext* pctx, MsaContext* mctx)
{
    char* fname = (char*)malloc( strlen(mctx->base_out) + 100);

    sprintf(fname, "%s.cons.fa", mctx->base_out);
    mctx->fileOutConsensus = fopen(fname, "w");

    sprintf(fname, "%s.msa", mctx->base_out);
    mctx->fileOutMsa = fopen(fname, "w");

    sprintf(fname, "%s.ids", mctx->base_out);
    mctx->fileOutIds = fopen(fname, "w");

    free(fname);

    mctx->twidth = pctx->twidth;
}

static void post_msa(MsaContext* mctx)
{
    fclose(mctx->fileOutConsensus);
    fclose(mctx->fileOutMsa);
}

static int handler_msa(void* _ctx, Overlap* pOvls, int nOvls)
{
    MsaContext* ctx = _ctx;

    int a = pOvls->aread;
    int alen = DB_READ_LEN(ctx->db, pOvls->aread);

    if (ctx->rid != -1 && ctx->rid != a)
    {
        return 1;
    }

    msa* m;
    m = msa_init();

    m->twidth = ctx->twidth;

    load_reads(ctx, pOvls, nOvls);

    msa_add(m, ctx->reads[0], 0, alen, 0, alen, NULL, 0, a);

    int j;
    int used = 0;
    for (j = 0; j < nOvls; j++)
    {
        Overlap* o = pOvls + j;

        if (o->path.abpos <= ctx->a_from && o->path.aepos >= ctx->a_to)
        {
            msa_add( m, ctx->reads[ j + 1 ],
                     o->path.abpos, o->path.aepos,
                     o->path.bbpos, o->path.bepos,
                     o->path.trace, o->path.tlen,
                     o->bread );

            if ( o->bread == 163126 )
            {
                msa_print( m, stdout, ctx->a_from, ctx->a_to );
                msa_print_profile( m, stdout, ctx->a_from, ctx->a_to, 1 );
                printf( "\n" );
            }

            used += 1;
        }
    }

    msa_print_simple(m, ctx->fileOutMsa, ctx->fileOutIds, ctx->a_from, ctx->a_to);
    // msa_print_profile(m, stdout, 1);

    char* cons = msa_consensus(m, 0);
    write_seq(ctx->fileOutConsensus, cons);

    msa_free(m);

    return ( ctx->rid == -1 );
}

static void usage()
{
    printf("usage: <db> <overlaps> <base_out> <read.id> <max.ovls> <from> <to>\n");
}

int main(int argc, char* argv[])
{
    FILE* fileOvls;
    HITS_DB db;
    PassContext* pctx;
    MsaContext mctx;

    bzero(&mctx, sizeof(MsaContext));

    mctx.db = &db;

    if (argc != 7)
    {
        usage();
        exit(1);
    }

    char* pcPathReadsIn = argv[1];
    char* pcPathOverlaps = argv[2];
    mctx.base_out = argv[3];
    mctx.rid = atoi(argv[4]);
    mctx.a_from = atoi(argv[5]);
    mctx.a_to = atoi(argv[6]);

    if ( (fileOvls = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }

    if (Open_DB(pcPathReadsIn, &db))
    {
        fprintf(stderr, "could not open '%s'\n", pcPathReadsIn);
        exit(1);
    }

    pctx = pass_init(fileOvls, NULL);

    pctx->split_b = 0;
    pctx->load_trace = 1;
    pctx->unpack_trace = 1;
    pctx->data = &mctx;

    pre_msa(pctx, &mctx);

    pass(pctx, handler_msa);

    post_msa(&mctx);

    Close_DB(&db);

    return 0;
}
