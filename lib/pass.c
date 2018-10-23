
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "pass.h"
#include "oflags.h"

static inline size_t ovl_header_length()
{
    return sizeof(ovl_header_novl) + sizeof(ovl_header_twidth);
}


PassContext* pass_init(FILE* fileOvlIn, FILE* fileOvlOut)
{
    PassContext* ctx = malloc(sizeof(PassContext));

    ctx->fileOvlIn = fileOvlIn;
    ctx->trace = NULL;
    ctx->tmax = 0;

    // get file size
    fseeko(ctx->fileOvlIn, 0L, SEEK_END);
    ctx->sizeOvlIn = ftello(ctx->fileOvlIn);
    fseeko(ctx->fileOvlIn, 0L, SEEK_SET);

    if ( ctx->sizeOvlIn == 0 )
    {
        free(ctx);
        return NULL;
    }

    ctx->progress_tick = ctx->sizeOvlIn / 10;

    ovl_header_read(fileOvlIn, &(ctx->novl), &(ctx->twidth));

    if ( ctx->novl == 0 && ctx->sizeOvlIn > (off_t)ovl_header_length() )
    {
        free(ctx);
        return NULL;
    }

    ctx->tbytes = TBYTES( ctx->twidth );

    ctx->off_start = ctx->off_end = 0;
    ctx->progress = 0;
    ctx->split_b = 0;

    if (fileOvlOut)
    {
        ctx->fileOvlOut = fileOvlOut;
        ctx->novl_out = 0;
        ctx->novl_out_discarded = 0;

        ovl_header_write(fileOvlOut, ctx->novl_out, ctx->twidth);

        ctx->write_overlaps = 1;
    }
    else
    {
        ctx->fileOvlOut = NULL;

        ctx->write_overlaps = 0;
    }

    return ctx;
}

void pass_part(PassContext* ctx, off_t start, off_t end)
{
    assert(start < end);

    ctx->off_start = start;
    ctx->off_end = end;
}

void pass_free(PassContext* ctx)
{
    if (ctx->write_overlaps)
    {
        ovl_header_write(ctx->fileOvlOut, ctx->novl_out, ctx->twidth);
    }

    free(ctx->trace);
    free(ctx);
}

void read_unpacked_trace(FILE* fileOvl, Overlap* ovl, size_t tbytes)
{
    Read_Trace(fileOvl, ovl, tbytes);

    if (tbytes == sizeof(uint8))
    {
        Decompress_TraceTo16(ovl);
    }
}

void pass(PassContext* ctx, pass_handler handler)
{
    Overlap* pOvls = NULL;
    int omax = 500;
    pOvls = (Overlap*)malloc(sizeof(Overlap)*omax);

    int split_b = ctx->split_b;
    int load_trace = ctx->load_trace;
    int unpack_trace = ctx->unpack_trace;
    int write_overlaps = ctx->write_overlaps;
    int purge_discarded = ctx->purge_discarded;

    if (ctx->off_start)
    {
        fseeko(ctx->fileOvlIn, ctx->off_start, SEEK_SET);
    }
    else
    {
        fseeko(ctx->fileOvlIn, sizeof(ovl_header_novl) + sizeof(ovl_header_twidth), SEEK_SET);
    }

    if (Read_Overlap(ctx->fileOvlIn, pOvls))
    {
        free(pOvls);

        return ;
    }

    int a, b, n, cont;

    ovl_header_novl i;

    n = i = 0;
    cont = 1;

    ctx->progress_nexttick = ctx->progress_tick;

    while (cont)
    {
        if (ctx->progress)
        {
            off_t pos = ftello(ctx->fileOvlIn);

            if (pos >= ctx->progress_nexttick)
            {
                printf("%3.0f%% done\n", 100.0 * pos / ctx->sizeOvlIn);
                ctx->progress_nexttick += ctx->progress_tick;
            }
        }

        pOvls[0] = pOvls[n];
        a = pOvls->aread;
        b = pOvls->bread;

        if (load_trace)
        {
            if (pOvls[0].path.tlen > ctx->tmax)
            {
                ctx->tmax = 1.2 * ctx->tmax + pOvls[0].path.tlen;
                ctx->trace = realloc(ctx->trace, ctx->tmax * sizeof(ovl_trace));
            }

            pOvls[0].path.trace = ctx->trace;

            Read_Trace(ctx->fileOvlIn, pOvls, ctx->tbytes);

            ctx->tcur = pOvls[0].path.tlen;

            if (unpack_trace && ctx->tbytes == sizeof(uint8))
            {
                Decompress_TraceTo16(pOvls);
            }
        }
        else
        {
            fseeko(ctx->fileOvlIn, ctx->tbytes * pOvls[0].path.tlen, SEEK_CUR);
        }

        n = 1;

        while (1)
        {
            if (Read_Overlap(ctx->fileOvlIn, pOvls+n) ||
                pOvls[n].aread != a ||
                (split_b && pOvls[n].bread != b))
            {
                break;
            }

            if (load_trace)
            {
                if (pOvls[n].path.tlen + ctx->tcur > ctx->tmax)
                {
                    ctx->tmax = 1.2 * ctx->tmax + pOvls[n].path.tlen;

                    ovl_trace* trace = realloc(ctx->trace, ctx->tmax * sizeof(ovl_trace));

                    int j;
                    for (j = 0; j < n; j++)
                    {
                        pOvls[j].path.trace = trace + ((ovl_trace*)(pOvls[j].path.trace) - ctx->trace);
                    }

                    ctx->trace = trace;
                }

                pOvls[n].path.trace = ctx->trace + ctx->tcur;
                Read_Trace(ctx->fileOvlIn, pOvls+n, ctx->tbytes);

                ctx->tcur += pOvls[n].path.tlen;

                if (unpack_trace && ctx->tbytes == sizeof(uint8))
                {
                    Decompress_TraceTo16(pOvls + n);
                }
            }
            else
            {
                fseeko(ctx->fileOvlIn, ctx->tbytes * pOvls[n].path.tlen, SEEK_CUR);
            }

            n += 1;
            if (n >= omax)
            {
                omax = 1.2 * n + 10;
                pOvls = (Overlap*)realloc(pOvls, sizeof(Overlap) * omax);
            }
        }

        cont = handler(ctx->data, pOvls, n);

        if (write_overlaps)
        {
            int j;
            for (j = 0; j < n; j++)
            {
                int isDiscarded = (pOvls[j].flags & OVL_DISCARD);
                if (!purge_discarded || !isDiscarded)
                {
                    if (unpack_trace && ctx->tbytes == sizeof(uint8) && load_trace)
                    {
                        Compress_TraceTo8(pOvls + j);
                    }

                    if (!load_trace)
                    {
                        pOvls[j].path.tlen = 0;
                    }

                    pOvls[j].flags &= ~OVL_TEMP;

                    Write_Overlap(ctx->fileOvlOut, pOvls + j, ctx->tbytes);
                    ctx->novl_out++;

                    if (isDiscarded)
                    {
                        ctx->novl_out_discarded++;
                    }
                }
            }
        }

        i += n;

        if ( !cont || (ctx->off_start && ftello(ctx->fileOvlIn) >= ctx->off_end) || i >= ctx->novl )
        {
            cont = 0;
        }
    }

    free(pOvls);
}

int ovl_header_read(FILE* fileOvl, ovl_header_novl* novl, ovl_header_twidth* twidth)
{
    rewind(fileOvl);

    if (fread(novl, sizeof(ovl_header_novl), 1, fileOvl) != 1)
    {
        return 0;
    }

    if (fread(twidth, sizeof(ovl_header_twidth), 1, fileOvl) != 1)
    {
        return 0;
    }

    return 1;
}

void ovl_header_write(FILE* fileOvl, ovl_header_novl novl, ovl_header_twidth twidth)
{
    rewind(fileOvl);

    fwrite(&novl, sizeof(ovl_header_novl), 1, fileOvl);
    fwrite(&twidth, sizeof(ovl_header_twidth), 1, fileOvl);
}

