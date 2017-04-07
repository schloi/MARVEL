
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "DB.h"
#include "lib/tracks.h"
#include "FA2x.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

int find_track(CreateContext* ctx, const char* name)
{
    int i;

    for (i = 0; i < ctx->t_cur; i++)
    {
        if (strcmp(ctx->t_name[i], name) == 0)
        {
            return i;
        }
    }

    if (ctx->t_cur >= ctx->t_max)
    {
        int nmax = ctx->t_cur * 1.2 + 10;

        ctx->t_name = (char**) realloc(ctx->t_name, sizeof(char*) * nmax);
        ctx->t_anno = (track_anno**) realloc(ctx->t_anno, sizeof(track_anno*) * nmax);
        ctx->t_data = (track_data**) realloc(ctx->t_data, sizeof(track_data*) * nmax);

        ctx->t_max_anno = (int*) realloc(ctx->t_max_anno, sizeof(int) * nmax);
        ctx->t_max_data = (int*) realloc(ctx->t_max_data, sizeof(int) * nmax);

        ctx->t_cur_anno = (int*) realloc(ctx->t_cur_anno, sizeof(int) * nmax);
        ctx->t_cur_data = (int*) realloc(ctx->t_cur_data, sizeof(int) * nmax);

        ctx->t_max = nmax;
    }

    // printf("find_track> creating '%s' -> %d\n", name, i);

    ctx->t_name[i] = strdup(name);

    ctx->t_max_anno[i] = 1000;
    ctx->t_cur_anno[i] = 0;
    ctx->t_anno[i] = (track_anno*) malloc(sizeof(track_anno) * ctx->t_max_anno[i]);
    bzero(ctx->t_anno[i], sizeof(track_anno) * ctx->t_max_anno[i]);

    ctx->t_max_data[i] = 1000;
    ctx->t_cur_data[i] = 0;
    ctx->t_data[i] = (track_data*) malloc(sizeof(track_data) * ctx->t_max_data[i]);

    ctx->t_cur++;

    return i;
}

void add_to_track(CreateContext* ctx, int track, int64 read, track_data value)
{
    if (read >= ctx->t_max_anno[track])
    {
        uint64 omax = ctx->t_max_anno[track];
        uint64 nmax = read * 1.2 + 1000;

        ctx->t_anno[track] = (track_anno*) realloc(ctx->t_anno[track], sizeof(track_anno) * nmax);
        bzero(ctx->t_anno[track] + omax, sizeof(track_anno) * (nmax - omax));

        ctx->t_max_anno[track] = nmax;
    }

    if (ctx->t_cur_data[track] >= ctx->t_max_data[track])
    {
    	uint64 nmax = ctx->t_cur_data[track] * 1.2 + 1000;
        ctx->t_data[track] = (track_data*) realloc(ctx->t_data[track], sizeof(track_data) * nmax);
        ctx->t_max_data[track] = nmax;
    }

    if (strcmp(ctx->t_name[track], TRACK_PACBIO_CHEM) == 0)
    {
        ctx->t_anno[track][read] += sizeof(char);
    }
    else
    {
        ctx->t_anno[track][read] += sizeof(track_data);
    }

    ctx->t_data[track][ctx->t_cur_data[track]] = value;
    ctx->t_cur_data[track]++;
}

static int getOffsetFromTrack(CreateContext* ctx, char* tname)
{
    FILE* tfile;
    char* fileName = NULL;

    fileName = (char*) malloc(strlen(ctx->pwd) + strlen(ctx->root) + strlen(tname) + 30);
    sprintf(fileName, "%s%s%s.%s.anno", ctx->pwd, PATHSEP, ctx->root, tname);

    if ((tfile = fopen(fileName, "r")) == NULL)
    {
        return 0;
    }

    track_header_len tracklen;
    track_header_size tracksize;

    if (fread(&tracklen, sizeof(track_header_len), 1, tfile) != 1)
    {
        fclose(tfile);
        free(fileName);
        return 0;
    }

    if (tracklen != ctx->initialUreads)
    {
        printf("tracklen != ctx->initialUreads (%d, %d)\n", tracklen, ctx->initialUreads);
        fclose(tfile);
        free(fileName);
        return 0;
    }

    if (fread(&tracksize, sizeof(track_header_size), 1, tfile) != 1)
    {
        fprintf(stderr, "[WARNING] Annotation file %s is corrupt. Unable to parse header size!\n", tname);
        fclose(tfile);
        free(fileName);
        return 0;
    }

    track_anno offset;

    if (fseek(tfile, sizeof(track_anno) * (ctx->initialUreads), SEEK_CUR))
    {
        free(fileName);
        fclose(tfile);
        fprintf(stderr, "[ERROR] - Unable to get last annotation field of track %s\n", tname);
        return 0;
    }

    if (fread(&offset, sizeof(track_anno), 1, tfile) != 1)
    {
        free(fileName);
        fclose(tfile);
        fprintf(stderr, "[ERROR] - Unable to get last annotation field of track %s\n", tname);
        return 0;
    }

    free(fileName);
    fclose(tfile);

    return offset;
}

static void finalize_tracks(CreateContext* ctx)
{
    track_anno coff, off;

    int i;

    for (i = 0; i < ctx->t_cur; i++)
    {
        if (ctx->db->ureads >= ctx->t_max_anno[i])
        {
            int omax = ctx->t_max_anno[i];
            int nmax = ctx->db->ureads + 1;

            ctx->t_anno[i] = (track_anno*) realloc(ctx->t_anno[i], sizeof(track_anno) * nmax);
            bzero(ctx->t_anno[i] + omax, sizeof(track_anno) * (nmax - omax));

            ctx->t_max_anno[i] = nmax;
        }

        off = getOffsetFromTrack(ctx, ctx->t_name[i]);
        int j;

        for (j = ctx->initialUreads; j <= ctx->ureads; j++)
        {
            coff = ctx->t_anno[i][j];
            ctx->t_anno[i][j] = off;
            off += coff;
        }
    }
}

void write_tracks(CreateContext* ctx, char* dbpath)
{
    if (ctx->t_cur == 0)
    {
        return;
    }

    finalize_tracks(ctx);

    char* root = Root(dbpath, ".db");
    char* pwd = PathTo(dbpath);
    FILE* fileOut;

    int i;
    int tmax = strlen(ctx->t_name[0]);

    for (i = 1; i < ctx->t_cur; i++)
    {
        int slen = strlen(ctx->t_name[i]);

        if (tmax < slen)
        {
            tmax = slen;
        }
    }

    int len = strlen(pwd) + 2 + strlen(root) + 1 + tmax + 6;
    char* fname = (char*) malloc(len);
    track_header_len tlen = ctx->ureads;
    track_header_size tsize = sizeof(track_anno);

    for (i = 0; i < ctx->t_cur; i++)
    {
        // anno

        sprintf(fname, "%s/.%s.%s.anno", pwd, root, ctx->t_name[i]);
        // check if file is already available

        int fileExists = 0;

        if ((fileOut = fopen(fname, "r")) != NULL)
        {
            fileExists = 1;
            fclose(fileOut);
        }

        if (fileExists)
        {
            if ((fileOut = fopen(fname, "r+")) == NULL)
            {
                fprintf(stderr, "[ERROR] - Cannot open file %s for appending track %s\n", fname, ctx->t_name[i]);
                exit(1);
            }

            fwrite(&tlen, sizeof(track_header_len), 1, fileOut);
            fwrite(&tsize, sizeof(track_header_size), 1, fileOut);
            fflush(fileOut);
            fseeko(fileOut, -sizeof(track_anno), SEEK_END);
            fwrite(ctx->t_anno[i] + ctx->initialUreads, sizeof(track_anno), (tlen + 1) - ctx->initialUreads, fileOut);
        }
        else
        {
            if ((fileOut = fopen(fname, "w")) == NULL)
            {
                fprintf(stderr, "[WARNING] Cannot create file %s. Skip track %s.\n", fname, ctx->t_name[i]);
                continue;
            }

            fwrite(&tlen, sizeof(track_header_len), 1, fileOut);
            fwrite(&tsize, sizeof(track_header_size), 1, fileOut);
            fflush(fileOut);
            fwrite(ctx->t_anno[i], sizeof(track_anno), (tlen + 1), fileOut);
        }

        fclose(fileOut);

        // data
        sprintf(fname, "%s/.%s.%s.data", pwd, root, ctx->t_name[i]);
        fileOut = fopen(fname, "a");

        if (strcmp(ctx->t_name[i], TRACK_PACBIO_CHEM) == 0)
        {
            int v;

            for (v = 0; v < ctx->t_cur_data[i]; v++)
            {
                fputc((char) ctx->t_data[i][v], fileOut);
            }
        }

        else
        {
            fwrite(ctx->t_data[i], sizeof(track_data), ctx->t_cur_data[i], fileOut);
        }

        fclose(fileOut);
    }

    free(fname);
}

void free_tracks(CreateContext* ctx)
{
    int i;

    for (i = 0; i < ctx->t_cur; i++)
    {
        free(ctx->t_name[i]);
        free(ctx->t_anno[i]);
        free(ctx->t_data[i]);
    }

    free(ctx->t_name);
    free(ctx->t_anno);
    free(ctx->t_data);

    free(ctx->t_max_anno);
    free(ctx->t_cur_anno);
    free(ctx->t_max_data);
    free(ctx->t_cur_data);
}

