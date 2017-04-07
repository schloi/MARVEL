
/*
    maintains coverage statistics for all reads based on .las files produced by daligner and serves mask track

    Author:     MARVEL Team

    Date:       September 2016

    overall usage:
        - create initial mask track with DBdust and split database into blocks with DBsplit
        - start dmask_server with db, expected coverage and optional checkpoint file
        - server listenes on <port> (argument -p) for messages
        - daligner jobs connect to server and retrieve mask track for the block to be processed
        - finished daligner jobs report path of .las file to the dmask server
        - server processes .las files, updates coverage statistics and derives new mask track from it
        - regions of reads receiving an excess of coverage are masked
        - on server shutdown a mask track is written to disk
        - every -c <minutes> the server outputs a checkpoint file containing
          its current state. checkpoints receive an alternating suffix of either .0 or .1. This ensures
          at least one intact file in case the server crashes while writing a checkpoint.
        - to resume from a checkpoint drop the suffix, the server will resume from the file automatically
          on startup
        - amount of worker threads for processing .las files can be set using -t <threads>

    memory usage:
        50x human genome needs roughly 20GB of memory

    CPU/IO requirements:
        .las files are large initially when no repeat regions have been masked

    notes:
        we recommend processing the diagonal of the block vs block matrix initially in order
        to mask the highly repetitive regions of all reads first
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <pthread.h>
#include <semaphore.h>
#include <fcntl.h>
#include <signal.h>
#include <locale.h>

#if defined(__APPLE__)
    #include <sys/syslimits.h>
#else
    #include <linux/limits.h>
#endif

#include "lib/dmask.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/dmask_proto.h"
#include "lib/utils.h"
#include "lib/oflags.h"


#define DEF_ARG_P   DMASK_DEFAULT_PORT
#define DEF_ARG_T        4       // worker threads
#define DEF_ARG_C       10       // checkpoint intervals (min)
#define DEF_ARG_U       10       // track update (min)
#define DEF_ARG_R       10       // report every (min)
#define DEF_ARG_CC       0       // mask contained reads completely
#define DEF_ARG_E        0       // no repeat masking n bases from the read ends

#define COV_LOCK_GRANULARITY    4 // must be a power of 2

#define MAX_COV        255

#define BACKLOG         20       // Passed to listen()

#define COV_FACTOR_THRESHOLD 2   // mask if more than 2 times expected coverage

#define REPORT_INTERVALS             "dmask.report.txt"

#define SEMAPHORE_FILL_COUNT_PREFIX  "sem.queue.fill.count"     // semaphore name prefix

#define IO_BUFFER_SIZE (1024 * 1024)

#undef TRANSFER_ANNOTATION

typedef struct _ServerContext ServerContext;
typedef struct _WorkerContext WorkerContext;
typedef struct _WorkQueueItem WorkQueueItem;
typedef struct _ReadCoverageRle ReadCoverageRle;
typedef struct _ReadCoverage ReadCoverage;

struct _WorkQueueItem
{
    char path[PATH_MAX + 1];
    WorkQueueItem* next;
};

struct _ReadCoverageRle
{
    unsigned char value;
    uint32 count;
};

struct _ReadCoverage
{
    ReadCoverageRle* data;
    size_t dmax;
};

/*
    context for each worker thread
*/
struct _WorkerContext
{
    HITS_DB* db;                  // database
    int id;                       // thread id

    ServerContext* sctx;          // pointer to server context

    Overlap* ovls;                // storage for overlaps
    ovl_header_novl maxovl;       // size allocated for ovls

    Overlap** ovls_sorted;        // pointers to sorted overlaps

    unsigned char* read_cov;             // used for collecting per-base coverage stats
    unsigned char* read_cov_temp; // only used by rle_unpack

    ReadCoverage cov_temp;        // temporary location outside ServerContext for reduced (un)locking

    uint32 trace_max;             // allocated elements for trace
    uint32 trace_cur;             // next free element in trace
    ovl_trace* trace;             // overlap trace points
};

/*
    global state of the dmask server
*/
struct _ServerContext
{
    HITS_DB* db;                            // database
    uint64 bases;                           // number of bases in the db

    // socket buffer
    char* data;                             // socket input buffer
    int dmax;                               // buffer size

    // threads
    pthread_t thread_socket_listener;
    pthread_t thread_reporter;
    pthread_t thread_track;
    pthread_t thread_checkpoint;

    // work queue
    WorkQueueItem* queue_start;             // global work queue containing .las files
    WorkQueueItem* queue_end;               // pointers to start and end of the single-linked list
    pthread_mutex_t queue_lock;             // lock for the work queue
    sem_t* queue_fill_count;                // semaphore containing queue length
    char* path_sem_fill_count;
    int queue_len;

    // coverage statistics
    ReadCoverage* cov;                      // coverage statistics for each read
    int cov_changed;                        // have the coverage stats changed since the last update_track call
    // pthread_mutex_t cov_lock;               // lock for cov and cov_changed

    pthread_mutex_t cov_locks[COV_LOCK_GRANULARITY];

    uint32 lock_mask;
    uint32 lock_shift;

    // track data
    uint64 t_dmax;
    track_anno* t_anno;
    track_data* t_data;
    pthread_mutex_t t_lock;

    // command line arguments
    int cov_expected;
    int port;
    int worker_threads;
    char* checkpoint;

    int checkpoint_suffix;
    int shutdown;
    int report_intervals;                   // write detailed intervals to disk with next report
    int lock;                               // don't update coverage statistics

    int checkpoint_wait;                    // wait time between checkpoints
    int report_wait;
    int track_update_wait;

    int mask_contained;
    int keep_ends;                          // keep n bases at the read ends not masked
};

/*
    we keep a pointer to the ServerContext in the global namespace in order to
    make it available to the signal handlers
*/

ServerContext* g_sctx;

#define rid2lock(rid) ( (rid) >> g_sctx->lock_shift )

/*
static int rid2lock(uint32 rid)
{
    return ( rid >> g_sctx->lock_shift );
}
*/

static void queue_add(ServerContext* ctx, char* path)
{
    WorkQueueItem* item = (WorkQueueItem*)malloc(sizeof(WorkQueueItem));

    if (ctx->queue_end != NULL)
    {
        ctx->queue_end->next = (WorkQueueItem*)item;
    }
    else
    {
        ctx->queue_start = item;
    }

    ctx->queue_end = item;

    item->next = NULL;
    strcpy(item->path, path);

    ctx->queue_len++;
}

static char* queue_remove(ServerContext* ctx)
{
    char* path = strdup(ctx->queue_start->path);
    WorkQueueItem* item = ctx->queue_start;

    if (ctx->queue_start == ctx->queue_end)
    {
        ctx->queue_end = NULL;
    }

    ctx->queue_start = (WorkQueueItem*)(item->next);

    free(item);

    ctx->queue_len--;

    return path;
}

static int cmp_ovls(const void* a, const void* b)
{
    Overlap* x = *(Overlap**)a;
    Overlap* y = *(Overlap**)b;

    return x->aread - y->aread;
}

static int rle_pack(unsigned char* data, int ndata, ReadCoverageRle** _packed, size_t* _maxpacked)
{
    ReadCoverageRle* packed = *_packed;
    int maxpacked = *_maxpacked;

    int i, j;

    int segments = 1;
    for (i = 1; i < ndata; i++)
    {
        if (data[i-1] != data[i])
        {
            segments++;
        }
    }

    if (segments + 2 >= maxpacked)
    {
        *_maxpacked = maxpacked = segments * 1.2 + 20;
        *_packed = packed = realloc(packed, maxpacked * sizeof(ReadCoverageRle) );
    }

    unsigned char value = data[0];
    uint32 count = 1;

    for (i = 1, j = 0; i < ndata; i++)
    {
        if (value == data[i])
        {
            count++;
        }
        else
        {
            packed[j].count = count;
            packed[j++].value = value;

            count = 1;
            value = data[i];
        }
    }

    packed[j].count = count;
    packed[j++].value = value;

    packed[j].count = 0;

    return (j+1);
}

static int rle_unpack(unsigned char* unpacked, ReadCoverageRle* packed)
{
    int i, j;
    for (i = 0, j = 0; packed[i].count != 0; i++)
    {
        uint32 count = packed[i].count;
        unsigned char value = packed[i].value;

        while (count)
        {
            unpacked[j++] = value;
            count--;
        }
    }

    return j;
}

static void mask_contained_read(WorkerContext* wctx, int aread, int alen)
{
    ServerContext* sctx = wctx->sctx;
    int unpack = 0;

    int lock = rid2lock(aread); // aread & ( COV_LOCK_GRANULARITY - 1 );

    pthread_mutex_lock( sctx->cov_locks + lock );

    sctx->db->reads[aread].flags = 1;

    // TODO --- redundant ... fix
    // pthread_mutex_lock( &(sctx->cov_lock) );
    {
        if (sctx->cov[aread].data != NULL)
        {
            memcpy(wctx->cov_temp.data, sctx->cov[aread].data, sctx->cov[aread].dmax * sizeof(ReadCoverageRle));
            unpack = 1;
        }
    }
    // pthread_mutex_unlock( &(sctx->cov_lock) );

    if (unpack)
    {
        bzero(wctx->read_cov_temp, sizeof(unsigned char) * alen);
        rle_unpack(wctx->read_cov_temp, wctx->cov_temp.data);

        int i;
        for (i = 0; i < alen; i++)
        {
            wctx->read_cov[i] = MAX_COV;
        }
    }

    size_t dcur = rle_pack(wctx->read_cov, alen,  &(wctx->cov_temp.data), &(wctx->cov_temp.dmax));

    // pthread_mutex_lock( &(sctx->cov_lock) );
    {
        if (dcur >= sctx->cov[aread].dmax)
        {
            sctx->cov[aread].dmax = dcur * 1.2 + 10;
            sctx->cov[aread].data = realloc(sctx->cov[aread].data, sctx->cov[aread].dmax * sizeof(ReadCoverageRle));
        }

        memcpy(sctx->cov[aread].data, wctx->cov_temp.data, dcur * sizeof(ReadCoverageRle));

        sctx->cov_changed = 1;
    }
    // pthread_mutex_unlock( &(sctx->cov_lock) );

    pthread_mutex_unlock( sctx->cov_locks + lock );
}

static void update_coverage(WorkerContext* wctx, int aread, int alen)
{
    ServerContext* sctx = wctx->sctx;
    int unpack = 0;

    int lock = rid2lock(aread); // aread & ( COV_LOCK_GRANULARITY - 1 );

    pthread_mutex_lock( sctx->cov_locks + lock );

    // TODO --- redundant ... fix
    // pthread_mutex_lock( &(sctx->cov_lock) );
    {
        if (sctx->cov[aread].data != NULL)
        {
            memcpy(wctx->cov_temp.data, sctx->cov[aread].data, sctx->cov[aread].dmax * sizeof(ReadCoverageRle));
            unpack = 1;
        }
    }
    // pthread_mutex_unlock( &(sctx->cov_lock) );

    if (unpack)
    {
        bzero(wctx->read_cov_temp, sizeof(unsigned char) * alen);
        rle_unpack(wctx->read_cov_temp, wctx->cov_temp.data);

        int i;
        for (i = 0; i < alen; i++)
        {
            uint32 cov = wctx->read_cov[i] + wctx->read_cov_temp[i];

            if (cov > MAX_COV)
            {
                cov = MAX_COV;
            }

            wctx->read_cov[i] = cov;
        }
    }

    size_t dcur = rle_pack(wctx->read_cov, alen,  &(wctx->cov_temp.data), &(wctx->cov_temp.dmax));

    // pthread_mutex_lock( &(sctx->cov_lock) );
    {
        if (dcur >= sctx->cov[aread].dmax)
        {
            sctx->cov[aread].dmax = dcur * 1.2 + 10;
            sctx->cov[aread].data = realloc(sctx->cov[aread].data, sctx->cov[aread].dmax * sizeof(ReadCoverageRle));
        }

        memcpy(sctx->cov[aread].data, wctx->cov_temp.data, dcur * sizeof(ReadCoverageRle));

        sctx->cov_changed = 1;
    }
    // pthread_mutex_unlock( &(sctx->cov_lock) );

    pthread_mutex_unlock( sctx->cov_locks + lock );
}

static void write_mask_tracks(ServerContext* sctx)
{
    HITS_DB* db = sctx->db;
    int nreads = db->nreads;

    track_anno* anno_r = malloc(sizeof(track_anno) * (db->nreads + 1));
    track_data* data_r = malloc( sctx->t_anno[nreads] );
    int dcur_r = 0;

    track_anno* anno_c = malloc(sizeof(track_anno) * (db->nreads + 1));
    track_data* data_c = malloc( 2 * sizeof(track_data) * db->nreads );
    int dcur_c = 0;

    int i;
    for ( i = 0; i < nreads; i++ )
    {
        track_anno beg = sctx->t_anno[i] / sizeof(track_data);
        track_anno end = sctx->t_anno[i + 1] / sizeof(track_data);

        anno_r[i] = 0;
        anno_c[i] = 0;

        if (db->reads[i].flags == 1)
        {
            data_c[dcur_c++] = 0;
            data_c[dcur_c++] = DB_READ_LEN(db, i);

            anno_c[i] += 2 * sizeof(track_data);
        }
        else
        {
            while (beg < end)
            {
                track_data b = sctx->t_data[beg];
                track_data e = sctx->t_data[beg + 1];

                data_r[dcur_r++] = b;
                data_r[dcur_r++] = e;

                beg += 2;
                anno_r[i] += 2 * sizeof(track_data);
            }
        }
    }

    track_anno off_r = 0;
    track_anno off_c = 0;

    for (i = 0; i <= nreads; i++)
    {
        track_anno tmp = anno_c[i];
        anno_c[i] = off_c;
        off_c += tmp;

        tmp = anno_r[i];
        anno_r[i] = off_r;
        off_r += tmp;
    }

    track_write(db, TRACK_MASK_R, 0, anno_r, data_r, anno_r[ nreads ] / sizeof(track_data));
    track_write(db, TRACK_MASK_C, 0, anno_c, data_c, anno_c[ nreads ] / sizeof(track_data));

    // write_track_trimmed(db, TRACK_MASK_R, 0, anno_r, data_r, anno_r[ nreads ] / sizeof(track_data));
    // write_track_trimmed(db, TRACK_MASK_C, 0, anno_c, data_c, anno_c[ nreads ] / sizeof(track_data));
}

static void process_las(WorkerContext* ctx, const char* path)
{
    int mask_contained = ctx->sctx->mask_contained;

    FILE* fileIn = fopen(path, "r");

    if (fileIn == NULL)
    {
        fprintf(stderr, "could not open %s\n", path);
        return ;
    }

    ovl_header_novl novl;
    ovl_header_twidth twidth;

    if (!ovl_header_read(fileIn, &novl, &twidth))
    {
        fprintf(stderr, "failed to read header of %s\n", path);
        return ;
    }

    // TODO ... handle garbage novl value

    int tbytes = TBYTES(twidth);
    if (novl > ctx->maxovl)
    {
        ctx->maxovl = novl * 1.2 + 100;
        ctx->ovls = (Overlap*)realloc(ctx->ovls, ctx->maxovl * sizeof(Overlap));
        ctx->ovls_sorted = (Overlap**)realloc(ctx->ovls_sorted, ctx->maxovl * sizeof(Overlap*));
    }

    ovl_header_novl i;
    for (i = 0; i < novl; i++)
    {
        Overlap* ovl = ctx->ovls + i;

        if (Read_Overlap(fileIn, ovl))
        {
            break;
        }

#ifdef TRANSFER_ANNOTATION

        if (ovl->path.tlen + ctx->trace_cur > ctx->trace_max)
        {
            ctx->trace_max = 1.2 * ctx->trace_max + ovl->path.tlen;

            ovl_trace* trace = realloc(ctx->trace, ctx->trace_max * sizeof(ovl_trace));

            ovl_header_novl j;

            for (j = 0; j < i; j++)
            {
                ctx->ovls[j].path.trace = trace + ((ovl_trace*) (ctx->ovls[j].path.trace) - ctx->trace);
            }

            ctx->trace = trace;
        }

        ovl->path.trace = ctx->trace + ctx->trace_cur;
        Read_Trace(fileIn, ovl, tbytes);

        ctx->trace_cur += ovl->path.tlen;

        if (tbytes == sizeof(uint8))
        {
            Decompress_TraceTo16(ovl);
        }

#else
        fseeko(fileIn, ovl->path.tlen * tbytes, SEEK_CUR);
#endif

        ctx->ovls_sorted[i] = ovl;
    }
    novl = i;

    fclose(fileIn);

    if (novl == 0)
    {
        fprintf(stderr, "reading las file yielded %lld overlaps\n", novl);
        return ;
    }

    qsort(ctx->ovls_sorted, novl, sizeof(Overlap*), cmp_ovls);

    // count multiple B-read overlaps as a single hit

    int ovlALen = DB_READ_LEN(ctx->db, ctx->ovls_sorted[0]->aread);

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl_o = ctx->ovls_sorted[i];

        if (mask_contained)
        {
            if (ovl_o->path.abpos == 0 && ovl_o->path.aepos == ovlALen)
            {
                mask_contained_read(ctx, ovl_o->aread, ovlALen);
            }

            if (ovl_o->path.bbpos == 0)
            {
                int blen = DB_READ_LEN(ctx->db, ovl_o->bread);

                if (ovl_o->path.bepos == blen)
                {
                    mask_contained_read(ctx, ovl_o->bread, blen);
                }
            }
        }

        ovl_header_novl j;
        for ( j = i + 1; j < novl ; j++ )
        {
            Overlap* ovl_i = ctx->ovls_sorted[j];

            if ( ovl_o->bread != ovl_i->bread )
            {
                break ;
            }

            if ( ovl_i->path.abpos <= ovl_o->path.aepos &&
                 ovl_o->path.abpos <= ovl_i->path.aepos )
            {
                int ab = MIN(ovl_o->path.abpos, ovl_i->path.abpos);
                int ae = MAX(ovl_o->path.aepos, ovl_i->path.aepos);

                ovl_i->path.abpos = ab;
                ovl_i->path.aepos = ae;

                ovl_o->flags |= OVL_DISCARD;
                break ;
            }
        }
    }

    int keep_ends = ctx->sctx->keep_ends;

    if (keep_ends != -1)
    {
        int prevovlALen = 0;
        bzero(ctx->read_cov, ovlALen);

        Overlap* prevovl = NULL;
        for (i = 0; i < novl; i++)
        {
            Overlap* ovl = ctx->ovls_sorted[i];
            ovlALen = DB_READ_LEN(ctx->db, ovl->aread);

            if (ovl->flags & OVL_DISCARD)
            {
                continue;
            }

            if (ovl->aread == ovl->bread)
            {
                continue;
            }

            if (prevovl && prevovl->aread != ovl->aread)
            {
                update_coverage(ctx, prevovl->aread, prevovlALen);
                bzero(ctx->read_cov, ovlALen);
            }

            int beg, end;
            if (keep_ends > 0)
            {
                beg = MAX(ovl->path.abpos, keep_ends);
                end = MIN(ovl->path.aepos, ovlALen - keep_ends);
            }
            else
            {
                beg = ovl->path.abpos;
                end = ovl->path.aepos;
            }

            int j;
            for (j = beg; j < end; j++)
            {
                if (ctx->read_cov[j] == MAX_COV)
                {
                    continue;
                }

                ctx->read_cov[j]++;
            }

            prevovl = ovl;
            prevovlALen = ovlALen;
        }

        update_coverage(ctx, prevovl->aread, prevovlALen);
    }

#ifdef TRANSFER_ANNOTATION
    int preva = -1;
    ServerContext* sctx = ctx->sctx;
    int cov_threshold = sctx->cov_expected * COV_FACTOR_THRESHOLD;
    int has_repeat = 0;

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ctx->ovls + i;
        int aread = ovl->aread;

        if (aread != preva)
        {
            has_repeat = 0;
            int lock = rid2lock(aread);

            pthread_mutex_lock( sctx->cov_locks + lock );

            unsigned char* unpacked = ctx->read_cov;
            ReadCoverageRle* packed = sctx->cov[aread].data;

            // TODO --- trim intervals to next segment boundary

            int k, j;
            for (k = 0, j = 0; packed[k].count != 0; k++)
            {
                uint32 count = packed[k].count;
                unsigned char value = packed[k].value;

                if (value > cov_threshold)
                {
                    // TODO --- only keep mod 100 intervals

                    has_repeat = 1;
                    memset(unpacked + j, MAX_COV, count);
                }

                j += count;
            }

            pthread_mutex_unlock( ctx->sctx->cov_locks + lock );

            preva = ovl->aread;
        }
    }

#endif

}

/*
static int map_repeat_to_b(Overlap* ovl, int* _beg, int* _end)
{
    int beg = *_beg;
    int end = *_end;

    int ab = ovl->path.abpos;
    int ae = ovl->path.aepos;

    if (end - beg < MIN_INT_LEN)
    {
        return 0;
    }

    // does the repeat interval intersect with overlap

    int iab = MAX(beg, ab);
    int iae = MIN(end, ae);

    if (iab >= iae)
    {
        return 0;
    }

    // establish conservative estimate of the repeat extent
    // relative to the b read given the trace points

    ovl_trace* trace = ovl->path.trace;
    int tlen = ovl->path.tlen;

    int ibb = -1;
    int ibe = -1;

    int aoff = ovl->path.abpos;
    int boff = ovl->path.bbpos;

    int j;
    for (j = 0; j < tlen; j += 2)
    {
        if ( (aoff >= iab || j == tlen - 2) && ibb == -1 )
        {
            ibb = boff;
        }

        aoff = ( (aoff + twidth) / twidth) * twidth;

        if ( aoff >= iae && ibe == -1 )
        {
            if (ibb == -1)
            {
                ibb = boff;
            }

            ibe = boff + trace[j + 1];

            break;
        }

        boff += trace[j + 1];
    }

    if (ibb == -1 || ibe == -1)
    {
        return 0;
    }

    if (ovl->flags & OVL_COMP)
    {
        int t = ibb;
        int blen = DB_READ_LEN(ctx->db, ovl->bread);

        ibb = blen - ibe;
        ibe = blen - t;
    }

    assert(ibb <= ibe);

    *_beg = ibb;
    *_end = ibe;

    return 1;
}
*/

static void update_track(ServerContext* sctx)
{
    time_t time_beg = time(NULL);
    int nreads = sctx->db->nreads;
    int i;
    uint64 dcur = 0;

    /*
    for (i = 0; i < COV_LOCK_GRANULARITY; i++)
    {
        pthread_mutex_lock( sctx->cov_locks + i );
    }
    */


    bzero(sctx->t_anno, sizeof(track_anno) * (nreads + 1));

    track_data* t_data = sctx->t_data;
    track_anno* t_anno = sctx->t_anno;
    uint64 t_dmax = sctx->t_dmax;
    int cov_threshold = sctx->cov_expected * COV_FACTOR_THRESHOLD;

    for (i = 0; i < nreads; i++)
    {
        ReadCoverage* cov = sctx->cov + i;
        ReadCoverageRle* rle = cov->data;

        if (rle)
        {
            int j = 0;
            int pos = 0;
            uint64 dcur_prev = dcur;

            while (rle[j].count)
            {
                if (rle[j].value > cov_threshold)
                {
                    if (dcur + 2 >= t_dmax)
                    {
                        sctx->t_dmax = t_dmax = dcur * 1.2 + 1000;
                        sctx->t_data = t_data = (track_data*)realloc(t_data, sizeof(track_data) * t_dmax );
                    }

                    if (dcur != dcur_prev && pos == t_data[dcur-1])
                    {
                        t_data[dcur-1] = pos + rle[j].count;
                    }
                    else
                    {
                        t_data[dcur++] = pos;
                        t_data[dcur++] = pos + rle[j].count;

                        t_anno[i] += sizeof(track_data) * 2;
                    }

                }

                pos += rle[j].count;

                j++;
            }
        }
    }

    /*
    for (i = 0; i < COV_LOCK_GRANULARITY; i++)
    {
        pthread_mutex_unlock( sctx->cov_locks + i );
    }
    */


    track_anno off = 0;
    for (i = 0; i <= nreads; i++)
    {
        track_anno tmp = t_anno[i];
        t_anno[i] = off;
        off += tmp;
    }

    sctx->cov_changed = 0;

    printf("UPDATE TRACK (%ld secs)\n", time(NULL) - time_beg);
}

static void shutdown_server(ServerContext* ctx)
{
    ctx->shutdown = 1;

    // TODO ... cheap hack to wake up all worker threads so they terminate

    int i;
    for (i = 0; i < ctx->worker_threads; i++)
    {
        sem_post( ctx->queue_fill_count );
    }
}

static void socket_data_handler(ServerContext* ctx, int newsock, fd_set *set)
{
    DmHeader header;

    if (recv(newsock, &header, sizeof(header), 0) != sizeof(header))
    {
        fprintf(stderr, "CLOSE CONNECTION\n");

        FD_CLR(newsock, set);
        close(newsock);

        return ;
    }

    char* data;
    int dcur = 0;
    int pending = header.length - sizeof(header);

    if (ctx->dmax < pending)
    {
        ctx->dmax = pending * 2;
        ctx->data = realloc(ctx->data, ctx->dmax);
    }

    data = ctx->data;

    while (pending)
    {
        int received = recv(newsock, data + dcur, pending, 0);

        if (received < 1)
        {
            break;
        }

        dcur += received;
        pending -= received;
    }

    if (pending != 0)
    {
        fprintf(stderr, "error %d bytes pending\n", pending);
        return;
    }

    switch (header.type)
    {
        case DM_TYPE_SHUTDOWN:
            shutdown_server(ctx);

            break;

        case DM_TYPE_LOCK:
            if (ctx->lock)
            {
                printf("LOCK FAILED ... already locked\n");
            }
            else
            {
                printf("LOCK SUCCESS\n");
                ctx->lock = 1;
            }
            break;

        case DM_TYPE_UNLOCK:
            if (ctx->lock)
            {
                printf("UNLOCK SUCCESS\n");
                ctx->lock = 0;
            }
            else
            {
                printf("UNLOCK FAILED ... not locked\n");
            }
            break;

        case DM_TYPE_INTERVALS:
            ctx->report_intervals = 1;
            break;

        case DM_TYPE_WRITE_TRACK:

            pthread_mutex_lock( &(ctx->t_lock) );

            write_mask_tracks(ctx);

            pthread_mutex_unlock( &(ctx->t_lock) );

            break;

        case DM_TYPE_LAS_AVAILABLE:
            if (ctx->lock)
            {
                printf("updates locked, ignoring available .las files\n");
            }
            else
            {
                pthread_mutex_lock(&(ctx->queue_lock));
                {
                    int i;
                    int beg = 0;
                    for (i = 0; i < dcur; i++)
                    {
                        if (data[i] == '\0')
                        {
                            queue_add(ctx, data + beg);
                            sem_post( ctx->queue_fill_count );

                            printf("QUEUE LEN %3d ADD %s\n", ctx->queue_len, data + beg);

                            beg = i + 1;
                        }
                    }
                }

                pthread_mutex_unlock(&(ctx->queue_lock));
            }

            break;

        case DM_TYPE_REQUEST_TRACK:
            {
            uint64 bfirst = header.reserved1;
            uint64 nreads = header.reserved2;

            printf("REQUEST TRACK ... bfirst = %llu nreads = %llu\n", bfirst, nreads);

            track_anno* anno = (track_anno*)malloc( sizeof(track_anno) * (nreads + 1) );

            pthread_mutex_lock( &(ctx->t_lock) );
            {
                memcpy(anno, ctx->t_anno + bfirst, sizeof(track_anno) * (nreads + 1));
                uint64 i;
                track_anno off = anno[0];
                for (i = 0; i <= nreads; i++)
                {
                    anno[i] -= off;
                }
                uint64 dlen = anno[nreads];

                // DATA: ctx->t_data[ off : off + dlen ]

                DmHeader headerResp;
                bzero(&headerResp, sizeof(DmHeader));

                headerResp.version = DM_VERSION;
                headerResp.type = DM_TYPE_RESPONSE_TRACK;
                headerResp.length = sizeof(DmHeader) +
                                    sizeof(track_anno) * (nreads + 1) +
                                    dlen;

                printf("  SENDING %llu bytes (ANNO %llu DATA %llu)\n",
                        headerResp.length,
                        sizeof(track_anno) * (nreads + 1),
                        dlen);

                send(newsock, &headerResp, sizeof(DmHeader), 0);

                // send track

                send(newsock, anno, sizeof(track_anno) * (nreads + 1), 0);
                send(newsock, ctx->t_data + off / sizeof(track_data), dlen, 0);
            }
            pthread_mutex_unlock( &(ctx->t_lock) );

            free(anno);

            }
            break;

        default:
            fprintf(stderr, "unknown message type %d\n", header.type);
            break;
    }
}

static void* socket_listener_thread(void *arg)
{
    ServerContext* ctx = (ServerContext*)arg;

    int sock;
    fd_set socks;
    fd_set readsocks;
    int maxsock;
    int reuseaddr = 1; /* True */

    /* Create the socket */
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        perror("socket");
        return NULL;
    }

    /* Enable the socket to reuse the address */
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(int)) < 0)
    {
        perror("setsockopt");
        return NULL;
    }

    /* Bind to the address */
    struct sockaddr_in serv_addr;
    bzero(&serv_addr, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(ctx->port);

    if (bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
    {
        perror("bind");
        return NULL;
    }

    /* Listen */
    if (listen(sock, BACKLOG) < 0)
    {
        perror("listen");
        return NULL;
    }

    /* Set up the fd_set */
    FD_ZERO(&socks);
    FD_SET(sock, &socks);
    maxsock = sock;

    printf("socket listener thread reporting for duty (port %d)\n", ctx->port);

    /* Main loop */
    while (1)
    {
        if (ctx->shutdown)
        {
            printf("socket listener thread shutting down\n");
            break;
        }

        int s;
        readsocks = socks;
        if (select(maxsock + 1, &readsocks, NULL, NULL, NULL) == -1)
        {
            perror("select");
            return NULL;
        }

        for (s = 0; s <= maxsock; s++)
        {
            if (FD_ISSET(s, &readsocks))
            {
                // printf("SOCKET %d READY\n", s);

                if (s == sock)
                {
                    /* New connection */
                    int newsock;
                    struct sockaddr_in their_addr;
                    socklen_t size = sizeof(struct sockaddr_in);

                    newsock = accept(sock, (struct sockaddr*)&their_addr, &size);

                    if (newsock == -1)
                    {
                        perror("accept");
                    }
                    else
                    {
                        printf("CONNECT FROM %s:%d\n", inet_ntoa(their_addr.sin_addr), htons(their_addr.sin_port));

                        FD_SET(newsock, &socks);
                        if (newsock > maxsock)
                        {
                            maxsock = newsock;
                        }
                    }
                }
                else
                {
                    socket_data_handler(ctx, s, &socks);
                }
            }
        }
    }

    close(sock);

    return NULL;
}

static void* worker_thread(void *arg)
{
    WorkerContext* wctx = (WorkerContext*)arg;
    ServerContext* sctx = wctx->sctx;

    printf("worker thread %d reporting for duty\n", wctx->id);

    while (1)
    {
        if (sem_wait( sctx->queue_fill_count ))
        {
            perror("sem_wait");
        }

        if (sctx->shutdown)
        {
            break;
        }

        pthread_mutex_lock(&(sctx->queue_lock));

        // TODO ... file doesn't necessarily have to be visible to us (slow distributed filesystem, etc)

        char* path = queue_remove(sctx);
        printf("THREAD %2d PROCESS %s\n", wctx->id, path);

        pthread_mutex_unlock(&(sctx->queue_lock));

        process_las(wctx, path);

        free(path);
    }

    return NULL;
}

static void worker_ctx_init(ServerContext* sctx, WorkerContext* wctx)
{
    wctx->sctx = sctx;
    wctx->maxovl = 1000;
    wctx->ovls = (Overlap*)malloc(sizeof(Overlap) * wctx->maxovl);
    wctx->ovls_sorted = (Overlap**)malloc(sizeof(Overlap*) * wctx->maxovl);

    wctx->read_cov = malloc( sctx->db->maxlen );
    wctx->read_cov_temp = malloc( sctx->db->maxlen );

    wctx->cov_temp.dmax = sctx->db->maxlen * sizeof(ReadCoverageRle);
    wctx->cov_temp.data = malloc( wctx->cov_temp.dmax );
    wctx->db = sctx->db;

    wctx->trace_max = 1 * 1024 * 1024;
    wctx->trace_cur = 0;
    wctx->trace = malloc( sizeof(ovl_trace) * wctx->trace_max );
}

static void worker_ctx_free(WorkerContext* wctx)
{
    free(wctx->ovls);
    free(wctx->ovls_sorted);
    free(wctx->read_cov);
    free(wctx->read_cov_temp);
    free(wctx->cov_temp.data);
    free(wctx->trace);
}

static void* update_track_thread(void* arg)
{
    ServerContext* sctx = (ServerContext*)arg;

    printf("track update thread reporting for duty\n");

    while (1)
    {
        int i = sctx->track_update_wait * 60;

        while (i > 0 && !sctx->shutdown)
        {
            sleep(1);
            i--;
        }

        if (sctx->shutdown)
        {
            break;
        }

        for ( i = 0 ; i < COV_LOCK_GRANULARITY; i++ )
        {
            pthread_mutex_lock( sctx->cov_locks + i );
        }

        // pthread_mutex_lock( &(sctx->cov_lock) );

        if (sctx->cov_changed)
        {
            pthread_mutex_lock( &(sctx->t_lock) );

            update_track(sctx);

            pthread_mutex_unlock( &(sctx->t_lock) );
        }

        // pthread_mutex_unlock( &(sctx->cov_lock) );

        for ( i = 0 ; i < COV_LOCK_GRANULARITY; i++ )
        {
            pthread_mutex_unlock( sctx->cov_locks + i );
        }
    }

    return NULL;
}

static int checkpoint_write(ServerContext* sctx, FILE* fileOut)
{
    time_t time_beg = time(NULL);

    fwrite( &(sctx->db->nreads), sizeof(sctx->db->nreads), 1, fileOut );

    int i;
    for (i = 0; i < sctx->db->nreads; i++)
    {
        ReadCoverage* cov = sctx->cov + i;

        // count data elements

        size_t items = 0;
        if (cov->data)
        {
            while (cov->data[items].count)
            {
                items++;
            }
        }

        // write count & elements

        fwrite(&items, sizeof(size_t), 1, fileOut);

        if (items > 0)
        {
            fwrite(cov->data, sizeof(ReadCoverageRle), items, fileOut);
        }
    }

    printf("CHECKPOINT (%ld secs)\n", time(NULL) - time_beg);

    return 1;
}

static int checkpoint_read(ServerContext* sctx, FILE* fileIn)
{
    int nreads = 0;

    if (fread(&nreads, sizeof(nreads), 1, fileIn) != 1)
    {
        fprintf(stderr, "failed to read nreads from checkpoint file\n");
        return 0;
    }

    if (nreads != sctx->db->nreads)
    {
        fprintf(stderr, "failed to restore checkpoint. nreads in db not the same as in checkpoint\n");
        return 0;
    }

    size_t items;
    int read;

    for (read = 0; read < nreads && !feof(fileIn); read++)
    {
        if (fread(&items, sizeof(size_t), 1, fileIn) != 1)
        {
            break;
        }

        ReadCoverage* cov = sctx->cov + read;

        if (items >= cov->dmax)
        {
            cov->dmax = items * 1.2 + 10;
            cov->data = (ReadCoverageRle*)realloc(cov->data, sizeof(ReadCoverageRle) * cov->dmax);
        }

        if (fread(cov->data, sizeof(ReadCoverageRle), items, fileIn) != items)
        {
            break;
        }

        cov->data[items].count = 0;
        cov->data[items].value = 0;
    }

    if (read != nreads)
    {
        fprintf(stderr, "warning: checkpoint didn't contain data for all reads.\n");

        for (read = 0; read < nreads; read++)
        {
            ReadCoverage* cov = sctx->cov + read;

            if (cov->data)
            {
                cov->data[0].count = 0;
                cov->data[0].value = 0;
            }
        }
    }

    return 1;
}

static void* checkpoint_thread(void* arg)
{
    ServerContext* sctx = (ServerContext*)arg;

    // pthread_mutex_t* cov_lock = &(sctx->cov_lock);

    char* path = malloc( strlen(sctx->checkpoint) + 20 );

    printf("checkpoint thread reporting for duty\n");

    while (1)
    {
        int i = sctx->checkpoint_wait * 60;

        while (i > 0 && !sctx->shutdown)
        {
            sleep(1);
            i--;
        }

        for ( i = 0 ; i < COV_LOCK_GRANULARITY ; i++)
        {
            pthread_mutex_lock( sctx->cov_locks + i );
        }

        // pthread_mutex_lock(cov_lock);

        sprintf(path, "%s.%d", sctx->checkpoint, sctx->checkpoint_suffix);
        sctx->checkpoint_suffix = (sctx->checkpoint_suffix + 1) % 2;

        FILE* fileOut = fopen(path, "w");
        setvbuf(fileOut, NULL, _IOFBF, IO_BUFFER_SIZE);

        checkpoint_write(sctx, fileOut);

        fclose(fileOut);

        // pthread_mutex_unlock(cov_lock);

        for ( i = 0 ; i < COV_LOCK_GRANULARITY ; i++)
        {
            pthread_mutex_unlock( sctx->cov_locks + i );
        }

        if (sctx->shutdown)
        {
            break;
        }
    }

    free(path);

    return NULL;
}

static void* reporter_thread(void* arg)
{
    int suffix = 0;
    char* path_report = (char*)malloc(strlen(REPORT_INTERVALS) + 20);

    ServerContext* sctx = (ServerContext*)arg;

    uint64 nreads = sctx->db->nreads;
    // pthread_mutex_t* cov_lock = &(sctx->cov_lock);
    FILE* fileOut = NULL;

    printf("report thread reporting for duty\n");

    while (1)
    {
        uint64 i = sctx->report_wait * 60;

        while (i > 0 && !sctx->shutdown)
        {
            sleep(1);
            i--;
        }

        if (sctx->shutdown)
        {
            break;
        }

        if (sctx->report_intervals)
        {
            sprintf(path_report, "%s.%d", REPORT_INTERVALS, suffix);
            suffix++;

            fileOut = fopen(path_report, "w");

            if (!fileOut)
            {
                printf("failed to open %s\n", path_report);
            }
        }

        uint64 reads_with_intervals, bases_masked;
        reads_with_intervals = bases_masked = 0;

        for (i = 0; i < COV_LOCK_GRANULARITY; i++)
        {
            pthread_mutex_lock( sctx->cov_locks + i );
        }

        // pthread_mutex_lock(cov_lock);
        {
            track_anno* t_anno = sctx->t_anno;
            track_data* t_data = sctx->t_data;

            printf("REPORT");

            if (fileOut)
            {
                printf(" (dumping intervals to %s)", path_report);
            }

            printf("\n");

            for (i = 0; i < nreads; i++)
            {
                track_anno ob = t_anno[i];
                track_anno oe = t_anno[i+1];

                if (ob >= oe)
                {
                    continue;
                }

                ob /= sizeof(track_data);
                oe /= sizeof(track_data);

                reads_with_intervals++;

                while (ob < oe)
                {
                    track_data db = t_data[ ob++ ];
                    track_data de = t_data[ ob++ ];

                    bases_masked += de - db;

                    if (sctx->report_intervals)
                    {
                        if (fileOut)
                        {
                            fprintf(fileOut, "%llu %d %d\n", i, db, de);
                        }
                    }
                }
            }
        }
        // pthread_mutex_unlock(cov_lock);

        for (i = 0; i < COV_LOCK_GRANULARITY; i++)
        {
            pthread_mutex_unlock( sctx->cov_locks + i );
        }

        if (sctx->report_intervals)
        {
            sctx->report_intervals = 0;

            if (fileOut)
            {
                fclose(fileOut);
                fileOut = NULL;
            }
        }

        printf("      reads %'12d   bases %'18llu\n",
                sctx->db->nreads, sctx->bases);
        printf("  annotated %'12llu  masked %'18llu (%2d%%)\n",
                reads_with_intervals, bases_masked, (int)(100.0 * bases_masked / sctx->bases));
    }

    free(path_report);

    return NULL;
}

static void ctx_init(ServerContext* ctx, HITS_DB* db, HITS_TRACK* mask)
{
    ctx->shutdown = 0;
    ctx->checkpoint_suffix = 0;
    ctx->report_intervals = 0;
    ctx->lock = 0;

    ctx->t_anno = (track_anno*)malloc(sizeof(track_anno) * (db->nreads + 1));

    ctx->t_dmax = db->nreads * 2;
    ctx->t_data = (track_data*)malloc(sizeof(track_data) * ctx->t_dmax);
    bzero(ctx->t_anno, sizeof(track_anno) * (db->nreads + 1));

    ctx->dmax = 10 * 1024;
    ctx->data = malloc( ctx->dmax );

    ctx->db = db;

    ctx->queue_start = NULL;
    ctx->queue_end = NULL;
    ctx->queue_len = 0;

    ctx->path_sem_fill_count = malloc( sizeof(SEMAPHORE_FILL_COUNT_PREFIX) + 20 );
    sprintf(ctx->path_sem_fill_count, "%s.%d", SEMAPHORE_FILL_COUNT_PREFIX, getpid());

    sem_unlink(ctx->path_sem_fill_count);
    ctx->queue_fill_count = sem_open(ctx->path_sem_fill_count, O_CREAT, 0666, 0);

    if (ctx->queue_fill_count == SEM_FAILED)
    {
        perror("sem_open");
        exit(1);
    }

    pthread_mutex_init(&(ctx->t_lock), NULL);

    pthread_mutex_init(&(ctx->queue_lock), NULL);

    ctx->cov = malloc( sizeof(ReadCoverage) * db->nreads );
    bzero(ctx->cov, sizeof(ReadCoverage) * db->nreads);

    // pthread_mutex_init(&(ctx->cov_lock), NULL);

    int i;
    for (i = 0; i < COV_LOCK_GRANULARITY; i++)
    {
        pthread_mutex_init( ctx->cov_locks + i, NULL );
    }

    FILE* fileIn;

    if (ctx->checkpoint && (fileIn = fopen(ctx->checkpoint, "r")))
    {
        printf("initialising from checkpoint\n");

        checkpoint_read(ctx, fileIn);
        fclose(fileIn);
    }
    else if (mask)
    {
        printf("initialising from mask track\n");

        int i;
        uint64 b, e, alen;
        track_data* data = mask->data;
        track_anno* anno = mask->anno;

        unsigned char* read_cov = malloc( DB_READ_MAXLEN(db) );

        for (i = 0; i < db->nreads; i++)
        {
            b = anno[i] / sizeof(track_data);
            e = anno[i+1] / sizeof(track_data);

            if (b >= e)
            {
                continue;
            }

            // TODO ... rewrite, no need for read_cov

            alen = DB_READ_LEN(db, i);
            bzero(read_cov, alen);

            while (b < e)
            {
                int dust_beg = data[b];
                int dust_end = data[b+1];

                memset(read_cov + dust_beg, MAX_COV, dust_end - dust_beg);

                /*
                while (dust_beg < dust_end)
                {
                    read_cov[ dust_beg ] = MAX_COV;
                    dust_beg++;
                }
                */

                b += 2;
            }

            rle_pack(read_cov, alen, &(ctx->cov[i].data), &(ctx->cov[i].dmax));
        }

        ctx->cov_changed = 1;

        free(read_cov);
    }
    else
    {
        printf("no dust track or checkpoint for initialisation\n");

        unsigned char* read_cov = malloc( DB_READ_MAXLEN(db) );
        bzero(read_cov, DB_READ_MAXLEN(db));

        int i;
        for (i = 0; i < db->nreads; i++)
        {
            int alen = DB_READ_LEN(db, i);
            rle_pack(read_cov, alen, &(ctx->cov[i].data), &(ctx->cov[i].dmax));
        }

        ctx->cov_changed = 1;

        free(read_cov);
    }

    update_track(ctx);

    HITS_READ* reads = ctx->db->reads;

    int64 bases = 0;
    for (i = 0; i < ctx->db->nreads; i++)
    {
        bases += reads[i].rlen;
    }

    ctx->bases = bases;
}

static void ctx_free(ServerContext* ctx)
{
    free(ctx->t_anno);
    free(ctx->t_data);

    sem_close(ctx->queue_fill_count);
    sem_unlink(ctx->path_sem_fill_count);
    free(ctx->path_sem_fill_count);

    pthread_mutex_destroy( &(ctx->t_lock) );
    pthread_mutex_destroy( &(ctx->queue_lock) );
    // pthread_mutex_destroy( &(ctx->cov_lock) );

    int i;
    for (i = 0; i < COV_LOCK_GRANULARITY; i++)
    {
        pthread_mutex_destroy( ctx->cov_locks + i );
    }

    for (i = 0; i < ctx->db->nreads; i++)
    {
        if (ctx->cov[i].data)
        {
            free(ctx->cov[i].data);
        }
    }

    free(ctx->cov);

    free(ctx->data);
}

static void sigint_handler(int sig, siginfo_t *si, void *unused)
{
    printf("SIGINT received ... shutting down\n");

    // silence unused variable warnings
    UNUSED(sig);
    UNUSED(si);
    UNUSED(unused);

    pthread_cancel(g_sctx->thread_socket_listener);

    shutdown_server(g_sctx);
}

static void usage()
{
    printf("usage: [-cd] [-t <threads>] [-p <port>] [-c <minutes>] [-r <minutes>] [-u <minutes>] <db> <expected.coverage> [checkpoint.file]\n");
    printf("       -t ... worker threads (%d)\n", DEF_ARG_T);
    printf("       -p ... listen port (%d)\n", DEF_ARG_P);
    printf("       -c ... minutes between checkpoints (%d)\n", DEF_ARG_C);
    printf("       -r ... minutes between reports (%d)\n", DEF_ARG_R);
    printf("       -u ... minutes between track updates (%d)\n", DEF_ARG_U);
    printf("       -d ... don't initialize from dust track\n");

    printf("       -e ... (EXPERIMENTAL) no repeat masking <int> bases from the read ends. -1 to disable repeat masking altogether (%d)\n", DEF_ARG_E);
    printf("       -C ... (EXPERIMENTAL) mask contained reads completely (%d)\n", DEF_ARG_CC);
}

uint32 ilog2(uint32 x)
{
    uint32 shift = 0;
    while ( x )
    {
        x >>= 1;
        shift += 1;
    }

    return shift;
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    HITS_TRACK* dust;
    ServerContext ctx;
    int i;
    int init_dust = 1;

    pthread_t* socket_listener = &(ctx.thread_socket_listener);
    pthread_t* reporter = &(ctx.thread_reporter);
    pthread_t* track = &(ctx.thread_track);
    pthread_t* checkpoint = &(ctx.thread_checkpoint);
    pthread_t* worker;
    WorkerContext* wctx;

    // use the environments locale
    setlocale(LC_ALL, "");

    // global pointer to the ServerContext, only used by the signal handlers
    // which otherwise wouldn't have access to it

    g_sctx = &ctx;

    // process arguments

    ctx.port = DEF_ARG_P;
    ctx.worker_threads = DEF_ARG_T;
    ctx.checkpoint = NULL;
    ctx.checkpoint_wait = DEF_ARG_C;
    ctx.report_wait = DEF_ARG_R;
    ctx.track_update_wait = DEF_ARG_U;
    ctx.mask_contained = DEF_ARG_CC;
    ctx.keep_ends = DEF_ARG_E;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "Cde:u:r:t:p:c:")) != -1)
    {
        switch (c)
        {
            case 'e':
                      ctx.keep_ends = atoi(optarg);
                      break;

            case 'C':
                      ctx.mask_contained = 1;
                      break;

            case 'd':
                      init_dust = 0;
                      break;

            case 'c':
                      ctx.checkpoint_wait = atoi(optarg);
                      break;

            case 'r':
                      ctx.report_wait = atoi(optarg);
                      break;

            case 'u':
                      ctx.track_update_wait = atoi(optarg);
                      break;

            case 'p':
                      ctx.port = atoi(optarg);
                      break;

            case 't':
                      ctx.worker_threads = atoi(optarg);
                      break;

            default:
                      usage();
                      exit(1);
        }
    }

    if (opterr || argc - optind < 2)
    {
        usage();
        exit(1);
    }

    char* pathDb = argv[optind++];
    ctx.cov_expected = atoi(argv[optind++]);

    if (argc - optind == 1)
    {
        printf("checkpointing enabled\n");
        ctx.checkpoint = argv[optind++];
    }

    if (ctx.checkpoint_wait < 1)
    {
        fprintf(stderr, "checkpoint interval is too low\n");
        exit(1);
    }

    if (ctx.report_wait < 1)
    {
        fprintf(stderr, "report interval is too low\n");
        exit(1);
    }

    if (ctx.track_update_wait < 1)
    {
        fprintf(stderr, "track update interval is too low\n");
        exit(1);
    }

    if (ctx.worker_threads < 1)
    {
        fprintf(stderr, "number of workers threads must be greater than zero\n");
        exit(1);
    }

    if ( ctx.port == 0 )
    {
        fprintf(stderr, "invalid listen port %d\n", ctx.port);
        exit(1);
    }

    if (ctx.cov_expected < 1 || ctx.cov_expected > MAX_COV)
    {
        fprintf(stderr, "expected coverage outside valid internval [1..%d]\n", MAX_COV);
        exit(1);
    }

    // initialise

    // catch ctrl+c

    struct sigaction sa;

    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = sigint_handler;
    if (sigaction(SIGINT, &sa, NULL) == -1)
    {
        fprintf(stderr, "failed to register SIGINT handler\n");
    }

    if (Open_DB(pathDb, &db))
    {
        fprintf(stderr, "could not open db %s\n", pathDb);
        exit(1);
    }

    // lock mask & shift
    uint32 n_shift = ilog2(db.nreads);
    uint32 g_shift = ilog2(COV_LOCK_GRANULARITY - 1);

    ctx.lock_shift = n_shift - g_shift;
    ctx.lock_mask = (COV_LOCK_GRANULARITY - 1) << ctx.lock_shift;

    // reset db flags
    for (i = 0; i < db.nreads; i++)
    {
        db.reads[i].flags = 0;
    }

    worker = (pthread_t*)malloc( sizeof(pthread_t) * ctx.worker_threads );
    wctx = (WorkerContext*)malloc( sizeof(WorkerContext) * ctx.worker_threads );

    printf("db opened\n");

    if (init_dust)
    {
        dust = track_load(&db, TRACK_DUST);
        printf("dust track loaded\n");
    }
    else
    {
        dust = NULL;
    }

    ctx_init(&ctx, &db, dust);

    if (dust)
    {
        Close_Track(&db, TRACK_DUST);
    }

    printf("initialised\n");

    // debug_add_las(&ctx);

    // start threads

    pthread_create(socket_listener, NULL, socket_listener_thread, &ctx);
    pthread_create(reporter, NULL, reporter_thread, &ctx);
    pthread_create(track, NULL, update_track_thread, &ctx);

    if (ctx.checkpoint)
    {
        pthread_create(checkpoint, NULL, checkpoint_thread, &ctx);
    }

    for (i = 0; i < ctx.worker_threads; i++)
    {
        worker_ctx_init(&ctx, wctx + i);
        wctx[i].id = i;

        pthread_create(worker + i, NULL, worker_thread, wctx + i);
    }

    // wait for threads

    pthread_join(*socket_listener, NULL);
    pthread_join(*reporter, NULL);
    pthread_join(*track, NULL);

    for (i = 0; i < ctx.worker_threads; i++)
    {
        pthread_join(worker[i], NULL);
    }

    for (i = 0; i < ctx.worker_threads; i++)
    {
        worker_ctx_free(wctx+i);
    }

    // clean up

    update_track(&ctx);
    write_mask_tracks(&ctx);

    ctx_free(&ctx);

    Close_DB(&db);

    free(worker);
    free(wctx);

    return 0;
}
