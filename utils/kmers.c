
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#include <sys/param.h>

#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"
#include "lib/lasidx.h"

#include "db/DB.h"
#include "dalign/align.h"

#define DEF_ARG_E 12
#define DEF_ARG_M  0

#define BINS_NORMAL 0
#define BINS_NOISE  1
#define BINS_REPEAT 2

#define VERBOSE
#undef DEBUG
#undef DEBUG_CLASSIFY

typedef uint64 KMER;

typedef unsigned char KMER_COUNT;
#define KMER_COUNT_MAX 255

// typedef uint16 KMER_COUNT;
// #define KMER_COUNT_MAX 65535

typedef struct
{
    HITS_DB* db;
    char* pathDb;

    // k-mer k
    int k;

    uint16 error;
    uint16 coverage;
    uint16 genomesize;

    uint16 merge;

    uint16 noise;
    uint16 over;
    double over_expected;
    double noise_expected;

    // window size
    int wnd;

    // kmer occurance counts. capped at KMER_COUNT_MAX
    KMER_COUNT* kcounts;
    uint64 kcount;

    // kmer occurance histogram
    uint64* histo_counts;
    uint64 histo_sum;

    // expected window sum
    double ews;
    double ews_nocap_sigma;
    double ews_nocap_mu;
} KmersContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

void print_bits(size_t const size, void const * const ptr)
{
    unsigned char* b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i = size - 1; i >= 0; i--)
    {
        for (j = 7; j >= 0; j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }

        printf(" ");
    }
    puts("");
}

static void track_add(track_anno* anno, track_data** _data, track_anno* _dmax, track_anno* _dcur, int merge, int rid, int beg, int end)
{
    track_data* data = *_data;
    track_anno dmax = *_dmax;
    track_anno dcur = *_dcur;

    if ( dcur + 2 >= dmax )
    {
        dmax = dmax * 1.2 + 1000;
        data = realloc( data, sizeof(track_data) * dmax );
    }

    if ( anno[rid] > 0 && data[dcur - 1] + merge >= beg )
    {
        data[ dcur - 1 ] = end;
    }
    else
    {
        // printf("%d @ %5d..%5d\n", read, i, i + j);
        anno[ rid ] += sizeof(track_data) * 2;

        data[ dcur ] = beg;
        data[ dcur + 1 ] = end;
        dcur += 2;
    }

    *_data = data;
    *_dmax = dmax;
    *_dcur = dcur;
}

static int kmer_count_noise(KmersContext* kctx)
{
    // double prob_kmer_conserved = exp( -1.0 * (kctx->error / 100.0) * kctx->k );
    double err = kctx->error / 100.0;
    int k = kctx->k;
    double prob_kmer_conserved = pow(1-err, k+1) + (1.0/k) * ( pow(1-err, k-1) - pow(1-err, 2*k - 1) );

    double exp_kmer_count = MAX( 1.0, kctx->genomesize * 1000.0 * 1000.0 * 0.5 / pow(4, kctx->k) );

    return floorl( exp_kmer_count * kctx->coverage * prob_kmer_conserved );
}

static void compute_cutoffs(KmersContext* kctx)
{
#ifdef VERBOSE
    printf("computing k-mer cutoffs\n");
#endif

    uint64 sum = 0;
    int i = KMER_COUNT_MAX;
    uint64 histo_sum = kctx->histo_sum;
    uint64* histo_counts = kctx->histo_counts;
    uint64 count = 0;

    while ( (double)(sum) / histo_sum < 0.01 )
    {
        count += histo_counts[i] * i;
        sum += histo_counts[i];
        i -= 1;
    }

    double exp = 1.0 * count / ( kctx->genomesize * 1000.0 * 1000.0 * kctx->coverage * 0.5 );

    int wnd = kctx->wnd - kctx->k + 1;
    printf("cutoff repeat @ %d sum %lld count %lld -> %.2f %.2f\n", i, sum, count, exp, exp * wnd);

    kctx->over_expected = exp;
    kctx->over = i;

    kctx->noise = kmer_count_noise(kctx);

    i = kctx->noise;
    sum = 0;
    count = 0;
    while ( i != 0 )
    {
        count += histo_counts[i] * i;
        sum += histo_counts[i];
        i -= 1;
    }

    exp = 1.0 * count / ( kctx->genomesize * 1000.0 * 1000.0 * kctx->coverage * 0.5 );

    printf("cutoff noise @ %d sum %lld count %lld -> %.2f %.2f\n", kctx->noise, sum, count, exp, exp * wnd);

    kctx->noise_expected = exp;
}

static int classify_bins(KmersContext* kctx, uint64* bins, uint64* _nnoise, uint64* _nnormal, uint64* _nover)
{
    KMER_COUNT noise = kctx->noise;
    KMER_COUNT over = kctx->over;

    uint64 nnoise = 0;
    uint64 nnormal = 0;
    uint64 nover = 0;

    uint64 i;
    for ( i = 0 ; i < noise ; i++ )
    {
        nnoise += bins[i];
    }

    for ( ; i < over ; i++ )
    {
        nnormal += bins[i];
    }

    for ( ; i <= KMER_COUNT_MAX ; i++ )
    {
        nover += bins[i];
    }

    *_nnoise = nnoise;
    *_nnormal = nnormal;
    *_nover = nover;

    int wnd = kctx->wnd - kctx->k + 1;

    double over_expected = kctx->over_expected * ( wnd - nnoise );
    double noise_expected = kctx->noise_expected * wnd;

    if ( nover > over_expected )
    {
        return BINS_REPEAT;
    }
    else if ( nnoise > noise_expected )
    {
        return BINS_NOISE;
    }
    else
    {
        return BINS_NORMAL;
    }
}

static void create_track(KmersContext* kctx, int maxreads)
{
    printf("creating k-mer repeat track\n");

    char* pathDb = kctx->pathDb;
    KMER_COUNT* kcounts = kctx->kcounts;
    uint64 kcount = kctx->kcount;
    int wnd = kctx->wnd;
    int k = kctx->k;
    int merge = kctx->merge;

    uint64 kmask = kcount - 1;
    HITS_DB db;

    char fname[PATH_MAX];
    sprintf(fname, "bstats.%d.txt", kctx->k);
    FILE* fileOutBase = fopen(fname, "w");
    sprintf(fname, "wstats.%d.txt", kctx->k);
    FILE* fileOutWnd = fopen(fname, "w");

    Open_DB(pathDb, &db);

    KMER* readmers = malloc( sizeof(KMER) * db.maxlen );

    track_anno* anno = malloc(sizeof(track_anno) * (DB_NREADS(&db) + 1));
    track_anno dmax = DB_NREADS(&db) * 2;
    track_anno dcur = 0;
    track_data* data = malloc(sizeof(track_data) * dmax);
    bzero(anno, sizeof(track_anno) * (DB_NREADS(&db) + 1));

    track_anno* noise_anno = malloc(sizeof(track_anno) * (DB_NREADS(&db) + 1));
    track_anno noise_dmax = DB_NREADS(&db) * 2;
    track_anno noise_dcur = 0;
    track_data* noise_data = malloc(sizeof(track_data) * noise_dmax);
    bzero(noise_anno, sizeof(track_anno) * (DB_NREADS(&db) + 1));

    uint64* bins = malloc(sizeof(uint64) * (KMER_COUNT_MAX + 1));
    uint64 nnoise, nnormal, nover;

    int nblocks = DB_Blocks(pathDb);
    int block;
    for (block = 1; block <= nblocks; block++)
    {
        printf("block %d/%d\n", block, nblocks);

        // if ( block != 25 ) continue;

        HITS_DB dbb;

        Open_DB_Block(pathDb, &dbb, block);

        Read_All_Sequences(&dbb, 0);

        int read;

        for (read = 0; read < dbb.nreads; read++)
        {
            int idx = 0;
            KMER kmer = 0;
            char* bases = dbb.bases + dbb.reads[read].boff;
            int rlen = dbb.reads[read].rlen;
            int rid = dbb.ufirst + read;

            bzero(readmers, db.maxlen);

            while (idx < k - 1)
            {
                kmer = (kmer << 2) + bases[idx];
                idx++;
            }

            while (idx < rlen)
            {
                kmer = ( (kmer << 2) & kmask ) + bases[idx];
                readmers[idx] = kmer;

                idx++;
            }

            int i;

            if ( read < maxreads )
            {
                for (i = k - 1; i < rlen; i++)
                {
                    fprintf(fileOutBase, "%d %d %d\n", rid, i, kcounts[ readmers[i] ]);
                }
            }

            for (i = 0; i < rlen - wnd; i++)
            {
              bzero(bins, sizeof(uint64) * (KMER_COUNT_MAX + 1));

              int j;
              int sum = 0;
              for (j = k - 1; j < wnd; j++) {
                kmer = readmers[i + j];
                sum += kcounts[kmer];

                bins[kcounts[kmer]] += 1;
              }

#ifdef DEBUG_CLASSIFY
                printf("%d..%d -> ", i, i + j);
#endif

                switch ( classify_bins(kctx, bins, &nnoise, &nnormal, &nover) )
                {
                    case BINS_REPEAT:
                        track_add(anno, &data, &dmax, &dcur, merge, rid, i, i + j);
                        break;

                    case BINS_NOISE:
                        track_add(noise_anno, &noise_data, &noise_dmax, &noise_dcur, merge, rid, i, i + j);
                        break;

                }

                if (read < maxreads)
                {
                    fprintf(fileOutWnd, "%d %d %d %lld %lld %lld\n", rid, i, i + wnd - 1, nnoise, nnormal, nover);
                }

                i += wnd;
            }

            switch ( classify_bins(kctx, bins, &nnoise, &nnormal, &nover) )
            {
                case BINS_REPEAT:
                    track_add(anno, &data, &dmax, &dcur, merge, rid, i, rlen);
                    break;

                case BINS_NOISE:
                    track_add(noise_anno, &noise_data, &noise_dmax, &noise_dcur, merge, rid, i, rlen);
                    break;
            }

            //break;
        }

        Close_DB(&dbb);
        break;
    }

    int i;
    track_anno coff, off;
    track_anno noise_off;

    noise_off = off = 0;
    for (i = 0; i <= DB_NREADS(&db); i++)
    {
        coff = anno[i];
        anno[i] = off;
        off += coff;

        coff = noise_anno[i];
        noise_anno[i] = noise_off;
        noise_off += coff;
    }

    track_write(&db, TRACK_KREPEATS, 0, anno, data, dcur);
    track_write(&db, TRACK_KNOISE, 0, noise_anno, noise_data, noise_dcur);

    free(noise_anno);
    free(noise_data);

    free(anno);
    free(data);
    free(readmers);

    fclose(fileOutBase);
    fclose(fileOutWnd);

    Close_DB(&db);
}

/*
static void print_histograms(KmersContext* kctx)
{
    uint64 total, count;
    total = count = 0;

    uint64* bins = kctx->histo_counts;

    uint64 i;
    for ( i = 0 ; i <= KMER_COUNT_MAX ; i++ )
    {
        printf("%lld %lld\n", i, bins[i]);

        if (i > 0)
        {
            total += i * bins[i];
            count += bins[i];
        }
    }

    printf("avg    %.2f (with %d)\n", (double)(total) / count, KMER_COUNT_MAX);
    total -= KMER_COUNT_MAX * bins[KMER_COUNT_MAX];
    count -= bins[KMER_COUNT_MAX];
    printf("avg    %.2f\n", (double)(total) / count);
}
*/

static int load_kcounts(KmersContext* kctx)
{
    int k = kctx->k;
    char* pathDb = kctx->pathDb;
    HITS_DB* db = kctx->db;

    uint64 kcount = kctx->kcount = (1llu << (2*k));
    uint64 kmask = kcount - 1;

    KMER_COUNT* kcounts = kctx->kcounts = malloc( sizeof(KMER_COUNT) * kcount );
    bzero(kcounts, sizeof(KMER_COUNT) * kcount);

    if (kcounts == NULL)
    {
        fprintf(stderr, "failed to allocate kcounts\n");
        return 0;
    }

    int read = 0;

    char pcPathKmers[PATH_MAX];
    sprintf(pcPathKmers, "%s.%dmers", pathDb, k);
    FILE* fileKmers = fopen(pcPathKmers, "r");

    if ( fileKmers )
    {
#ifdef VERBOSE
        printf("loading k-mer counts\n");
#endif

        if ( fread(kcounts, sizeof(KMER_COUNT), kcount, fileKmers) != kcount )
        {
            fprintf(stderr, "ERROR: failed to load kmer counts\n");
            exit(1);
        }

        fclose(fileKmers);
    }
    else
    {
#ifdef VERBOSE
        printf("computing k-mer counts\n");
#endif

        int nblocks = DB_Blocks(pathDb);

        int block;
        for (block = 1; block <= nblocks; block++)
        {
            fprintf(stderr, "%s %d/%d\n", pathDb, block, nblocks);

            Open_DB_Block(pathDb, db, block);

            // pass

            Read_All_Sequences(db, 0);

            for (read = 0; read < db->nreads; read++)
            {
                char* bases = db->bases + db->reads[read].boff;
                int rlen = db->reads[read].rlen;

                int idx = 0;
                KMER kmer = 0;
                while (idx < k - 1)
                {
                    kmer = (kmer << 2) + bases[idx];
                    idx++;
                }

                while (idx < rlen)
                {
                    kmer = ( (kmer << 2) + bases[idx] ) & kmask;
                    idx++;

                    if (kcounts[kmer] < KMER_COUNT_MAX)
                    {
                        kcounts[kmer] += 1;
                    }
                }
            }

            Close_DB(db);
        }

        FILE* fileOut = fopen(pcPathKmers, "w");
        fwrite(kcounts, sizeof(KMER_COUNT), kcount, fileOut);
        fclose(fileOut);
    }

    // create k-mer occurance count histogram

#ifdef VERBOSE
    printf("computing k-mer histogram\n");
#endif

    uint64* bins = kctx->histo_counts = malloc( sizeof(uint64) * (KMER_COUNT_MAX + 1) );
    bzero(bins, sizeof(uint64) * (KMER_COUNT_MAX + 1));

    uint64 i;
    for ( i = 0 ; i < kcount ; i++ )
    {
        bins[ kcounts[i] ] += 1;
    }

    uint64 sum = 0;
    for ( i = 1 ; i < KMER_COUNT_MAX ; i++ )
    {
        sum += bins[i];
    }

    kctx->histo_sum = sum;

    return 1;
}

static void usage()
{
    printf("usage:   [-e <error>] [-m <merge>] <db> <k> <coverage> <genome.mb>\n");
    printf("options: -e sequence error rate (default %d)\n", DEF_ARG_E);
    printf("         -m merge distance for repeat intervals (default k)\n");
    printf("         -r only report computed cutoffs\n");
}

int main(int argc, char* argv[])
{
    KmersContext kctx;
    HITS_DB db;
    int c;
    int do_work = 1;

    bzero(&kctx, sizeof(KmersContext));
    kctx.db = &db;
    kctx.error = DEF_ARG_E;

    // TODO -- hardcoded
    kctx.wnd = 250;

    kctx.merge = DEF_ARG_M;

    // process arguments

    opterr = 0;

    while ((c = getopt(argc, argv, "re:m:")) != -1)
    {
        switch (c)
        {
            case 'e':
                kctx.error = atoi(optarg);
                break;

            case 'm':
                kctx.merge = atoi(optarg);
                break;

            case 'r':
                do_work = 0;
                break;

            default:
                printf("Unknow option: %s\n", argv[optind - 1]);
                usage();
                exit(1);
        }
    }

    if (argc - optind < 4)
    {
        usage();
        exit(1);
    }

    kctx.pathDb = argv[optind++];
    kctx.k = atoi( argv[optind++] );
    kctx.coverage = atoi( argv[optind++] );
    kctx.genomesize = atoi( argv[optind++] );

    if (kctx.merge == 0 && kctx.merge < kctx.k)
    {
        kctx.merge = kctx.k;
    }

#ifdef VERBOSE
    printf("k = %d  coverage = %d  genome size = %dMB  merge = %d\n", kctx.k, kctx.coverage, kctx.genomesize, kctx.merge);
#endif

    if (!load_kcounts(&kctx))
    {
        exit(1);
    }

    // print_sequence(&kctx, 200);
    // return 1;

    compute_cutoffs(&kctx);

    // print_histograms(&kctx);
    // return 1;

#ifdef VERBOSE
    printf("0 < noise < %d < normal < %d < overrepresented < %d\n", kctx.noise, kctx.over, KMER_COUNT_MAX);
#endif

    if ( do_work )
    {
        create_track(&kctx, 200);
    }

    // cleanup

    free(kctx.kcounts);
    free(kctx.histo_counts);

    return 0;
}
