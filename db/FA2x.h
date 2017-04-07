

typedef struct
{
  int well;
  int beg;
  int end;
  int len;
  int hasPacbioHeader;
  int seqIDinFasta;

  int maxPrologLen;
  char *prolog;
  int maxSequenceLen;
  char *seq;

  int maxtracks;
  int ntracks;
  int maxName;

  char **trackName;
  int **trackfields;

} pacbio_read;


// used in fasta2DB and fasta2DAM
typedef struct
{
    HITS_DB* db;

    int t_cur;
    int t_max;

    char** t_name;
    track_anno** t_anno;
    track_data** t_data;

    int* t_max_anno;
    int* t_max_data;
    int* t_cur_anno;
    int* t_cur_data;

    //
    int VERBOSE;
    int BEST;                // if true, incorporate only best read from each well of pacbio SMRT cell
    int opt_min_length;      // skip reads, that are shorter than this threshold
    int appendReadsToNewBlock;
    int lastParameterIdx;
    int useFullHqReadsOnly;
    int createTracks;

    FILE* IFILE;

    char * root;
    char *dbname;            //    dbname = full name of db = <pwd>/<root>.db
    char *pwd;

    int ifiles;              //    ifiles = # of .fasta files to add
    int ofiles;              //    ofiles = # of .fasta files already in db
    char **flist;            //    flist  = [0..ifiles+ofiles] list of file names (root only) added to db so far

    FILE* istub;             //    istub  = open db file (if adding) or NULL (if creating)
    FILE* ostub;             //    ostub  = new image of db file (will overwrite old image at end)

    FILE *bases;             //    bases  = .bps file positioned for appending
    FILE *indx;              //    indx   = .idx file positioned for appending

    int64 offset;            //    offset = offset in .bps at which to place next sequence
    int64 boff;              //    boff   = offset in .bps file to truncate to if command fails
    int64 ioff;              //    ioff   = offset in .idx file to truncate to if command fails

    int initialUreads;
    int ureads;              //    ureads = # of reads currently in db
    int64 totlen;            //    total # of bases in new .fasta files
    int maxlen;              //    longest read in new .fasta files
    int64 count[4];

    // used during file parsing
    int rmax;
    char *read;

    pacbio_read* pr1;
    pacbio_read* pr2;

} CreateContext;

int find_track(CreateContext* ctx, const char* name);
void add_to_track(CreateContext* ctx, int track, int64 read, track_data value);
void write_tracks(CreateContext* ctx, char* dbpath);
void free_tracks(CreateContext* ctx);

