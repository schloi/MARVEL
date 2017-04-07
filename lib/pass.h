
#pragma once

#include <sys/types.h>

#include "db/DB.h"
#include "dalign/align.h"

#define DB_READ_FLAGS(db, rid) ( (db)->reads[ (rid) ].flags )
#define DB_READ_LEN(db, rid)   ( (db)->reads[ (rid) ].rlen )
#define READ_LEN(r)            ( (r)->rlen )
#define DB_READ_MAXLEN(db)     ( (db)->maxlen )
#define DB_NREADS(db)          ( (db)->nreads )
//#define DB_READ_ID(db, rid)    ( (db)->reads[ (rid) ].id )

#define TBYTES(twidth)          ( (twidth) <= TRACE_XOVR ? sizeof(uint8) : sizeof(uint16) )

#define OVERLAP_IO_SIZE     (sizeof(Overlap) - sizeof(void*))

typedef uint64           ovl_header_novl;
typedef int              ovl_header_twidth;
typedef uint16           ovl_trace;

typedef struct
{
    // overlaps and trace

    FILE* fileOvlIn;
    FILE* fileOvlOut;

    ovl_header_twidth twidth;
    ovl_header_novl novl;               // number of overlaps reads in
    ovl_header_novl novl_out;           // written out

    ovl_header_novl novl_out_discarded;     // written out with OVL_DISCARD flag

    long sizeOvlIn;
    long progress_tick;
    long progress_nexttick;

    off_t off_start;
    off_t off_end;

    int tmax;
    int tcur;
    ovl_trace* trace;

    size_t tbytes;

    // user supplied data

    void* data;

    // pass settings

    int split_b;
    int load_trace;
    int unpack_trace;

    int write_overlaps;
    int purge_discarded;

    int progress;

} PassContext;

typedef int (*pass_handler)(void*, Overlap*, int);

PassContext* pass_init(FILE* fileOvlIn, FILE* fileOvlOut);

void pass(PassContext* ctx, pass_handler handler);
void pass_free(PassContext* ctx);

void read_unpacked_trace(FILE* fileOvl, Overlap* ovl, size_t tbytes);

int ovl_header_read(FILE* fileOvl, ovl_header_novl* novl, ovl_header_twidth* twidth);
void ovl_header_write(FILE* fileOvl, ovl_header_novl novl, ovl_header_twidth twidth);

