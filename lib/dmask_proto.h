
#pragma once

#include "../db/DB.h"

typedef struct
{
    unsigned char version;    // protocol version
    unsigned char type;       // message type DM_TYPE_xxx
    uint64 length;            // amount of data

    uint64 reserved1;
    uint64 reserved2;
    uint64 reserved3;
    uint64 reserved4;
} DmHeader;

#define DM_VERSION               0x1

#define DM_TYPE_LAS_AVAILABLE    (0x1 << 0)     // c -> s ... contains NULL separated paths as data after header
#define DM_TYPE_REQUEST_TRACK    (0x1 << 1)     // c -> s ... request track. reserved1 = bfirst, reserved2 = nreads
#define DM_TYPE_RESPONSE_TRACK   (0x1 << 2)     // c <- s ... dust track for the requested offset/block (bfirst & nreads)
#define DM_TYPE_SHUTDOWN         (0x1 << 3)     // c -> s ... initiate server shutdown

#define DM_TYPE_LOCK             (0x1 << 4)     // c -> s ... lock (do not update) coverage statistics
#define DM_TYPE_UNLOCK           (0x1 << 5)     // c -> s ... unlock coverage statistics
#define DM_TYPE_INTERVALS        (0x1 << 6)     // c -> s ... dump dusted intervals to text file

#define DM_TYPE_WRITE_TRACK      (0x1 << 7)     // c -> s ... write track
