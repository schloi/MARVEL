
#pragma once

#define DMASK

#include <sys/socket.h>
#include <resolv.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "dalign/align.h"
#include "db/DB.h"
#include "lib/tracks.h"

#define DMASK_DEFAULT_PORT 12345

typedef struct
{
    int sockfd;
    struct sockaddr_in dest;

    int send_next;
} DynamicMask;

DynamicMask* dm_init(const char* host, uint16 port);
void dm_free(DynamicMask* dm);

void dm_write_track(DynamicMask* dm);
void dm_shutdown(DynamicMask* dm);
void dm_lock(DynamicMask* dm);
void dm_unlock(DynamicMask* dm);
void dm_intervals(DynamicMask* dm);
int dm_done(DynamicMask* dm, char** files);

HITS_TRACK* dm_load_track(HITS_DB* db, DynamicMask* dm, char* trackName);

int dm_send_block_done(DynamicMask* dm, int run,
                       HITS_DB* blocka, char* namea,
                       HITS_DB* blockb, char* nameb);

void dm_send_next(DynamicMask* dm, int run,
                  HITS_DB* blocka, char* namea,
                  HITS_DB* blockb, char* nameb);
