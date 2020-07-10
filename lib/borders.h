
#pragma once

#include "db/DB.h"
#include "dalign/align.h"

#include "lib/pass.h"

#define EVENT_BEGIN 1
#define EVENT_END   2

typedef struct
{
    int            ovl;

    unsigned short pos;
    unsigned short type;
    unsigned short ovh;
    unsigned short span;
} Event;

typedef struct
{
    Event* peb;
    Event* pee;

    int eb, ee;
    int type;
    int link;
    int done;
} Border;

void find_borders(Border** ppBorder, int* bmax, int* bcur, Event** pEvents, int l, int r,
            float min_density, int min_events, int max_dist);
