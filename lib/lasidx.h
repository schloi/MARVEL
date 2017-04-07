
#pragma once

#include <stdio.h>
#include <db/DB.h>

typedef off_t lasidx;

lasidx* lasidx_create(HITS_DB* db, const char* pathLas);
lasidx* lasidx_load(HITS_DB* db, const char* pathLas, int create);

void lasidx_close(lasidx* idx);
