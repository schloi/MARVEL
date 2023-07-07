
#pragma once

#include <inttypes.h>

#include "db/DB.h"
#include "dalign/align.h"

#define UNUSED(x) (void)(x)

int fread_integers(FILE* fileIn, int** out_values, int* out_nvalues);
size_t fread_integer_sets(FILE* fileIn, int64_t** _values, uint64_t** _sets);

int intersect(int ab, int ae, int bb, int be);
void get_trim(HITS_DB* db, HITS_TRACK* trimtrack, int rid, int* b, int* e);

void wrap_write(FILE* fileOut, char* seq, int len, int width);
void revcomp(char* c, int len);
void rev(char* c, int len);

char* format_bytes(unsigned long bytes);

char* bp_format(uint64_t num, int dec);
char* bp_format_alloc(uint64_t num, int dec, int alloc);
uint64_t bp_parse(const char* num);

int trace_valid( Overlap* ovl );

#if !defined( fgetln )

char* fgetln_( FILE* stream, size_t* len );

#define fgetln fgetln_

#endif

