/* bitarr.h */

/*
   Prototypes for bitarr.c

   IMPORTANT NOTE: A version of this code appeared in
                   Dr. Dobb's Journal issue #233 (August 1995)
                   Volume 20 Issue 8
                   in an article entitled `Implementing Bit Vectors in C'
                   by James Blustein
                   Pages 42, 44, 46 (article) and pages 96, 98-100 (code)
                   The code is (c) copyright 1995 by Miller Freeman, Inc.

   See "bitarr.c" for further details.
*/

#pragma once

elem_t ba_init(void);
bit *ba_new(const elem_t nelems);
void ba_copy(bit dst[], const bit src[], const elem_t size);
void ba_assign(bit arr[], elem_t elem, const bool value);
void ba_assign_range(bit arr[], elem_t elem_beg, elem_t elem_end, const bool value);

bool ba_value(const bit arr[], const elem_t elem);
void ba_toggle(bit arr[], const elem_t elem);
void ba_all_assign(bit arr[], const elem_t lsize, const bool value);
bit *ba_ul2b(unsigned long num, bit *arr, elem_t *size);
unsigned long ba_count(const bit arr[], const elem_t size);
bool ba_intersection(bit first[], bit second[], bit * result[], const elem_t size_first, const elem_t size_second);
bool ba_union(bit first[], bit second[], bit * result[],  const elem_t size_first, const elem_t size_second);
bool ba_diff(bit first[], bit second[], bit * result[], const elem_t size_first, const elem_t size_second);
void ba_complement(bit arr[], const elem_t lsize);
unsigned long ba_dotprod(const bit first[], const bit second[], const elem_t size_first, const elem_t size_second);
char * ba_b2str(const bit arr[], const elem_t size, char * dest);
bool ba_print(const bit arr[], const elem_t size, FILE * dest);

size_t ba_bufsize(const elem_t nelems);
