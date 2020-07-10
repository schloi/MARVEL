#include "hashmap.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>

/*

  Fairly fast implementation, except:

  --- remove not implemented

  --- get-or-put isn't optimize/joined yet

  And it's not thorougly tested yet.

*/

#include <stdio.h>             /* for dump */

static void resize_up(Hashmap *h_old);

static unsigned int table_size[] = {
    /* I tried to pick primes which, which some overhead was added,
       would still be under the power-of-two that some mallocs use -
       but I'm not sure the amount of overhead.  Probably just 1,
       since malloc probably wants no more than sizeof(Entry). */
    7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093, 8191,
    16381, 32749, 65521, 131071,
    /* switching to just 2^n-1, because I don't have a list */
    262143, 524287, 1048575, 2097151, 4194303, 8388607,
    16777211, 33554431, 67108863, 134217727, 268435455,
    536870911, 1073741823, 2147483647  /* I think we can
                      stop here... */,
    0};


typedef struct hashmap_entry Entry;
#define KEY(x) ( ((void*)x) + sizeof(Entry) )

/*--- HashPJW ---------------------------------------------------
 *  An adaptation of Peter Weinberger's (PJW) generic hashing
 *  algorithm based on Allen Holub's version. Accepts a pointer
 *  to a datum to be hashed and returns an unsigned integer.
 *     Modified by sandro to include datum_end
 *     Taken from http://www.ddj.com/articles/1996/9604/9604b/9604b.htm?topic=algorithms
 *-------------------------------------------------------------*/
#include <limits.h>
#define BITS_IN_int     ( sizeof(int) * CHAR_BIT )
#define THREE_QUARTERS  ((int) ((BITS_IN_int * 3) / 4))
#define ONE_EIGHTH      ((int) (BITS_IN_int / 8))
#define HIGH_BITS       ( ~((unsigned int)(~0) >> ONE_EIGHTH ))
static unsigned int hash(const char *datum, const char *datum_end)
{
    unsigned int hash_value, i;
    if (datum_end) {
    for ( hash_value = 0; datum<datum_end; ++datum )
    {
        hash_value = ( hash_value << ONE_EIGHTH ) + *datum;
        if (( i = hash_value & HIGH_BITS ) != 0 )
        hash_value =
            ( hash_value ^ ( i >> THREE_QUARTERS )) &
            ~HIGH_BITS;
    }
    } else {
    for ( hash_value = 0; *datum; ++datum )
    {
        hash_value = ( hash_value << ONE_EIGHTH ) + *datum;
        if (( i = hash_value & HIGH_BITS ) != 0 )
        hash_value =
            ( hash_value ^ ( i >> THREE_QUARTERS )) &
            ~HIGH_BITS;
    }
    /* and the extra null value, so we match if working by length */
    hash_value = ( hash_value << ONE_EIGHTH ) + *datum;
    if (( i = hash_value & HIGH_BITS ) != 0 )
        hash_value =
        ( hash_value ^ ( i >> THREE_QUARTERS )) &
        ~HIGH_BITS;
    }

    /* printf("Hash value of %s//%s is %d\n", datum, datum_end, hash_value); */
    return ( hash_value );
}

void hashmap_open(Hashmap *h, unsigned int initial_size)
{
    h->used_slots=0;
    h->table_size_index=0;
    while (table_size[h->table_size_index] < initial_size) {
    h->table_size_index++;
    if (table_size[h->table_size_index] == 0) {
        h->table_size_index--;
        break;
    }
    }
    h->table=calloc(table_size[h->table_size_index], sizeof(Entry *));
}

void hashmap_close(Hashmap *h)
{
    unsigned int i;
    for (i=0; i<table_size[h->table_size_index]; i++) {
    Entry *e = h->table[i];
    Entry *e_next;
    while (e) {
        e_next = e->next_in_bucket;
        free(e);
        e=e_next;
    }
    }
}

void hashmap_dump(Hashmap *h)
{
    unsigned int i;
    for (i=0; i<table_size[h->table_size_index]; i++) {
    Entry *e = h->table[i];

    printf("\nslot %3d: ", i);
    while (e) {
        printf("\"%s\"=>\"%s\" ", (char*) KEY(e), (char*) e->value);
        e=e->next_in_bucket;
    }
    }
    printf("\n");
}

void* hashmap_get(Hashmap *h, void *key, void *key_end)
{
    unsigned int hashval = hash(key, key_end) % table_size[h->table_size_index];
    Entry *e = h->table[hashval];
    size_t keysize = key_end ? (key_end - key) : strlen(key)+1;
    while (e) {
    void* oldkey = (void *) (e+1);
    size_t oldkeysize = e->value - oldkey;
    assert(oldkeysize > 0 && oldkeysize < 1000000);
    if (keysize == oldkeysize && !memcmp(key, oldkey, keysize)) {
        return e->value;
    }
    e=e->next_in_bucket;
    }
    return 0;
}

void hashmap_put(Hashmap *h, void *key, void *key_end,
          void *data, void *data_end)
{
    if (h->used_slots * 2 > table_size[h->table_size_index]) resize_up(h);

    {
    unsigned int hashval = hash(key, key_end) % table_size[h->table_size_index];
    // int x=hash(key, key_end) % table_size[h->table_size_index];
    Entry *e = h->table[hashval];
    size_t keysize = key_end ? (key_end - key) : strlen(key)+1;
    size_t datasize = data_end ? (data_end - data) : strlen(data)+1;
    Entry *n = (Entry *) malloc(sizeof(Entry) + keysize + datasize);
    void *keyspot = ((void*)n) + (sizeof(Entry));
    assert(n);

    n->next_in_bucket = e;
    h->table[hashval] = n;
    memcpy(keyspot, key, keysize);
    n->value = keyspot + keysize;
    memcpy(n->value, data, datasize);
    h->used_slots++;
    }
}

void* hashmap_get_or_put(Hashmap *h,
                void *key, void *key_end,
                void *data, void *data_end)
{
    /* we can make it more efficient some day.  :-) */

    void *value = hashmap_get(h, key, key_end);
    if (value) return value;
    hashmap_put(h, key, key_end, data, data_end);
    return 0;
}

/*
   ...  could be generalized to resize up/down if we ever need down.
*/
void resize_up(Hashmap *h_old)
{
    Hashmap h;
    unsigned int i;
    if (table_size[h_old->table_size_index + 1] == 0) return;

    h.used_slots=0;
    h.table_size_index=h_old->table_size_index + 1;
    h.table=calloc(table_size[h.table_size_index], sizeof(Entry *));

    /*
      traverse the old hash table entries and re-link them into their
      new places in the new table.
    */
    for (i=0; i<table_size[h_old->table_size_index]; i++) {
    Entry *e = h_old->table[i];
    Entry *next_e;
    while (e) {
        unsigned int hashval = hash(KEY(e), e->value)
        % table_size[h.table_size_index];
        next_e = e->next_in_bucket;
        e->next_in_bucket = h.table[hashval];
        h.table[hashval] = e;
        e=next_e;
    }
    }

    /*
      overwrite the old with the new
    */
    free(h_old->table);
    h_old->table = h.table;
    h_old->table_size_index++;
}
