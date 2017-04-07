
/* bitarr.c */

/**************************************************************************

  Bit Vectors in C

  FILENAME: bitarr.c
  LANGUAGE: ANSI C
  REQUIRES: "types.h" (see below for minimal contents)

  AUTHOR:   James Blustein <jamie@csd.uwo.ca>
  CREATED:  22 March 1995
  MODIFIED: 20 July 1996
            (see http://www.csd.uwo.ca/~jamie/BitVectors/changes.html
             for a record of changes).

  IMPORTANT NOTE: A version of this code appeared in
                  Dr. Dobb's Journal issue #233 (August 1995)
                  Volume 20 Issue 8
                  in an article entitled `Implementing Bit Vectors in C'
                  by James Blustein
                  Pages 42, 44, 46 (article) and pages 96, 98-100 (code)
                  The code is (c) copyright 1995 by Miller Freeman, Inc.

  DESCRIPTION: Functions to create and manipulate arrays of bits, i.e. `bit
     vectors'.  Functions to: dynamically create arrays, access (read and
     write) elements; convert from numbers to bit vectors and bit vectors
     to strings.  Additional mathematical functions (union, intersection,
     complement, number of set bits) are provided that are more efficient
     than naive implementations.
               The module was designed to be robust enough to work with
     machines of different word sizes.  Only a couple of minor changes are
     required to change it from using unsigned char for `bits' to another
     integer type.  See ba_init() and the definition of BITS_SZ for details.
               Only minimal optimization has been attempted.

               It is the caller's responsibility to know the size of the
     bit vector.  One way to keep track of the size is to wrap the bit
     vector in a data structure like the one below.  Note that the // is
     used to mark a single line comment.  Only the first two items are
     necessary -- the others are included for illustration only.

      typedef struct {
                elem_t  size;     // how many items in selected
                bits *  selected; // bit vector recording which
                                  // elements are selected
                elem_t  max;      // maximum possible size
                char ** name;     // array of names of items
                elem_t  max_len;  // maximum possible length of a name
                char *  title;    // what data is represented
                                  // by this struct?
            } chose_t;

  TYPEDEF NAMES
            "types.h" must include definitions of the following 4 types
                      bool   = a Boolean type (0 == FALSE, !0 == TRUE)
                      string = char *
                      elem_t = a number (used as a count, i.e. never < 0,
                               throughout).
                      bit    = an unsigned integer
                               + If this is not unsigned char then the #define
                                 of BITS_SZ as CHAR_BIT should be changed.
                               + If this is not internally represented by 8
                                 bits then the lookup table in ba_count() must
                                 be replaced.
                               + SEE NOTE dated 13 August 1996 in changes.html
                                 for a better solution.

 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "types.h"
#include "bitarr.h" /* exported prototypes */
#ifdef NEED_FPUTS_PROTO
  int fputs(char *s, FILE *stream);
#endif

typedef struct {elem_t size; bit *vector;} BitVector;

static void first_is_biggest(BitVector bv[2], unsigned *, unsigned *);


/*                                                                  *\
   ----------------------------------------------------------------
                               	Macros
   ----------------------------------------------------------------
\*                                                                  */

/*
 macro NELEM()

   The number of elements, nelem, in an array of N bits can be
 computed using the formula:

   if (0 == (N % BITS_SZ))
      nelem = N/BITS_SZ
   else
      nelem = N/BITS_SZ + 1

   This can be represented in any of these ways:
     nelem = N/(BITS_SZ)  +  1 - (0 == (N %(BITS_SZ)))
     nelem = N/(BITS_SZ)  +  !(0 == (N %(BITS_SZ)))
     nelem = N/(BITS_SZ)  +  (0 != (N %(BITS_SZ)))

   The macro NELEM used this last form until Frans F.J. Faase
   <Frans_LiXia at wxs dot nl> suggested the form below (see changes.html).
*/
#define NELEM(N,ELEMPER) ((N + (ELEMPER) - 1) / (ELEMPER))

/*
 macro CANONIZE()
  Array is an array of `NumInts' type `bit' representing `NumElem' bits
  Forces `Array' into canonical form, i.e. all unused bits are set to 0
*/
#define CANONIZE(Array,NumInts,NumElem)                                 \
   (Array)[NumInts - 1] &= (bit)~0 >> (BITS_SZ - ((NumElem % BITS_SZ)   \
                                                  ? (NumElem % BITS_SZ) \
		                                  : BITS_SZ));

/*
 BITS_SZ

   BITS_SZ is the number of bits in a single `bits' type.
*/


/* Definition of BITS_SZ */
#ifdef CHAR_BIT
   /* assumes typedef unsigned char bits */
   #define BITS_SZ (CHAR_BIT)
      /** SEE 13 August 1996 note in changes.html for suggested improvement **/
#else
   static elem_t bits_size(void);

   elem_t BITS_SZ = 0;  /* until it is initialized by ba_init() */

   static elem_t bits_size(void) {

       /*
          Adapted from the wordlength() function on page 54 (Exercise
          2-8) of _The C Answer Book_ (2nd ed.) by Clovis L. Tondo
          and Scott E. Gimpel.  Prentice-Hall, Inc., 1989.
       */

      elem_t i;
      bit    v = (bit)~0;

      for (i=1; (v = v >> 1) > 0; i++)
         ; /* EMPTY */
      return (i);
   }
#endif


/*                                                                  *\
   ----------------------------------------------------------------
                  Initialization and Creation Code
   ----------------------------------------------------------------
\*                                                                  */

elem_t ba_init(void)
{
/*
   ba_init()

   PRE:  Must be called before use of any other ba_ functions.  Should
         only be called once.
   POST: Returns the number of values that can be stored in one
         variable of type `bit'.  If <limits.h> does not define
         `CHAR_BIT' then the module global variable `BITS_SZ' has been
         set to the appropriate value.
*/

   #ifndef BITS_SZ
      if (!BITS_SZ) {
         BITS_SZ = bits_size();
      }
   #endif
   return (BITS_SZ);
} /* ba_init() */


bit *ba_new(const elem_t nelems)
{
/*
   ba_new()

   PURPOSE: dynamically allocate space for an array of `nelems' bits
         and initalize the bits to all be zero.
   PRE:  nelems is the number of Boolean values required in an array
   POST: either a pointer to an initialized (all zero) array of bit
      OR
         space was not available and NULL was returned
   NOTE: calloc() guarantees that the space has been initialized to 0.

   Used by: ba_ul2b(), ba_intersection() and ba_union().
*/

   size_t howmany =  NELEM(nelems,(BITS_SZ));

   return ((bit *)calloc(howmany, sizeof(bit)));
} /* ba_new() */

size_t ba_bufsize(const elem_t nelems)
{
    return NELEM(nelems, (BITS_SZ)) * sizeof(bit);
}

void ba_copy(      bit    dst[],
             const bit    src[],
             const elem_t size)
{
/*
   ba_copy()

   PRE:  `dst' has been initialized to hold `size' elements.  `src'
         is the array of bit to be copied to `dst'.
   POST: `dst' is identical to the first `size' bits of `src'.
         `src' is unchanged.
   Used by: ba_union()
*/
            elem_t nelem  = NELEM(size,(BITS_SZ));
   register elem_t i;

   for (i=0; i < nelem; i++) {
      dst[i] = src[i];
   }
} /* ba_copy() */


/*                                                                 *\
   ---------------------------------------------------------------
                   Assigning and Retrieving Values
   ---------------------------------------------------------------
\*                                                                 */
void ba_assign(      bit    arr[],
                     elem_t elem,
               const bool   value)
{
/*
   ba_assign()

   PURPOSE: set or clear the bit in position `elem' of the array
         `arr'
   PRE:     arr[elem] is to be set (assigned to 1) if value is TRUE,
            otherwise it is to be cleared (assigned to 0).
   POST:    PRE fulfilled.  All other bits unchanged.
   SEE ALSO: ba_all_assign()
   Used by:  ba_ul2b()
*/

   if (value) {
      arr[elem / BITS_SZ] |= (1 << (elem % BITS_SZ));
   } else {
      arr[elem / BITS_SZ] &= ~(1 << (elem % BITS_SZ));
   }
} /* ba_assign() */


void ba_assign_range(bit arr[], elem_t elem_beg, elem_t elem_end, const bool value)
{
    if (value)
    {
        do
        {
            arr[elem_beg / BITS_SZ] |= (1 << (elem_beg % BITS_SZ));
            elem_beg++;
        } while ( elem_beg % BITS_SZ && elem_beg <= elem_end);

        while ( (elem_end + 1) % BITS_SZ && elem_end >= elem_beg)
        {
            arr[elem_end / BITS_SZ] |= (1 << (elem_end % BITS_SZ));
            elem_end--;
        }

        while (elem_beg < elem_end)
        {
            arr[elem_beg / BITS_SZ] = ~0;
            elem_beg += BITS_SZ;
        }
    }
    else
    {
        do
        {
            arr[elem_beg / BITS_SZ] &= ~(1 << (elem_beg % BITS_SZ));
            elem_beg++;
        } while ( elem_beg % BITS_SZ && elem_beg <= elem_end);

        while ( (elem_end + 1) % BITS_SZ && elem_end >= elem_beg)
        {
            arr[elem_end / BITS_SZ] &= ~(1 << (elem_end % BITS_SZ));
            elem_end--;
        }
                
        while (elem_beg < elem_end)
        {
            arr[elem_beg / BITS_SZ] = 0;
            elem_beg += BITS_SZ;
        }
    }

}

bool ba_value(const bit    arr[],
              const elem_t elem)
{
/*
   ba_value()

   PRE:  arr must have at least elem elements
   POST: The value of the `elem'th element of arr has been returned
         (as though `arr' was just a 1-dimensional array of bit)
   Used by: ba_b2str() and ba_count()
*/

   return( (arr[elem / BITS_SZ] & (1 << (elem % BITS_SZ))) ?TRUE :FALSE );
} /* ba_value() */



void ba_toggle(      bit     arr[],
               const elem_t  elem)
{
/*
   ba_toggle()

   PRE:  arr must have at least elem elements
   POST: The value of the `elem'th element of arr has been flipped,
         i.e. if it was 1 it is 0; if it was 0 it is 1.
   SEE ALSO: ba_complement()
*/

 arr[elem / BITS_SZ] ^= (1 << (elem % BITS_SZ));
} /* ba_toggle() */



void ba_all_assign(      bit    arr[],
                   const elem_t size,
                   const bool   value)
{
/*
   ba_all_assign()

   PRE:  arr has been initialized to have *exactly* size elements.
   POST: All `size' elements of arr have been set to `value'.
         The array is in canonical form, i.e. trailing elements are
         all 0.
   NOTE: The array allocated by ba_new() has all elements 0 and is
         therefore in canonical form.
   SEE ALSO: ba_assign()
   Used by: ba_ul2b()
*/
            elem_t nelem  = NELEM(size,(BITS_SZ));
            bit    setval = (value) ?~0 :0;
   register elem_t i;

   for (i=0; i < nelem; i++) {
      arr[i] = setval;
   }
   /* force canonical form */
   CANONIZE(arr,nelem,size);
} /* ba_all_assign() */


/*                                                                  *\
   ----------------------------------------------------------------
                         Conversion Routines
   ----------------------------------------------------------------
\*                                                                  */

bit * ba_ul2b(unsigned long num,
              bit *         arr,
              elem_t *      size)
{
/*
   ba_ul2b()

   PRE:  Either
           `arr' points to space allocated to hold enough `bit's to
         represent `num' (namely the ceiling of the base 2 logarithm
         of `num').  `size' points to the number of bit to use.
       OR
           `arr' is NULL and the caller is requesting that enough
         space be allocated to hold the representation before the
         translation is made. `size' points to space allocated to
         hold the count of the number of bit needed for the
         conversion (enough for MAXLONG).
   POST: A pointer to a right-aligned array of bits representing the
         unsigned value num has been returned and `size' points to
         the number of `bit's needed to hold the value.
       OR
         the request to allocate space for such an array could not
         be granted

   NOTES: - The first argument is unsigned.
          - It is bad to pass a `size' that is too small to hold the
            bit array representation of `num' [K&R II, p.100].
          - Should the `size' be the maximum size (if size > 0) even
            if more bits are needed?  The user can always use a filter
            composed of all 1s (see ba_all_assign()) intersected with
            result (see ba_intersection()).
*/

   register elem_t i;

   if (NULL != arr) {
      ba_all_assign(arr, *size, 0);
   } else {
      *size = NELEM(sizeof(num),sizeof(bit));
      *size *= BITS_SZ;
      if (NULL == (arr = ba_new(*size))) {
         return (arr);
      }
   }

   /* usual base conversion algorithm */
   for (i=0; num; num >>= 1, i++) {
      ba_assign(arr, (*size - i - 1), (1 == (num & 01)));
   }
   return (arr);
} /* ba_ul2b() */



char * ba_b2str(const bit    arr[],
                const elem_t size,
                      char * dest)
{
/*
   ba_b2str()

   PRE: `arr' is a bit array with at least `size' elements.  Either
         `dest' points to enough allocated space to hold `size' + 1
         characters or `dest' is NULL and such space is to be
         dynamically allocated.
   POST: Either `dest' points to a null-terminated string that
         contains a character representation of the first `size'
         elements of the bit array `arr';
      OR
         `dest' is NULL and a request to dynamically allocate memory
         for a string to hold a character representation of `arr' was
         not be granted.
   Used by: ba_print()
*/
   register elem_t i;

   if ((NULL != dest) || \
       (NULL != (dest = (char *)malloc(size + 1)))) {

      for (i=0; i < size; i++) {
         dest[i] = (ba_value(arr,i) ?'1' :'0');
      }
      dest[size] = '\0';
   }
   return (dest);
} /* ba_b2str() */


/*                                                                  *\
   ----------------------------------------------------------------
                      Mathematical Applications
   ----------------------------------------------------------------
\*                                                                  */

unsigned long ba_count(const bit    arr[],
                       const elem_t size)
{
/*
   ba_count()

   PRE:  `arr' is an allocated bit array with at least `size'
         elements
   POST: The number of 1 bits in the first `size' elements of `arr'
         have been returned.
   NOTE: if arr is not in canonical form, i.e. if some unused bits
         are 1, then an unexpected value may be returned.
*/

  register     unsigned long count;
  register     elem_t        i;
               elem_t        nelem = NELEM(size,(BITS_SZ));

  static const unsigned bitcount[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, \
        2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, \
        4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, \
        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, \
        3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, \
        4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, \
        5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, \
        2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, \
        4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, \
        4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, \
        6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, \
        4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, \
        5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, \
        6, 6, 7, 6, 7, 7, 8};

   if (bitcount[(sizeof bitcount / sizeof bitcount[0]) - 1] == BITS_SZ) {
     /* lookup table will speed this up a lot */
     for (count = 0L, i = 0; i < nelem; i++) {
        count += bitcount[arr[i]];
     }
   } else {
     for (count = 0L, i = 0; i < size; i++) {
        if (ba_value(arr, i)) {
           count++;
        }
     }
   }
   return (count);
} /* ba_count() */


bool ba_intersection(      bit    first[],
                           bit    second[],
                           bit *  result[],
                     const elem_t size_first,
                     const elem_t size_second)
{
 /*
   ba_intersection()

   PRE:  `first'  is a bit array of at least `size_first'  elements.
         `second' is a bit array of at least `size_second' elements.
         `result' points to enough space to hold the as many elements
         as the smallest of `size_first' and `size_second';
       OR
         `result' points to NULL and such space is to be dynamically
         allocated.
   POST: TRUE has been returned and
         `result' points to a bit array containing the intersection
         of the two arrays up to the smallest of the two sizes;
       OR
         FALSE has been returned and
         `result' pointed to NULL (a request was made to allocate
         enough memory to store the intersection) but the required
         memory could not be obtained.
   NOTE: This runs faster if the `first' array is not smaller than
       `second'.
 */

   register elem_t i;
            elem_t numints;
            unsigned  largest=0, smallest=1;
            BitVector bv[2];

   bv[largest].size   = size_first;
   bv[largest].vector = first;
   bv[smallest].size   = size_second;
   bv[smallest].vector = second;
   first_is_biggest(bv, &largest, &smallest);

   /* allocate space if *result is NULL */
   if ((NULL == *result) && \
      (NULL == (*result = ba_new(bv[largest].size)))) {
      return(FALSE); /* can't get memory, so can't continue */
   } else {
      numints = NELEM(size_second,(BITS_SZ));
      for (i=0; i < numints; i++) {
         (*result)[i] = (bv[smallest].vector[i] & \
                         bv[largest].vector[i]);
      }
      /* bits beyond size_second should be zero -- canonical form */
      CANONIZE(*result, numints, size_second);
      return(TRUE);
   }
} /* ba_intersection() */



bool ba_union(      bit    first[],
                    bit    second[],
                    bit *  result[],
              const elem_t size_first,
              const elem_t size_second)
{
 /*
   ba_union()

   PRE:  `first'  is a bit array of at least `size_first'  elements.
         `second' is a bit array of at least `size_second' elements.
         `result' points to enough space to hold the as many elements
         as the largest of `size_first' and `size_second';
       OR
         `result' points to NULL and such space is to be dynamically
         allocated.
   POST: TRUE has been returned and
         `result' points to a bit array containing the union of the
         two arrays (up to the size of the largest of the two sizes);
       OR
         FALSE has been returned and
         `result' pointed to NULL (a request was made to allocate
         enough memory to store the union) but the required memory
         could not be obtained.
   NOTE: This runs faster if the `first' array is not smaller than
         `second'.
 */

    register elem_t    i;
             elem_t    numints;
             unsigned  largest=0, smallest=1;
             BitVector bv[2];

   bv[largest].size   = size_first;
   bv[largest].vector = first;
   bv[smallest].size   = size_second;
   bv[smallest].vector = second;
   first_is_biggest(bv, &largest, &smallest);
   if ((NULL == *result) && \
      (NULL == (*result = ba_new(bv[largest].size)))) {
      return(FALSE);
   } else {
      ba_copy(*result, bv[largest].vector, bv[largest].size);
      numints = NELEM(bv[smallest].size,(BITS_SZ));
      for (i=0; i < numints; i++) {
         (*result)[i] |= bv[smallest].vector[i];
      }
      CANONIZE(*result, numints, bv[largest].size);
      return(TRUE);
   }
} /* ba_union() */



bool ba_diff(      bit    first[],
                   bit    second[],
                   bit *  diff[],
             const elem_t size_first,
             const elem_t size_second)
{
 /*
   ba_diff()

   PRE:  `first'  is a bit array of at least `size_first'  elements.
         `second' is a bit array of at least `size_second' elements.
         `diff' points to enough space to hold the as many elements
         as the largest of `size_first' and `size_second';
       OR
         `diff' points to NULL and such space is to be dynamically
         allocated.
   POST: TRUE has been returned and
         `diff' points to a bit array containing the union of the
         two arrays (up to the size of the largest of the two sizes);
       OR
         FALSE has been returned and
         `result' pointed to NULL (a request was made to allocate
         enough memory to store the result) but the required memory
         could not be obtained.
   NOTE: This runs faster if the `first' array is not smaller than
         `second'.
 */
    register elem_t    i;
             elem_t    numints;
             unsigned  largest=0, smallest=1;
             BitVector bv[2];

   bv[largest].size   = size_first;
   bv[largest].vector = first;
   bv[smallest].size   = size_second;
   bv[smallest].vector = second;
   first_is_biggest(bv, &largest, &smallest);
   if ((NULL == *diff) && \
      (NULL == (*diff = ba_new(bv[largest].size)))) {
      return(FALSE);
   } else {
      ba_copy(*diff, bv[largest].vector, bv[largest].size);
      numints = NELEM(bv[smallest].size,(BITS_SZ));
      for (i=0; i < numints; i++) {
         (*diff)[i] ^= bv[smallest].vector[i];
      }
      CANONIZE(*diff, numints, bv[largest].size);
      return(TRUE);
   }
} /* ba_diff() */



void ba_complement(      bit    arr[],
                   const elem_t size)
{
/*
   ba_complement()

   PRE:  `arr' is a bit array composed of *exactly* `size' elements.
   POST: All the bits in `arr' have been flipped and `arr' is in
        canonical form.
   SEE ALSO: ba_toggle()
*/
            elem_t nelem = NELEM(size,(BITS_SZ));
   register elem_t i;

   for (i=0; i < nelem; i++) {
      arr[i] = ~arr[i];
   }
   /* force canonical form */
   CANONIZE(arr, nelem, size);
} /* ba_complement() */



unsigned long ba_dotprod(const bit    first[],
                         const bit    second[],
                         const elem_t size_first,
                         const elem_t size_second)
{
 /*
   ba_dotprod()

   PRE: `first' is an array of at least `size_first' bits.  `second'
         is an array of at least `size_second' bits.
   POST: The scalar product of the two vectors represented by the
         first `size_first' elements of `first' and the first
         `size_second' elements of `second' have been returned.
*/
   register elem_t        i, j;
   register unsigned long sum = 0L;

   for (i=0; i < size_first; i++) {
      for (j=0; j < size_second; j++) {
         sum += (first[i/BITS_SZ]  & (1<<(i % BITS_SZ))) \
               &&                                        \
                (second[j/BITS_SZ] & (1<<(j % BITS_SZ)));
      }
   }
   return (sum);
} /* ba_dotprod() */

/*                                                                  *\
   ----------------------------------------------------------------
                             Internal Function
   ----------------------------------------------------------------
\*                                                                  */

static
void first_is_biggest(BitVector  bv[2],
                      unsigned * big,
                      unsigned * small)
{
   if (bv[*big].size < bv[*small].size) {
      unsigned temp;

      temp = *big;
      *big = *small;
      *small = temp;
   }
} /* first_is_biggest() */


/*                                                                  *\
   ----------------------------------------------------------------
                               Miscellaneous
   ----------------------------------------------------------------
\*                                                                  */

bool ba_print(const bit    arr[],
              const elem_t size,
                    FILE * dest)
{
   char * to_print = ba_b2str(arr, size, NULL);

   if (NULL != to_print) {
      bool status = (EOF != fputs(to_print, dest) );
      free(to_print);
      return (status);
   } else {
      return (FALSE);
   }
} /* ba_print() */
