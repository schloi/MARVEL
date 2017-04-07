/* "types.h" */

/*
   IMPORTANT NOTE: This is part of the code that appeared in
                   Dr. Dobb's Journal issue #233 (August 1995)
                   Volume 20 Issue 8
                   in an article entitled `Implementing Bit Vectors in C'
                   by James Blustein 
                   Pages 42, 44, 46 (article) and pages 96, 98-100 (code)
                   The code is (c) copyright 1995 by Miller Freeman, Inc.

   See "bitarr.c" for further details.
*/

#pragma once

#include <stddef.h>

typedef enum bool {FALSE, TRUE} bool;
typedef size_t                  elem_t;
typedef unsigned char           bit;
typedef char *                  string;
