
#include "cycle.h"

// http://en.wikipedia.org/wiki/Binary_GCD_algorithm

static unsigned int calc_GCD( unsigned int a, unsigned int b )
{
    unsigned int shift, tmp;

    if ( a == 0 )
        return b;
    if ( b == 0 )
        return a;

    // Find power of two divisor
    for ( shift = 0; ( ( a | b ) & 1 ) == 0; shift++ )
    {
        a >>= 1;
        b >>= 1;
    }

    // Remove remaining factors of two from a - they are not common
    while ( ( a & 1 ) == 0 )
        a >>= 1;

    do
    {
        // Remove remaining factors of two from b - they are not common
        while ( ( b & 1 ) == 0 )
            b >>= 1;

        if ( a > b )
        {
            tmp = a;
            a   = b;
            b   = tmp;
        } // swap a,b
        b = b - a;
    } while ( b != 0 );

    return a << shift;
}

// circle shift an array left (towards index zero)
// - ptr    array to shift
// - n      number of elements
// - es     size of elements in bytes
// - shift  number of places to shift left
void array_cycle_left( void* _ptr, size_t n, size_t es, size_t shift )
{
    char* ptr = (char*)_ptr;
    if ( n <= 1 || !shift )
        return;        // cannot mod by zero
    shift = shift % n; // shift cannot be greater than n

    // Using GCD
    size_t i, j, k, gcd = calc_GCD( n, shift );
    char tmp[ es ];

    // i is initial starting position
    // Copy from k -> j, stop if k == i, since arr[i] already overwritten
    for ( i = 0; i < gcd; i++ )
    {
        memcpy( tmp, ptr + es * i, es ); // tmp = arr[i]
        for ( j = i; 1; j = k )
        {
            k = j + shift;
            if ( k >= n )
                k -= n;
            if ( k == i )
                break;
            memcpy( ptr + es * j, ptr + es * k, es ); // arr[j] = arr[k];
        }
        memcpy( ptr + es * j, tmp, es ); // arr[j] = tmp;
    }
}

// cycle right shifts away from zero
void array_cycle_right( void* _ptr, size_t n, size_t es, size_t shift )
{
    if ( !n || !shift )
        return;        // cannot mod by zero

    shift = shift % n; // shift cannot be greater than n

    // cycle right by `s` is equivalent to cycle left by `n - s`
    array_cycle_left( _ptr, n, es, n - shift );
}
