
#include "iseparator.h"

#include <inttypes.h>
#include <stdio.h>

static void iseparator_rec( size_t lidx, size_t ridx, void* pts, size_t npts, size_t elsz, int64_t maxdist,
                            uint64_t** intervals, size_t* maxintervals, size_t* nintervals, eldistfunc eldist )
{
    uint64_t* _intervals = *intervals;
    size_t _maxintervals = *maxintervals;
    size_t _nintervals   = *nintervals;

    if ( eldist( pts + ridx * elsz, pts + lidx * elsz ) <= maxdist )
    {
        if ( _nintervals + 2 >= _maxintervals )
        {
            _maxintervals = _maxintervals * 1.2 + 100;
            _intervals    = realloc( _intervals, sizeof( uint64_t ) * _maxintervals );
        }

        _intervals[ _nintervals ]     = lidx;
        _intervals[ _nintervals + 1 ] = ridx;

        _nintervals += 2;

        *intervals    = _intervals;
        *maxintervals = _maxintervals;
        *nintervals   = _nintervals;

        return;
    }

    int64_t ptsmaxdist     = 0;
    uint64_t ptsmaxdistidx = 0;

    size_t i;
    for ( i = lidx; i < ridx; i++ )
    {
        int32_t dist = eldist( pts + ( i + 1 ) * elsz, pts + i * elsz );

        if ( dist > ptsmaxdist )
        {
            ptsmaxdist    = dist;
            ptsmaxdistidx = i;
        }
    }

    iseparator_rec( lidx, ptsmaxdistidx, pts, npts, elsz, maxdist, intervals, maxintervals, nintervals, eldist );
    iseparator_rec( ptsmaxdistidx + 1, ridx, pts, npts, elsz, maxdist, intervals, maxintervals, nintervals, eldist );
}

size_t iseparator( void* pts, size_t npts, size_t elsz, uint64_t maxdist, uint64_t** intervals, size_t* maxintervals,
                   eldistfunc eldist )
{
    size_t nintervals = 0;

    iseparator_rec( 0, npts - 1, pts, npts, elsz, maxdist, intervals, maxintervals, &nintervals, eldist );

    return nintervals;
}
