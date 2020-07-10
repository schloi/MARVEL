
#ifndef __QSORTR_HEADER
#define __QSORTR_HEADER

// qsort_r is not part of the POSIX standard and the various incarnations of the 
// function have different argument orders for the qsort_r call itself and the
// comparison function

#if defined (__APPLE__) || defined (__FreeBSD__)
    #define QSORT_R(A,B,C,D,E) qsort_r(A,B,C,D,E)
    #define QSORT_R_DATA_FIRST
#elif defined(__linux)
    #ifndef __USE_GNU
        #define __USE_GNU
    #endif
    
    #define QSORT_R(A,B,C,D,E) qsort_r(A,B,C,E,D)
    #define QSORT_R_DATA_LAST
#else
    #error "no qsort_r support"
#endif

#endif // not __QSORTR_HEADER