#ifndef RADIX_SORT
#define RADIX_SORT

#ifndef WORD_SIZE
#define WORD_SIZE 16
#endif

void Set_Radix_Params( int nthread, int verbose );

void* Radix_Sort( long long len, void* src, void* trg, int* bytes );

#endif // RADIX_SORT
