
#include <inttypes.h>

int uncompress_chunks(void* ibuf, uint64_t ilen, void* obuf, uint64_t olen);
int compress_chunks(void* ibuf, uint64_t ilen, void** _obuf, uint64_t* _olen);

