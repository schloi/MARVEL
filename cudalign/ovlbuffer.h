
#pragma once

#define ENABLE_OVL_IO_BUFFER

#include <inttypes.h>
#include "types.h"

Overlap_IO_Buffer* CreateOverlapBuffer( int nthreads, int tbytes, int no_trace );
int AddOverlapToBuffer( Overlap_IO_Buffer* iobuf, Overlap* ovl, int tbytes );
void Write_Overlap_Buffer( Align_Spec* spec, const char* path, char* ablock, char* bblock, int lastRead );
void Reset_Overlap_Buffer( Align_Spec* spec );
Overlap_IO_Buffer* OVL_IO_Buffer( Align_Spec* espec );
int Num_Threads( Align_Spec* espec );

int mkdir_p( const char* path );
