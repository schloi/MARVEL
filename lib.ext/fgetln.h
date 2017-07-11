
/*
	fgetln is only available on bsd derived systems.
	getline on the other hand is only in gnu libc derived systems.

	if fgetln is missing emulate it using getline
*/

#include <stdio.h>

#if !defined(fgetln) && !HAVE_FGETLN

static char* fgetln_(FILE* stream, size_t* len)
{
	static char* linep = NULL;
	static size_t linecap = 0;
	ssize_t length = getline(&linep, &linecap, stream);

	if (length == -1)
	{
		free(linep);
		linep = NULL;
		linecap = 0;
		length = 0;
	}
	if (len)
	{
		*len = length;
	}

	return linep;
}

#define fgetln  fgetln_

#endif
