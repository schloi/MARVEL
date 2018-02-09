
#include <assert.h>
#include <stdlib.h>
#include <sys/param.h>
#include <ctype.h>
#include <string.h>

#include "tracks.h"
#include "lib/pass.h"


/*
        fgetln is only available on bsd derived systems.
        getline on the other hand is only in gnu libc derived systems.

        if fgetln is missing emulate it using getline
*/

#if !defined( fgetln )

char* fgetln_( FILE* stream, size_t* len )
{
    static char* linep    = NULL;
    static size_t linecap = 0;
    ssize_t length        = getline( &linep, &linecap, stream );

    if ( length == -1 )
    {
        free( linep );
        linep   = NULL;
        linecap = 0;
        length  = 0;
    }

    if ( len )
    {
        *len = length;
    }

    return linep;
}

#endif

uint64_t bp_parse(const char* num)
{
    char* ptr;
    double count = strtod(num, &ptr);

    if ( *ptr == '\0' )
    {
        return count;
    }

    char unit = tolower( *ptr );

    switch (unit)
    {
        case 'g':
            count *= 1000;
            // fall through

        case 'm':
            count *= 1000;
            // fall through

        case 'k':
            count *= 1000;
            break;

        default:
            fprintf(stderr, "unknown suffix '%c' specified\n", unit);
            break;
    }

    return count;
}

char* bp_format_alloc(uint64_t num, int dec, int alloc)
{
    static char sstr[128];
    static char* str;

    if (alloc)
    {
        str = malloc(128);
    }
    else
    {
        str = sstr;
    }

    char* suffix = "KMG";
    double dnum = num;
    int ns = -1;

    while ( dnum > 1000 )
    {
        dnum /= 1000;
        ns += 1;
    }

    sprintf(str, "%.*f", dec, dnum);

    if (ns != -1)
    {
        int len = strlen(str);
        str[len] = suffix[ns];
        str[len + 1] = '\0';
    }

    return str;
}

char* bp_format(uint64_t num, int dec)
{
    return bp_format_alloc(num, dec, 0);
}

int fread_integers(FILE* fileIn, int** out_values, int* out_nvalues)
{
    int maxvalues = 100;
    int nvalues = 0;
    int* values = malloc( sizeof(int) * maxvalues ) ;
    int n;

    while ( fscanf(fileIn, "%d\n", &n) == 1 )
    {
        if ( maxvalues == nvalues )
        {
            maxvalues = 1.2 * maxvalues + 100;
            values = realloc(values, sizeof(int) * maxvalues);
        }

        values[nvalues] = n;
        nvalues += 1;
    }

    *out_values = values;
    *out_nvalues = nvalues;

    return nvalues;
}

char* format_bytes(unsigned long bytes)
{
    const char* suffix[] = {"b", "kb", "mb", "gb", "tb", "eb"};

    char* buf = malloc(128);

    if (bytes < 1024)
    {
        sprintf(buf, "%lub", bytes);
    }
    else
    {
        double b = bytes;
        int i = 0;

        while ( b >= 1024 )
        {
            b /= 1024.0;
            i++;
        }

        sprintf(buf, "%.1f%s", b, suffix[i]);
    }

    return buf;
}

void wrap_write(FILE* fileOut, char* seq, int len, int width)
{
    int j;
    for (j = 0; j + width < len; j += width)
    {
        fprintf(fileOut, "%.*s\n", width, seq + j);
    }

    if (j < len)
    {
        fprintf(fileOut, "%.*s\n", len - j, seq + j);
    }
}

void revcomp(char* c, int len)
{
    static char comp[128] =
    {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 't', 0, 'g', 0, 0, 0, 'c',
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 'a', 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 't', 0, 'g', 0, 0, 0, 'c',
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 'a', 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    };

    char* b = c;
    char* e = c + len;

    while (b < e)
    {
        unsigned char t = *b;

        *b = comp[ (int)(*e) ];
        *e = comp[ t ];

        b++;
        e--;
    }

    if (b == e)
    {
        *b = comp[ (int)(*b) ];
    }
}

void rev(char* c, int len)
{
    char* b = c;
    char* e = c + len;

    while (b < e)
    {
        unsigned char t = *b;

        *b = *e;
        *e = t;

        b++;
        e--;
    }
}

int intersect(int ab, int ae, int bb, int be)
{
    int b = MAX(ab, bb);
    int e = MIN(ae, be);

    return (b < e ? (e-b) : 0);
}

void get_trim(HITS_DB* db, HITS_TRACK* trimtrack, int rid, int* b, int* e)
{
    track_anno* anno = (track_anno*)trimtrack->anno;

    track_anno ob = anno[rid] / sizeof(track_data);
    track_anno oe = anno[rid+1] / sizeof(track_data);

    assert( (ob == oe) || ((oe - ob) == 2) );

    if (ob == oe)
    {
        *b = 0;
        *e = DB_READ_LEN(db, rid);
    }
    else
    {
        track_data* data = (track_data*)trimtrack->data;

        *b = data[ob];
        *e = data[ob+1];

        if (*b >= *e)
        {
            *b = *e = 0;
        }
    }
}
