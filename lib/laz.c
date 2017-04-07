
#include "laz.h"
#include <zlib.h>

LAZ* laz_open(char* fpath, int create)
{
    FILE* fin = NULL;
    LAZ* laz = NULL;

    if (create)
    {
        laz->file = fopen(fpath, "wb");
    }
    else
    {
        laz->file = fopen(fpath, "rb");
    }

    if (laz->file == NULL)
    {
        return NULL;
    }

    laz = calloc( 1, sizeof(LAZ) );
    laz->file = fin;

    return laz;
}

int laz_read_header(LAZ* laz)
{
    LAZ_HEADER header;

    if ( fread(&header, sizeof(LAZ_HEADER), 1, laz->file) != 1 )
    {
        return 0;
    }

    if (header.magic != LAZ_MAGIC)
    {
        return 0;
    }

    laz->version = header.version;
    laz->novl = header.novl;
    laz->twidth = header.twidth;

    return 1;
}

int laz_read_index(LAZ* laz, LAZ_INDEX* lidx)
{
    if ( fread(lidx, sizeof(LAZ_INDEX), 1, laz->file) != 1 )
    {
        return 0;
    }

    return 1;
}

int laz_close(LAZ* laz)
{
    if (laz->file)
    {
        fclose(laz->file);
    }

    if (laz)
    {
        free(laz);
    }

    return 1;
}

Overlap* laz_read(LAZ* laz)
{
    Overlap* ovl = NULL;

    if ( laz->ocur < laz->on )
    {
        ovl = laz->ovl + laz->ocur;
        laz->ocur += 1;
    }
    else
    {
        LAZ_INDEX lidx;

        if ( laz_read_index(laz, &lidx) )
        {
            if ( laz->omax < lidx.novl )
            {
                laz->omax = lidx.novl;
                laz->ovl = realloc( laz->ovl, sizeof(Overlap) * laz->omax );

            }

            laz->on = lidx.novl;
            laz->ocur = 0;

            off_t cur = ftell(laz->file);
            uint64_t olen = lidx.data - cur;
            uint64_t dlen = lidx.next - lidx.data;

            if ( olen > laz->bmax )
            {
                laz->bmax = olen;
                laz->buf = realloc( laz->buf, laz->bmax );
            }

            uLongf destlen;

            fread( laz->buf, olen, 1, laz->file );
            uncompress(laz->ovl, &destlen, laz->buf, olen);
            fseek( laz->file, dlen, SEEK_CUR );
        }
    }

    return ovl;
}

int laz_write(LAZ* laz, Overlap* ovl)
{

}

