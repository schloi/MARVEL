/************************************************************************************\
*                                                                                    *
* Copyright (c) 2014, Dr. Eugene W. Myers (EWM). All rights reserved.                *
*                                                                                    *
* Redistribution and use in source and binary forms, with or without modification,   *
* are permitted provided that the following conditions are met:                      *
*                                                                                    *
*  · Redistributions of source code must retain the above copyright notice, this     *
*    list of conditions and the following disclaimer.                                *
*                                                                                    *
*  · Redistributions in binary form must reproduce the above copyright notice, this  *
*    list of conditions and the following disclaimer in the documentation and/or     *
*    other materials provided with the distribution.                                 *
*                                                                                    *
*  · The name of EWM may not be used to endorse or promote products derived from     *
*    this software without specific prior written permission.                        *
*                                                                                    *
* THIS SOFTWARE IS PROVIDED BY EWM ”AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,    *
* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND       *
* FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL EWM BE LIABLE   *
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS  *
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     *
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN  *
* IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                      *
*                                                                                    *
* For any issues regarding this software and its use, contact EWM at:                *
*                                                                                    *
*   Eugene W. Myers Jr.                                                              *
*   Bautzner Str. 122e                                                               *
*   01099 Dresden                                                                    *
*   GERMANY                                                                          *
*   Email: gene.myers@gmail.com                                                      *
*                                                                                    *
\************************************************************************************/

/********************************************************************************************
 *
 *  Concate in block order all "block tracks" <DB>.<track>.# into a single track
 *    <DB>.<track>
 *
 *  Author:  Gene Myers
 *  Date  :  June 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "DB.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

// needed for getopt

extern char *optarg;
extern int optind, opterr, optopt;

static void usage()
{
    fprintf(stderr, "[-vd] <path:db> <track:name>\n");
}

int main(int argc, char* argv[])
{
    char* prefix;
    FILE* aout, *dout;
    int   VERBOSE = 0;
    int   DELETE = 0;

    //  Process arguments

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "vd")) != -1)
    {
        switch (c)
        {
            case 'v':
                      VERBOSE = 1;
                      break;

            case 'd':
                      DELETE = 1;
                      break;  
                      
            default:
                      usage();
                      exit(1);          
        }    
    }

    if (argc - optind != 2)
    {
        usage();
        exit(1);
    }
    
    char* path_db = argv[optind];
    char* track_name = argv[optind+1];
    
    {
        char* pwd, *root;

        pwd    = PathTo(path_db);
        root   = Root(path_db, ".db");
        prefix = Strdup(Catenate(pwd, PATHSEP, root, "."), "Allocating track name");
        free(pwd);
        free(root);

        aout = fopen(Catenate(prefix, track_name, ".", "anno"), "r");

        if (aout != NULL)
        {
            fprintf(stderr, "%s: Track file %s%s.anno already exists!\n", Prog_Name, prefix, track_name);
            fclose(aout);
            exit (1);
        }

        dout = fopen(Catenate(prefix, track_name, ".", "data"), "r");

        if (dout != NULL)
        {
            fprintf(stderr, "%s: Track file %s%s.data already exists!\n", Prog_Name, prefix, track_name);
            fclose(aout);
            exit (1);
        }

        aout = Fopen(Catenate(prefix, track_name, ".", "anno"), "w");

        if (aout == NULL)
        {
            exit (1);
        }

        dout = NULL;
    }

    {
        int   tracktot, tracksiz;
        int64 trackoff;
        int   nfiles;
        char  data[1024];
        void* anno;

        anno     = NULL;
        trackoff = 0;
        tracktot = tracksiz = 0;
        fwrite(&tracktot, sizeof(int), 1, aout);
        fwrite(&tracksiz, sizeof(int), 1, aout);

        nfiles = 0;
        
        if (VERBOSE)
        {
            printf("concatenating ... \n");
        }

        while (1)
        {
            FILE* afile, *dfile;
            int   i, size, tracklen;

            afile = fopen(Numbered_Suffix(prefix, nfiles + 1, Catenate(".", track_name, ".", "anno")), "r");

            if (afile == NULL)
            {
                break;
            }

            dfile = fopen(Numbered_Suffix(prefix, nfiles + 1, Catenate(".", track_name, ".", "data")), "r");

            if (VERBOSE)
            {
                printf("  %s%d.%s\n", prefix, nfiles + 1, track_name);
            }

            if (fread(&tracklen, sizeof(int), 1, afile) != 1)
            {
                SYSTEM_ERROR;
            }

            if (fread(&size, sizeof(int), 1, afile) != 1)
            {
                SYSTEM_ERROR;
            }

            if (nfiles == 0)
            {
                tracksiz = size;

                if (dfile != NULL)
                {
                    dout = Fopen(Catenate(prefix, track_name, ".", "data"), "w");

                    if (dout == NULL)
                    {
                        fclose(afile);
                        fclose(dfile);
                        goto error;
                    }
                }
                else
                {
                    anno = Malloc(size, "Allocating annotation record");

                    if (anno == NULL)
                    {
                        fclose(afile);
                        goto error;
                    }
                }
            }
            else
            {
                int escape = 1;

                if (tracksiz != size)
                {
                    fprintf(stderr, "%s: Track block %d does not have the same annotation size (%d)",
                            Prog_Name, nfiles + 1, size);
                    fprintf(stderr, " as previous blocks (%d)\n", tracksiz);
                }
                else if (dfile == NULL && dout != NULL)
                    fprintf(stderr, "%s: Track block %d does not have data but previous blocks do\n",
                            Prog_Name, nfiles + 1);
                else if (dfile != NULL && dout == NULL)
                    fprintf(stderr, "%s: Track block %d has data but previous blocks do not\n",
                            Prog_Name, nfiles + 1);
                else
                {
                    escape = 0;
                }

                if (escape)
                {
                    fclose(afile);

                    if (dfile != NULL)
                    {
                        fclose(dfile);
                    }

                    if (anno != NULL)
                    {
                        free(anno);
                    }

                    goto error;
                }
            }

            if (dfile != NULL)
            {
                int64 dlen;

                if (size == 4)
                {
                    int anno4;

                    for (i = 0; i < tracklen; i++)
                    {
                        if (fread(&anno4, sizeof(int), 1, afile) != 1)
                        {
                            SYSTEM_ERROR;
                        }

                        anno4 += trackoff;
                        fwrite(&anno4, sizeof(int), 1, aout);
                    }

                    if (fread(&anno4, sizeof(int), 1, afile) != 1)
                    {
                        SYSTEM_ERROR;
                    }

                    dlen = anno4;
                }
                else
                {
                    int64 anno8;

                    for (i = 0; i < tracklen; i++)
                    {
                        if (fread(&anno8, sizeof(int64), 1, afile) != 1)
                        {
                            SYSTEM_ERROR;
                        }

                        anno8 += trackoff;
                        fwrite(&anno8, sizeof(int64), 1, aout);
                    }

                    if (fread(&anno8, sizeof(int64), 1, afile) != 1)
                    {
                        SYSTEM_ERROR;
                    }

                    dlen = anno8;
                }

                trackoff += dlen;

                for (i = 1024; i < dlen; i += 1024)
                {
                    if (fread(data, 1024, 1, dfile) != 1)
                    {
                        SYSTEM_ERROR;
                    }

                    fwrite(data, 1024, 1, dout);
                }

                i -= 1024;

                if (i < dlen)
                {
                    if (fread(data, dlen - i, 1, dfile) != 1)
                    {
                        SYSTEM_ERROR;
                    }

                    fwrite(data, dlen - i, 1, dout);
                }
            }
            else
            {
                for (i = 0; i < tracklen; i++)
                {
                    if (fread(anno, size, 1, afile) != 1)
                    {
                        SYSTEM_ERROR;
                    }

                    fwrite(anno, size, 1, aout);
                }
            }

            tracktot += tracklen;
            nfiles   += 1;

            if (dfile != NULL)
            {
                fclose(dfile);
            }

            fclose(afile);
        }

        if (nfiles == 0)
        {
            fprintf(stderr, "%s: Couldn't find first track block %s1.%s.anno\n",
                    Prog_Name, prefix, track_name);
            goto error;
        }
        else
        {
            if (dout != NULL)
            {
                if (tracksiz == 4)
                {
                    int anno4 = trackoff;
                    fwrite(&anno4, sizeof(int), 1, aout);
                }
                else
                {
                    int64 anno8 = trackoff;
                    fwrite(&anno8, sizeof(int64), 1, aout);
                }
            }
            else
            {
                fwrite(anno, tracksiz, 1, aout);
                free(anno);
            }

            rewind(aout);
            fwrite(&tracktot, sizeof(int), 1, aout);
            fwrite(&tracksiz, sizeof(int), 1, aout);
        }
    }

    fclose(aout);

    if (dout != NULL)
    {
        fclose(dout);
    }

    if (DELETE)
    {
        int nfiles = 1;

        while (1)
        {
            if ( unlink(Numbered_Suffix(prefix, nfiles, Catenate(".", track_name, ".", "anno"))) != 0 )
            {
                break;
            }
            
            unlink(Numbered_Suffix(prefix, nfiles, Catenate(".", track_name, ".", "data")));
        
            nfiles++;
        }
    }
    

    free(prefix);

    exit (0);

error:
    fclose(aout);
    unlink(Catenate(prefix, track_name, ".", "anno"));

    if (dout != NULL)
    {
        fclose(dout);
        unlink(Catenate(prefix, track_name, ".", "data"));
    }

    free(prefix);

    exit (1);
}
