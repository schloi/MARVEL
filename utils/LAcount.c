/*******************************************************************************************
 *
 * validates the overlaps prior to sorting and merging
 *
 * Author: MARVEL Team
 *
 * Date  : January 2015
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "db/DB.h"
#include "dalign/filter.h"
#include "lib/oflags.h"
#include "lib/pass.h"

static void usage()
{
    fprintf(stderr, "usage  :<nthreads> <db> <blocks.dir>\n");
}

int main(int argc, char* argv[])
{            
    if (argc != 4)
    {
        usage();
        exit(1);
    }

    int nthreads = atoi(argv[1]);
    char* pcDb = argv[2];
    char* pcPathOverlaps = argv[3];
    
    char* pwd = PathTo(pcDb);
    char* root = Root(pcDb, ".db");

    FILE* fileDb;
    int nfiles, nblocks, i;


    // read the .db file and get the number blocks

    if ( (fileDb = fopen(Catenate(pwd,"/",root,".db"),"r")) == NULL )
    {
        fprintf(stderr, "failed to open database\n");
        exit(1);
    }

    if ( fscanf(fileDb, "files = %d\n", &nfiles) != 1 )
    {
        fprintf(stderr, "format error in database file\n");
        exit(1);
    }

    for (i = 0; i < nfiles; i++)
    { 
        char buffer[30001];
        if (fgets(buffer, 30000, fileDb) == NULL)
        {
            fprintf(stderr, "format error in database file\n");
            exit(1);
        }
    }

    if ( fscanf(fileDb, "blocks = %d\n", &nblocks) != 1 )
    {
        fprintf(stderr, "format error in database file\n");
        exit(1);
    }
    
    // which block are we checking
    
    int len = strlen(pcPathOverlaps);
    if (pcPathOverlaps[len-1] == '/')
    {
        pcPathOverlaps[len-1] = '\0';
    }
    
    char* pcBlock = strrchr(pcPathOverlaps, '_');

    if (!pcBlock)
    {
        fprintf(stderr, "format error in overlaps directory\n");
        exit(1);
    }
    
    char* end;
    int nblock = strtol(pcBlock + 1, &end, 10);
    
    if ( *end != '\0' || nblock < 1 || nblock > nblocks )
    {
        fprintf(stderr, "format error in overlaps directory\n");
        exit(1);
    }
    
    // make sure all .las files for this block exist and are not truncated
    
    printf("%d blocks total\n", nblocks);
    printf("%d threads\n", nthreads);
    printf("checking block %d\n", nblock);
    
    char* path = malloc( strlen(pcPathOverlaps) + 100 );
    
    int passed = 0;
    int failed = 0;

    int block;
    for (block = 1; block <= nblocks; block++)
    {
        int type;
        for (type = 0; type < 2; type++)
        {
            int thread;
            for (thread = 0; thread < nthreads; thread++)
            {
                sprintf(path, "%s/%s.%d.%s.%d.%c%d.las", pcPathOverlaps, root, nblock, root, block, "CN"[type], thread);
                
                FILE* fileLas = fopen(path, "r");
                
                // missing
                
                if (fileLas == NULL)
                {
                    printf("missing %s\n", path);
                    failed++;
                    
                    continue;
                }
                
                // truncated ( 0-byte file )
                
                struct stat st;
                stat(path, &st);
                
                if (st.st_size == 0)
                {
                    printf("zero-size %s\n", path);
                    failed++;
                    
                    fclose(fileLas);
                    
                    continue;
                }
                
                // not 0-byte, but header empty ( overlapper crashed most likely )
                
                ovl_header_novl novl = 0;
                ovl_header_twidth twidth;
                
                ovl_header_read(fileLas, &novl, &twidth);
                
                if (novl == 0)
                {
                    printf("bad header %s\n", path);
                    failed++;
                    
                    fclose(fileLas);
                    
                    continue;
                }
                
                passed++;
                fclose(fileLas);
            }
        }
    }
    
    free(path);

    printf("passed %d\n", passed);
    printf("failed %d\n", failed);

    return (failed != 0);
}
