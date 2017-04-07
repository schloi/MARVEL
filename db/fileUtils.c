#include "fileUtils.h"
#include "DB.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>  

int isPacBioHeader(char* header)
  {
    char* c = header;

    char* nameEnd;
    char *pch;
    // check if pacbio header is available
      {
        nameEnd = strchr(c, ' ');
        if (nameEnd == NULL)
          nameEnd = c + strlen(c) - 1;

        // TODO to be more flexible, ignore number of '_' symbols ??
        /*
        // check number of underscore symbols equals 6 (>m140913_050931_42139_c100713652400000001823152404301535_s1_p0/9/1607_26058)
        int ucnt = 0;
        pch = strchr(c, '_');
        while (pch != NULL && pch < nameEnd)
          {
            ucnt++;
            pch = strchr(pch + 1, '_');
          }

        if (ucnt != 6)
          return 0;
        */
        // check number of slash symbols equals 2 (>m140913_050931_42139_c100713652400000001823152404301535_s1_p0/9/1607_26058)
        int scnt = 0;
        pch = strchr(c, '/');
        while (pch != NULL && pch < nameEnd)
          {
            scnt++;
            pch = strchr(pch + 1, '/');
          }

        if (scnt != 2)
          return 0;
      }
    return 1;
  }  

File_Iterator *init_file_iterator(int argc, char **argv, FILE *input, int first)
{ File_Iterator *it;

  it = Malloc(sizeof(File_Iterator),"Allocating file iterator");
  it->argc  = argc;
  it->argv  = argv;
  it->input = input;
  if (input == NULL)
    it->count = first;
  else
    { it->count = 1;
      rewind(input);
    }
  return (it);
}

int next_file(File_Iterator *it)
{ static char nbuffer[MAX_NAME+8];

  if (it->input == NULL)
    { if (it->count >= it->argc)
        return (0);
      it->name = it->argv[it->count++];
    }
  else
    { char *eol;

      if (fgets(nbuffer,MAX_NAME+8,it->input) == NULL)
        { if (feof(it->input))
            return (0);
          SYSTEM_ERROR;
        }
      if ((eol = index(nbuffer,'\n')) == NULL)
        { fprintf(stderr,"%s: Line %d in file list is longer than %d chars!\n",
                         Prog_Name,it->count,MAX_NAME+7);
          it->name = NULL;
        }
      *eol = '\0';
      it->count += 1;
      it->name  = nbuffer;
    }
  return (1);
}  

Read_Iterator *init_read_iterator(FILE *input)
{ Read_Iterator *it;

  it = Malloc(sizeof(Read_Iterator),"Allocating file iterator");
  it->input = input;
  it->lineno = 1;
  rewind(input);
  return (it);
}

int next_read(Read_Iterator *it)
{ static char nbuffer[MAX_BUFFER];

  char *eol;
  int   x;

  if (fgets(nbuffer,MAX_BUFFER,it->input) == NULL)
    { if (feof(it->input))
        return (1);
      SYSTEM_ERROR;
    }
  if ((eol = index(nbuffer,'\n')) == NULL)
    { fprintf(stderr,"%s: Line %d in read list is longer than %d chars!\n",
                     Prog_Name,it->lineno,MAX_BUFFER-1);
      return (1);
    }
  *eol = '\0';
  x = sscanf(nbuffer," %d %d %d",&(it->read),&(it->beg),&(it->end));
  if (x == 1)
    it->beg = -1;
  else if (x != 3)
    { fprintf(stderr,"%s: Line %d of read list is improperly formatted\n",Prog_Name,it->lineno);
      return (1);
    }
  it->lineno += 1;
  return (0);
}

