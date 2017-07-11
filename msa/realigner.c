/*****************************************************************************************\
*                                                                                         *
*  These routines improve a multi-alignment of DNA sequences                              *
*                                                                                         *
*  This is a re-implementation of realigner, the original by Eric Anson, ~1997 and the    *
*    algorithm design by Anson and Myers.  This code is pretty much a complete rewrite    *
*    although the spirit of the original code is retained with some coding refinements    *
*    to improve space and time performance.  The code is also modularized and commented   *
*    so that other hackers can potentially integrate the code directly into another       *
*    program.                                                                             *
*                                                                                         *
*  Routines have been added to build an initial multialignment from pairwise overlaps     *
*    and to deliver columns and rows of the multi-alignment.  The construction routines   *
*    are space inefficient and could use a D&C alignment delivering approach but I have   *
*    not as yet had the time to craft such a complicated code.                            *
*                                                                                         *
*  Author:  Gene Myers                                                                    *
*  Date  :  March 2007                                                                    *
*                                                                                         *
\*****************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>

#include "realigner.h"

#undef  DEBUG
#undef  STATS
#undef  CHECK_PRINT
#undef  CHECK_MEMORY
#undef  DEBUG_REALIGN
#undef  DEBUG_SMALL            /* Turn on if fragments are small, makes prettier debug display */
#undef  SHOW_CUMULATIVE_SCORE  /* Prints cumulative column score with alignment
                                  in re_print_contig                                           */

#define MIX   .5  /* Mixture ratio of derivative vs. consensus score to use in realigning */
#define ALONE .5  /* Score for realigning a char in its own column (at a boundary)        */

#define INIT_MAX_CONTIG_DEPTH  128
#define INIT_MAX_FRAG_LENGTH  1024
#define SPACE_GROWTH_RATE      1.2
#define LIST_SPACE_BLOCK_SIZE 8192

#define MAX_CODE  5
#define DASH_CODE 4         /* Code for -'s must be second to last code */
#define N_CODE    5         /* Code for N's must be the last code */

/* Internally a multi-alignment is represented as a sparse matrix of symbols
   that are in doubly-linked, circular, and anchored column and fragment sequence
   lists.  The anchors of the column lists are themselves doubly-linked into
   a sequential list and further contain summary information on the frequency of
   characters in the column (ACGT-N --> 0-5).  The list of columns is terminated at each
   end by an anchor ('first' and 'last') with an empty column.  The anchors of each
   fragment are in a singly-linked list, in no particular order, have a '\0' char
   as their letter, and further contain a pointer to the column anchor containing
   the fragment's first character.

   NB: Leading and trailing '-'s are removed from fragments when they are read
   in.  This may give rise to an empty fragment.  These are not removed, but
   simply checked for where necessary and then ignored.
*/

typedef struct re_symbol
  { struct re_symbol *up;     /* double column chain links */
    struct re_symbol *down;
    struct re_symbol *prev;   /* double fragment sequence links */
    struct re_symbol *next;
    int               letter; /* letter at position */
  } re_symbol;

typedef struct re_column
  { struct re_symbol   *up;         /* anchor of circular, doubly-linked column chain */
    struct re_symbol   *down;
    struct re_column   *pred;       /* double column node link */
    struct re_column   *succ;
    int        depth;               /* total number of symbols in column */
    int        count[MAX_CODE+1];   /* number of each type of symbol in column */
                                    /* Used during realignment: */
    int        bound;               /*    inter-column position to left is a boundary */
    float      score[MAX_CODE+1];   /*    alignment scores for realignment            */
  } re_column;

typedef struct re_fragment
  { struct re_fragment *link;   /* single linked list of fragment records */
    struct re_column   *scol;   /* first column containing a symbol of the fragment */
    struct re_symbol   *prev;   /* anchor of circular, doubly linked fragment sequence */
    struct re_symbol   *next;
    int                 letter; /* '\0' char to identify record as a fragment and not a symbol */
    int                 row;    /* row sequence was on when input */
  } re_fragment;

typedef struct
  { re_fragment *frags;     /* anchor of singly-linked fragment list */
    re_column   *first;     /* first and last columns of contig */
    re_column   *last;
  } re_contig;


static int *encode = NULL;  /* Map DNA + '-' to [0,MAX_CODE] */

static void setup_encode()
{ static int _encode[128];
  int i;

  encode = _encode;
  for (i = 0; i < 128; i++)
    encode[i] = 5;
  encode['a'] = encode['A'] = 0;
  encode['c'] = encode['C'] = 1;
  encode['g'] = encode['G'] = 2;
  encode['t'] = encode['T'] = 3;
  encode['-'] = 4;
}


/* Uniform list space of symbol, column and fragment records
     (one macro defines all)                                   */

static int avail_cells = 0;

#define LIST_SPACE(type,new_routine,free_routine,free_ptr,free_link)			\
											\
static type *free_ptr = NULL;								\
											\
static type *new_routine(void)								\
{ type *ptr;										\
											\
  if (free_ptr == NULL)									\
    { int i;										\
      free_ptr = ptr = (type *)								\
          malloc(sizeof(type)*LIST_SPACE_BLOCK_SIZE);		\
      for (i = 0; i < LIST_SPACE_BLOCK_SIZE; i++)					\
        { ptr->free_link = ptr+1;							\
          ptr += 1;									\
        }										\
      ptr[-1].free_link = NULL;								\
      avail_cells += LIST_SPACE_BLOCK_SIZE;						\
    }											\
											\
  ptr      = free_ptr;									\
  free_ptr = ptr->free_link;								\
  return (ptr);										\
}											\
											\
static void free_routine(type *ptr)							\
{ ptr->free_link = free_ptr;								\
  free_ptr       = ptr;									\
}

LIST_SPACE(re_symbol,new_symbol,free_symbol,free_symbol_ptr,next)

LIST_SPACE(re_column,new_column,free_column,free_column_ptr,succ)

LIST_SPACE(re_fragment,new_fragment,free_fragment,free_fragment_ptr,link)

#ifdef CHECK_MEMORY

static void check_memory()
{ re_symbol   *s;
  re_column   *c;
  re_fragment *f;
  int          free_cells;

  free_cells = 0;
  for (s = free_symbol_ptr; s != NULL; s = s->next)
    free_cells += 1;
  for (c = free_column_ptr; c != NULL; c = c->succ)
    free_cells += 1;
  for (f = free_fragment_ptr; f != NULL; f = f->link)
    free_cells += 1;
  fprintf(stderr,"Avail %d Free %d\n",avail_cells,free_cells);
}

#endif

Re_Contig *Re_Read_Contig(FILE *input, FILE *comments)
{ static re_contig readctg;

  int          sym;
  re_column   *curcol, *firstcol;
  re_fragment *frags;

  if (encode == NULL)
    setup_encode();

  /* Start with empty first column, and empty fragment list */

  firstcol = curcol = new_column();
  curcol->up = curcol->down = (re_symbol *) curcol;
  curcol->pred = NULL;

  frags    = NULL;

  /* NB: The "trick" in this code is that elements containing blank are initially
     placed in a column and then removed after the succeeding column is built.  This
     allows one to easily detect when a new fragment is starting, i.e. blank in the
     previous column & non-blank in the current column of a *corresponding* row.     */

  do
    { if (firstcol != curcol)        /* Not first time through this loop ==> saw a blank line */
        { free_column(curcol);       /*   on the last pass, clean up and start over           */
          curcol = firstcol;
        }

      do
        { re_column *prevcol;
          re_symbol *p, *e;
          int        row;

          sym = fgetc(input);

          while (sym == '%')          /* Pass on any comment lines if comments != NULL */
            { while (sym != '\n')
                { if (comments != NULL)
                    fputc(sym,comments);
                  sym = fgetc(input);
                }
              if (comments != NULL)
                fputc('\n',comments);
              sym = fgetc(input);
            }

          /* Add new, initially empty column */

          prevcol = curcol;
          curcol  = new_column();
          curcol->up = curcol->down = (re_symbol *) curcol;
          prevcol->succ = curcol;
          curcol->pred  = prevcol;
          { int i;
            for (i = 0; i <= MAX_CODE; i++)
              curcol->count[i] = 0;
          }
          curcol->depth = 0;

          /* Add symbols to the new column (including blanks), traversing previous one
             in synchrony in order to detect new fragment beginnings.  If a new fragment
             begins with -'s, turn them into blanks and delay starting the fragment       */

          p   = (re_symbol *) prevcol->down;
          row = 0;
          if (sym != EOF)
            while (sym != '\n')
              { e = new_symbol();
                e->down = (re_symbol *) curcol;
                e->up   = curcol->up;
                e->down->up = e->up->down = e;

                if (sym != ' ')
                  { if (p == (re_symbol *) prevcol || p->letter == ' ') /* New fragment?  */
                      if (sym == '-')                                /* Starts with -? */
                        sym = ' ';
                      else
                        { re_fragment *f;
                          f = new_fragment();
                          f->scol   = curcol;
                          f->letter = '\0';
                          f->link   = frags;
                          f->row    = row;
                          frags     = f;
                          e->prev = e->next = (re_symbol *) f;
                        }
                    else
                      { e->prev = p;
                        e->next = p->next;
                      }
                    if (sym != ' ')
                      { e->prev->next = e->next->prev = e;
                        curcol->count[encode[sym]] += 1;
                        curcol->depth += 1;
                      }
                  }
                e->letter = sym;

                if (p != (re_symbol *) prevcol) /* Keep from running off the end */
                  p = p->down;
                sym  = fgetc(input);
                row += 1;
              }

          /* Purge blank symbols from the previous column, and
             if a fragment ends with -'s peel them back.        */

          { re_symbol *q, *r;
            re_column *c;

            e = curcol->down;
            for (p = prevcol->down; p != (re_symbol *) prevcol; p = q)
              { q = p->down;
                if (p->letter == ' ')
                  { q->up = p->up;
                    p->up->down = q;
                    free_symbol(p);
                  }
                else if (e == (re_symbol *) curcol || e->letter == ' ')
                  { for (c = prevcol; p->letter == '-'; p = r)
                      { r = p->prev;
                        p->next->prev = r;
                        r->next = p->next;
                        p->up->down = p->down;
                        p->down->up = p->up;
                        free_symbol(p);
                        c->count[DASH_CODE] -= 1;
                        c->depth -= 1;
                        c = c->pred;
                      }
                  }
                if (e != (re_symbol *) curcol)
                  e = e->down;
              }
          }

        }
      while (curcol->depth > 0);

      { re_symbol *s;       /* Remove any blank cells from the last column */

        while ((s = curcol->down) != (re_symbol *) curcol)
          { s->up->down = s->down;
            s->down->up = s->up;
            free_symbol(s);
          }
      }
    }
  while (sym != EOF && firstcol->succ == curcol);

  curcol->succ = NULL;

  if (firstcol->succ == curcol)   /* Nothing but zero or more blank lines and then EOF */
    { free_column(curcol);        /*   Free columns and return NULL                    */
      free_column(firstcol);
      return (NULL);
    }

  /* Fill in the contig record */

  readctg.frags = frags;
  readctg.first = firstcol;
  readctg.last  = curcol;

  return ((Re_Contig *) (&readctg));
}

typedef struct
  { int             next;   /* Next unallocated row (in ordinal order) */
    re_symbol      *token;  /* Current symbol element for this row, or NULL if unallocated */
  } rowrec;

void Re_Print_Contig(Re_Contig *contig, FILE *output, int samerows)
{ static rowrec *row;
  static int     avail;
  static int     maxrows = 0;
#ifdef SHOW_CUMULATIVE_SCORE
  int score;
#endif

  re_column *col;
  rowrec    *top;

  /* We keep an array, row[0..maxrows-1], that expands as necessary so that we
     can have a record row[i] for each row in which a fragment will be displayed.
     Each fragment is printed in a row in such a way that fragments in a given row
     are separated by at least one blank.  At a given column, the top row 'top'
     that contains a fragment is maintained.  A list of the rows to which
     a new fragment can be assigned is in the singly linked list anchored at
     avail and the list is ordered according to increasing row number. Integer
     indices into rows are used for the avail list as it may be moved by realloc.  */

  if (maxrows == 0)
    { rowrec *r;

      maxrows = INIT_MAX_CONTIG_DEPTH;
      row = (rowrec *) malloc(sizeof(rowrec)*maxrows);
      for (r = row+(maxrows-1); r >= row; r--)
        { r->next  = (r+1) - row;
          r->token = NULL;
        }
    }

#ifdef SHOW_CUMULATIVE_SCORE
  score = 0;
#endif

  top   = row-1;
  avail = 0;

  for (col = ((re_contig *) contig)->first->succ; col != NULL; col = col->succ)
    { rowrec    *r;
      re_symbol *s;

      /* Examine column to find fragments starts and allocate them to new rows */

      for (s = col->down; s != (re_symbol *) col; s = s->down)
        if (s->prev->letter == '\0')
          { int slot;

            if (samerows)
              { slot = ((re_fragment *) (s->prev))->row;
                if (slot < maxrows && row[slot].token != NULL)
                  slot = avail;
              }
            else
              slot = avail;

            if (slot >= maxrows)
              { int oldmax;

                oldmax  = maxrows;
                maxrows = SPACE_GROWTH_RATE*slot + INIT_MAX_CONTIG_DEPTH;
                row = (rowrec *) realloc(row,sizeof(rowrec)*maxrows);
                top = row + oldmax;
                for (r = row+(maxrows-1); r >= top; r--)
                  { r->next  = (r+1) - row;
                    r->token = NULL;
                  }
                if (avail != oldmax)
                  { rowrec *t;
                    t = row + avail;
                    while (t->next != oldmax)
                      t = row + (t->next);
                    t->next = oldmax;
                  }
              }

            r = row + slot;
            if (slot != avail)
              { rowrec *t;
                t = row + avail;
                while (t->next != slot)
                  t = row + (t->next);
                t->next = r->next;
              }
            else
              avail = r->next;
            if (r > top) top = r;
            r->token = s->prev;
          }

#ifdef SHOW_CUMULATIVE_SCORE
      { int i, max;

        max = col->count[DASH_CODE];
        for (i = 0; i < DASH_CODE; i++)
          if (col->count[i] > max)
            max = col->count[i];
         score += (col->depth-max);
      }
#endif

      /* Advance all active row pointers and rebuild avail
         list of all unallocated rows in the range row..top */

      avail = (top+1) - row;
      for (r = top; r >= row; r--)
        { if ((s = r->token) != NULL)
            { r->token = (s = s->next);
              if (s->letter == '\0')
                r->token = (s = NULL);
            }
          if (s == NULL)
            { r->next = avail;
              avail = r - row;
              if (r == top) top -= 1;
            }
        }

      /* Print the current row */

      for (r = row; r <= top; r++)
        if ((s = r->token) == NULL)
          fputc(' ',output);
        else
          fputc(s->letter,output);
#ifdef SHOW_CUMULATIVE_SCORE
      fprintf(output," %d",score);
#endif
      fputc('\n',output);
    }

#ifdef CHECK_PRINT
  { rowrec *r;

    if (top >= row)
      fprintf(stderr,"Top is not reset correctly = %d\n",top-row);
    for (r = row; r < row + (maxrows-1); r++)
      if (r->next != (r+1)-row)
        fprintf(stderr,"Free list broken %d->%d\n",r-row,r->next);
    r = row + (maxrows-1);
    if (r->next != maxrows)
      fprintf(stderr,"Free list broken %d->%d\n",r-row,r->next);
    if (avail != 0)
      fprintf(stderr,"Free list broken Free->%d\n",r-row,r->next);
  }
#endif
}

void Re_Print_Model(Re_Contig *contig, FILE *output)
{ int        i;
  re_column *c;

  fprintf(output,"\n");
  for (c = ((re_contig *) contig)->first; c != NULL; c = c->succ)
    { fprintf(output,"%p: ",c);
      for (i = 0; i <= MAX_CODE; ++i)
        fprintf(output,"%d   ",c->count[i]);
      fprintf(output,"\n");
    }
  fflush(output);
}

void Re_Print_Structure(Re_Contig *contig, FILE *output)
{ re_column   *c;
  re_fragment *f;
  re_symbol   *s;
  int          i;

  fprintf(output,"\nContig dump:\n");
  for (c = ((re_contig *) contig)->first; c != NULL; c = c->succ)
    { fprintf(output,"Column %p: ",c);
      for (i = 0; i <= MAX_CODE; i++)
        fprintf(output,"%d ",c->count[i]);
      fprintf(output,"+= %d\n  ",c->depth);
      for (s = c->down; s != (re_symbol *) c; s = s->down)
        fprintf(output," %c",s->letter);
      fprintf(output,"\n");
    }
  for (f = ((re_contig *) contig)->frags; f != NULL; f = f->link)
    { fprintf(output,"Fragment %p(%p)\n  '",f,f->scol);
      for (s = f->next; s != (re_symbol *) f; s = s->next)
        fprintf(output,"%c",s->letter);
      fprintf(output,"'\n");
    }
  fflush(output);
}

void Re_Free_Contig(Re_Contig *contig)
{ re_contig   *ctg = (re_contig *) contig;
  re_fragment *f, *g;
  re_symbol   *s;

  /* Each fragment list can be linked onto its free list as a
     linked block, as can the entire list of column anchors    */

  for (f = ctg->frags; f != NULL; f = g)
    { g = f->link;
      s = (re_symbol *) f;
      if (s != s->next)
        { s->prev->next   = free_symbol_ptr;
          free_symbol_ptr = s->next;
        }
      free_fragment(f);
    }
  ctg->last->succ   = free_column_ptr;
  free_column_ptr   = ctg->first;
}

Re_Contig *Re_Start_Contig(int id, char *seq)
{ static re_contig seedctg;

  re_column   *c, *d;
  re_symbol   *e, *f;
  char        *s;

  re_column   *first, *last;
  re_fragment *frag;

  if (encode == NULL)
    setup_encode();

  d = first = new_column();
  d->down = d->up = (re_symbol *) d;
  d->pred = NULL;
  f = (re_symbol *) (frag = new_fragment());

  for (s = seq; *s != '\0'; s++)
    { c = new_column();
      e = new_symbol();

      c->down = c->up = e;
      e->down = e->up = (re_symbol *) c;

      e->letter = *s;
      c->depth  = 1;
      { int i;
        for (i = 0; i <= MAX_CODE; i++)
          c->count[i] = 0;
      }
      c->count[encode[(int)(*s)]] = 1;

      e->prev = f;
      f->next = e;
      c->pred = d;
      d->succ = c;

      f = e;
      d = c;
    }

  e = (re_symbol *) frag;
  frag->letter = '\0';
  frag->row    = id;
  frag->scol   = (re_column *) e->next->up;
  frag->link   = NULL;
  e->prev   = f;
  f->next   = e;;

  c = last = new_column();
  c->down = c->up = (re_symbol *) c;
  c->pred = d;
  d->succ = c;
  c->succ = NULL;

  seedctg.first = first;
  seedctg.last  = last;
  seedctg.frags = frag;

  return ((Re_Contig *) (&seedctg));
}

/* Add the bread-sequence of overlap o to contig contig, where seq is
   the oriented sequence of the bread.  It is assumed that the aread
   sequence seeded the multi-alignment with a call to re_start_contig. */

void Re_Add_Overlap(Re_Contig *contig, Overlap *o, char *seq, int bandsize)
{ static re_fragment *last;

  static double *matrix = NULL;
  static char   *trace  = NULL;
  static int    *cseq   = NULL;
  static int     dpmax  = -1;
  static int     segmax = -1;

  re_contig   *ctg = (re_contig *) contig;

  re_fragment *base;             /* The re_fragment record for the aread                   */
  re_column   *cstart, *cfinis;  /* Column interval to realign against                     */
  int          lmargin, rmargin; /* Amount of band wrapped around left/right end of contig */
  int          seglen;           /* The # of columns between cstart & cfinis, inclusive    */
  int          blen;             /* Length of bread + 1                                    */

  double    *ncol;      /* Current dp array column                      */
  char      *cbck;      /* Current traceback column                     */
  re_column *mincol;    /* Next column in traceback during realignment  */

#define INS 0
#define SUB 1
#define DEL 2

  if (encode == NULL)
    setup_encode();

  /* Find reference read in fragment list.  Linear scan save always check for the
     last one => preorder traversal of overlap tree is linear time over all adds!   */

  { int aid;

    aid = o->aread;
    if (ctg->frags->link != NULL && last->row == aid)
      base = last;
    else
      { re_fragment *f;

        for (f = ctg->frags; f != NULL; f = f->link)
          if (f->row == aid)
            { base = f;
              break;
            }
        if (f == NULL)
          { fprintf(stderr,"Could not find read %d in current multi-alignment (Re_Add_Overlap)\n",
                    aid);
             exit (1);
          }
      }
    last = base;
  }

  /* Determine the range [cstart,cfinis) of columns against which to realign
     and the left and right margin counts the new read.                       */

  { re_symbol *s;
    re_column *c;
    int        x;

    c = base->scol;
    s = base->next;
    x = 1;
    while (x <= o->path.abpos)
      { if (s->letter != '-')
          x += 1;
        c = c->succ;
        s = s->next;
      }
    while (s->letter == '-')
      { c = c->succ;
        s = s->next;
      }
    cstart = c;
    seglen = 1 + 2*bandsize;
    while (x <= o->path.aepos)
      { if (s->letter != '-')
          x += 1;
        c = c->succ;
        s = s->next;
        seglen += 1;
      }
    cfinis = c;

    lmargin = rmargin = 0;
    for (x = 0; x < bandsize; x++)  /* Pad the read's column range +/- bandSize */
      { if (cstart != ctg->first)
          cstart = cstart->pred;
        else
          lmargin += 1;
        if (cfinis != ctg->last)
          cfinis = cfinis->succ;
        else
          rmargin += 1;
      }
    if (cstart == ctg->first)
      { cstart   = cstart->succ;
        lmargin += 1;
      }
  }

  /* Setup the cost of comparing each symbol against a relevant column in score */

  { re_column *c;

    for (c = cstart; c != cfinis; c = c->succ)
      { int        i, cnt, max;

        cnt = c->depth - c->count[N_CODE];
        if (cnt != 0)
          { max = c->count[DASH_CODE];
            for (i = 0; i < DASH_CODE; i++)
              if (max < c->count[i])
                max = c->count[i];
            for (i = 0; i < N_CODE; ++i)
              { c->score[i] = MIX - (MIX * c->count[i])/cnt;
                if (c->count[i] != max)
                  c->score[i] += (1.-MIX);
              }
          }
        else if (c->count[N_CODE] > 0)
          { for (i = 0; i < N_CODE; i++)
              c->score[i] = 0.0;
          }
        else
          { for (i = 0; i < N_CODE; i++)
              c->score[i] = ALONE;
          }
        c->score[N_CODE] = 0.0;
      }
  }

#ifdef DEBUG_REALIGN
  { int        i;

    printf("\nFragment %d\n",o->bread);
    printf("                         ");
    for (i = o->path.bbpos; i < o->path.bepos; i++)
      printf("    %c",seq[i]);
    printf("\n");
    fflush(stdout);
  }
#endif

  /* Space for d.p. trace, d.p. vectors, and encoded b-sequence sufficient? */

  { blen = (o->path.bepos - o->path.bbpos)+1;

    seglen += 1;
    if (seglen*blen > dpmax)
      { dpmax  = SPACE_GROWTH_RATE*seglen*blen + INIT_MAX_FRAG_LENGTH;
        trace  = (char *) realloc(trace ,  sizeof(char)*dpmax);
      }
    seglen -= 1;
    if (blen > segmax)
      { segmax = SPACE_GROWTH_RATE*blen + INIT_MAX_FRAG_LENGTH;
        matrix = (double *) realloc(matrix,sizeof(double)*segmax*2);
        cseq   = (int *) realloc(cseq,sizeof(int)*segmax);
      }
  }

  /* Do the d.p. in the forward direction computing successive
     column vectors, delivering the final one in 'ncol'           */

  { int        i;
    re_column *c;

    ncol    = matrix;
    cbck    = trace;
    for (i = 0; i <= lmargin; i++)
      { ncol[i] = 0.;
        cbck[i] = INS;
      }
    for (i = lmargin+1; i < blen; i++)
      { ncol[i] = ncol[i-1]+1;
        cbck[i] = INS;
      }

    seq += (o->path.bbpos-1);
    for (i = 1; i <= blen; i++)
      cseq[i] = encode[(int)(seq[i])];

    lmargin = 2*bandsize-lmargin;
    for (c = cstart; c != cfinis; c = c->succ)
      { int        t;
        float     *m;
        double     x, n;
        double    *ccol, d;

        ccol = ncol;
        if (ccol == matrix)
          ncol = ccol + blen;
        else
          ncol = matrix;
        cbck = cbck + blen;

        m = c->score;
        d = m[DASH_CODE];

        if (lmargin > 0)
          { ncol[0] = 0;
            lmargin -= 1;
          }
        else
          ncol[0] = ccol[0] + d;
        cbck[0] = DEL;

        for (i = 1; i < blen; i++)
          { t = INS;
            n = ncol[i-1] + 1;
            if ((x = ccol[i-1] + m[cseq[i]]) <= n)
              { n = x; t = SUB; }
            if ((x = ccol[i] + d) < n)
              { n = x; t = DEL; }
            ncol[i] = n;
            cbck[i] = t;
          }

        *m = ncol[blen-1];

#ifdef DEBUG_REALIGN
        { for (i = 0; i <= MAX_CODE; i++)
            printf(" %3.1f",c->score[i]);
          printf(" : ");
          for (i = 0; i < blen; i++)
            printf(" %4.1f",ncol[i]);
          printf("\n  ");
          for (i = 0; i < blen; i++)
            printf(" %4d",cbck[i]);
          printf("\n");
        }
#endif

      }

    if (rmargin > 0)
      { for (i = blen-rmargin; i < blen; i++)
          if (ncol[i-1] < ncol[i])
            { ncol[i] = ncol[i-1];
              cbck[i] = INS;
            }
      }
  }

  /* Determine the column of the best path not ending in a deletion (mincol) */

  { int        i;
    double     minval;
    re_column *c;
    char      *t;

    c = cfinis;
    t = cbck;
    minval = 1.e100;
    for (i = 0; i <= 2*bandsize-rmargin; i++)
      { c = c->pred;
        if (minval > c->score[0] && t[blen-1] != DEL)
          { cbck   = t;
            minval = c->score[0];
            mincol = c;
#ifdef DEBUG_REALIGN
            printf("\nSelecting %d as best\n",i);
#endif
          }
        t -= blen;
      }
  }

  /* Trace back through matrix and interweave fragment
     always placing it at the bottom of each column       */

  { int          j;
    re_symbol   *s, *p;
    re_fragment *read;

    s = (re_symbol *) (read = new_fragment());
    s->letter = '\0';
    for (j = blen-1; j > 0; j--)
      { while (cbck[j] == DEL)
          {
            /* Weave a '-' into the column mincol for the read */

            p = new_symbol();
            p->next   = s;
            s->prev   = p;
            p->letter = '-';
            p->up     = mincol->up;
            p->down   = (re_symbol *) mincol;
            p->down->up = p->up->down = p;
            s = p;

            mincol->depth += 1;
            mincol->count[DASH_CODE] += 1;

            mincol = mincol->pred;
            cbck  -= blen;
          }

        if (cbck[j] == INS)
          { re_column *c;
            re_symbol *u, *t;

            /* Add a new column between mincol and its successor that
               contains the non-dash char of the read and a column of
               '-'s for each read not ending at the boundary          */

            c = new_column();
            c->pred = mincol;
            c->succ = mincol->succ;
            c->pred->succ = c->succ->pred = c;
            c->up = c->down = (re_symbol *) c;
            { int i;
              for (i = 0; i <= MAX_CODE; i++)
                c->count[i] = 0;
            }
            c->depth = 1;
            c->count[encode[s->letter]] += 1;

            /* Add a '-' to fragments that span mincol and its successor */

            for (t = mincol->up; t != (re_symbol *) mincol; t = t->up)
              if (t->next->letter != '\0')
                { u = new_symbol();
                  u->letter = '-';
                  u->prev   = t;
                  u->next   = t->next;
                  u->down   = c->down;
                  u->up     = (re_symbol *) c;
                  u->prev->next = u->next->prev = u;
                  u->up->down = u->down->up = u;

                  c->depth += 1;
                  c->count[DASH_CODE] += 1;
                }

            p = new_symbol();
            p->next   = s;
            s->prev   = p;
            p->letter = seq[j];
            p->up   = c->up;
            p->down = (re_symbol *) c;
            p->down->up = p->up->down = p;
            s = p;
          }
        else

          /* Weave the char of the read into column mincol */

          { p = new_symbol();
            p->next   = s;
            s->prev   = p;
            p->letter = seq[j];
            p->up   = mincol->up;
            p->down = (re_symbol *) mincol;
            p->down->up = p->up->down = p;
            s = p;

            mincol->depth += 1;
            mincol->count[encode[(int)(seq[j])]] += 1;

            mincol = mincol->pred;
            cbck  -= blen;
          }
      }

    read->scol = mincol->succ;
    read->row  = o->bread;
    read->next = s;
    s->prev = (re_symbol *) read;

    read->link = ctg->frags;
    ctg->frags = read;
  }
}

static re_contig   *Fetch_Contig  = NULL;
static re_fragment *Next_Fragment = NULL;
static re_column   *Next_Column   = NULL;

void Re_Start_Scan(Re_Contig *contig)
{ Fetch_Contig  = (re_contig *) contig;
  Next_Fragment = NULL;
  Next_Column   = NULL;
}

char *Re_Next_Read(int *id)
{ static char *read  = NULL;
  static int   fragmax = -1;

  if (Fetch_Contig == NULL)
    { fprintf(stderr,"No contig is set up to be scanned (re_next_fragment)\n");
      exit (1);
    }

  if (Next_Fragment == NULL)
    Next_Fragment = Fetch_Contig->frags;
  else
    Next_Fragment = Next_Fragment->link;

  if (Next_Fragment == NULL)
    return (NULL);

  { re_symbol *s;
    int        len;

    len = 0;
    for (s = Next_Fragment->next; s != (re_symbol *) Next_Fragment; s = s->next)
      len += 1;

    if (len+1 >= fragmax)
      { fragmax = SPACE_GROWTH_RATE*(len+1) + INIT_MAX_FRAG_LENGTH;
        read    = (char *) realloc(read,sizeof(char)*fragmax);
      }

    len = 0;
    for (s = Next_Fragment->next; s != (re_symbol *) Next_Fragment; s = s->next)
      read[len++] = s->letter;
    read[len] = '\0';
  }

  *id = Next_Fragment->row;
  return (read);
}

char *Re_Next_Column(int **ids)
{ static char       *column = NULL;
  static int        *idvect = NULL;
  static re_symbol **nextel = NULL;
  static int         colmax = -1;
  static int         lastlen;

  if (Fetch_Contig == NULL)
    { fprintf(stderr,"No contig is set up to be scanned (re_next_column)\n");
      exit (1);
    }

  if (Next_Column == NULL)
    { Next_Column = Fetch_Contig->first->succ;
      lastlen     = 0;
    }
  else
    Next_Column = Next_Column->succ;

  if (Next_Column == Fetch_Contig->last)
    { Next_Column = NULL;
      return (NULL);
    }

  { re_symbol *s;
    int        i, j, len;

    j = 0;
    for (i = 0; i < lastlen; i++)
      { s = nextel[i];
        if (s->letter == '\0') continue;
        s = s->next;
        nextel[j] = s;
        if (s->letter == '\0')
          column[j] = ' ';
        else
          column[j] = s->letter;
        idvect[j] = idvect[i];
        j += 1;
      }

    len = j;
    for (s = Next_Column->down; s != (re_symbol *) Next_Column; s = s->down)
      if (s->prev->letter == '\0')
        len += 1;

    if (len+1 >= colmax)
      { colmax = SPACE_GROWTH_RATE*(len+1) + INIT_MAX_FRAG_LENGTH;
        column = (char *) realloc(column,sizeof(char)*colmax);
        idvect = (int *)  realloc(idvect,sizeof(int)*colmax);
        nextel = (re_symbol **) realloc(nextel,sizeof(re_symbol *)*colmax);
      }

    for (s = Next_Column->down; s != (re_symbol *) Next_Column; s = s->down)
      if (s->prev->letter == '\0')
        { nextel[j] = s;
          column[j] = s->letter;
          idvect[j] = ((re_fragment *) (s->prev))->row;
          j += 1;
        }
    lastlen = j;
    column[j] = '\0';
  }

  *ids = idvect;
  return (column);
}

static void Re_Align_Fragment(re_fragment *read, int bandsize)
{ static double *matrix = NULL;
  static char   *trace  = NULL;
  static int     readmax = -1;

  int        bandwidth;        /* Bandwidth for realignment                                */
  re_column *cstart, *cfinis;  /* Column interval to realign against [cstart,cfinis)       */
  int        readlen;          /* Padded length of fragment (before realignment)           */

  double    *crow;             /* (bandwidth+1) vector of d.p. values for current row      */
  char      *cbck;             /* (bandwidth+1) vector of traceback values for current row */
  re_column *ccol;             /* Column of first char in current band                     */

  int        minpos;           /* Traceback position in band vector during realignment     */
  re_column *mincol;           /* Next column in traceback during realignment.             */

#define INS 0
#define SUB 1
#define DEL 2

  bandwidth = 2*bandsize;
  if (encode == NULL)
    setup_encode();

  /* Strip fragment from structure, compute its padded length, and
     determine the range of columns against which to realign it. */

  { re_symbol *s;
    re_column *c;
    int        i;

    readlen = 1;
    cstart  = c = read->scol;
    for (s = read->next; s != (re_symbol *) read; s = s->next)
      { s->up->down = s->down;
        s->down->up = s->up;
        c->depth -= 1;
        c->count[encode[s->letter]] -= 1;
        c = c->succ;
        readlen += 1;
      }
    cfinis = c;

    for (i = 0; i < bandsize; i++)  /* Pad the read's column range +/- BandSize */
      { cstart = cstart->pred;
        cfinis = cfinis->succ;
      }
  }

  /* Setup the cost of comparing each symbol against a relevant column in score */

  { re_column *c;
    int        pemp;

    pemp = (cstart->depth == 0);
    for (c = cstart; c != cfinis; c = c->succ)
      { int        i, cnt, max, nemp;

        cnt = c->depth - c->count[N_CODE];
        if (cnt != 0)
          { max = c->count[DASH_CODE];
            for (i = 0; i < DASH_CODE; i++)
              if (max < c->count[i])
                max = c->count[i];
            for (i = 0; i < N_CODE; ++i)
              { c->score[i] = MIX - (MIX * c->count[i])/cnt;
                if (c->count[i] != max)
                  c->score[i] += (1.-MIX);
              }
          }
        else if (c->count[N_CODE] > 0)
          { for (i = 0; i < N_CODE; i++)
              c->score[i] = 0.0;
          }
        else
          { for (i = 0; i < N_CODE; i++)
              c->score[i] = ALONE;
          }
        c->score[N_CODE] = 0.0;

        nemp = (c->succ->depth == 0);
        c->bound = (pemp || nemp);
        pemp = nemp;
      }
  }

#ifdef DEBUG_REALIGN
  { int        i;
    re_column *c;

    printf("\nFragment %p\n",read);
    for (c = cstart; c != cfinis; c = c->succ)
      { printf("   ");
        for (i = 0; i <= MAX_CODE; i++)
          printf(" %3d",c->count[i]);
        printf("  :  ");
        for (i = 0; i <= MAX_CODE; i++)
          printf(" %3.1f",c->score[i]);
        printf(" (%d)\n",c->bound);
      }
    fflush(stdout);
  }
#endif

  /* Space for d.p. band sufficient? */

  if (readlen > readmax)
    { int newsize;
      readmax = SPACE_GROWTH_RATE*readlen + INIT_MAX_FRAG_LENGTH;
      newsize = readmax*(bandwidth+1);
      matrix  = (double *) realloc(matrix,sizeof(double)*newsize);
      trace   = (char   *) realloc(trace ,  sizeof(char)*newsize);
    }

  /* Do the d.p. in the forward direction computing successive
     band vectors, delivering the final one in 'crow'           */

  { int        i;
    re_symbol *s;

    crow = matrix;
    for (i = 0; i <= bandwidth; i++)
      crow[i] = 0.;
    cbck = trace;
    ccol = cstart;
    for (s = read->next; s != (re_symbol *) read; s = s->next)
      { int        e, t;
        double     x, n;
        re_column *c;
        double    *nrow;

        nrow = crow + (bandwidth+1);
        cbck = cbck + (bandwidth+1);

        e = encode[s->letter];
        c = ccol;

        t = INS;
        if (e >= DASH_CODE)
          n = crow[1];
        else if (c->bound)
          n = crow[1] + ALONE;
        else
          n = crow[1] + 1;
        if ((x = crow[0] + c->score[e]) <= n)
          { n = x; t = SUB; }
        nrow[0] = n;
        cbck[0] = t;

        for (i = 1; i < bandwidth; i++)
          { c = c->succ;

            t = INS;
            if (e >= DASH_CODE)
              n = crow[i+1];
            else if (c->bound)
              n = crow[i+1] + ALONE;
            else
              n = crow[i+1] + 1;
            if ((x = crow[i] + c->score[e]) <= n)
              { n = x; t = SUB; }
            if ((x = nrow[i-1] + c->score[DASH_CODE]) < n)
              { n = x; t = DEL; }
            nrow[i] = n;
            cbck[i] = t;
          }

        c = c->succ;

        t = SUB;
        n = crow[bandwidth] + c->score[e];
        if ((x = nrow[bandwidth-1] + c->score[DASH_CODE]) < n)
          { n = x; t = DEL; }
        nrow[bandwidth] = n;
        cbck[bandwidth] = t;

#ifdef DEBUG_REALIGN
        { re_symbol *t;
          printf(" %c:",s->letter);
#ifdef DEBUG_SMALL
          for (t = (re_symbol *) read; t != s; t = t->next)
            printf("    ");
#endif
          for (i = 0; i <= bandwidth; i++)
            printf(" %4.1f",nrow[i]);
          printf("\n  ");
#ifdef DEBUG_SMALL
          for (t = (re_symbol *) read; t != s; t = t->next)
            printf("    ");
#endif
          for (i = 0; i <= bandwidth; i++)
            printf(" %4d",cbck[i]);
          printf("\n");
        }
#endif

        ccol = ccol->succ;
        crow   = nrow;
      }
  }

  /* Determine the endpoint of the best path not ending in a
     deletion: index + column pointer (minpos, mincol)        */

  { int i, minval;

    minval = 0x7FFFFFFF;
    for (i = bandsize; i >= 0; i--)
      if (minval > crow[i] && cbck[i] != DEL)
        { minpos = i;
          minval = crow[i];
        }
    for (i = bandsize+1; i <= bandwidth; i++)
      if (minval > crow[i] && cbck[i] != DEL)
        { minpos = i;
          minval = crow[i];
        }
    mincol = ccol->pred;
    for (i = 0; i < minpos; i++)
      mincol = mincol->succ;
  }

#ifdef DEBUG_REALIGN
  printf("\nSelected %d as best\n",minpos);
#endif

  /* Trace back through matrix and interweave fragment
     always placing it at the bottom of each column       */

  { re_symbol *s;

    for (s = read->prev; s != (re_symbol *) read; s = s->prev)
      { while (cbck[minpos] == DEL)
          { re_symbol *p;

            /* Weave a '-' into the column mincol for the read */

            p = new_symbol();
            p->prev   = s;
            p->next   = s->next;
            p->letter = '-';
            p->up     = mincol->up;
            p->down   = (re_symbol *) mincol;
            p->down->up = p->up->down = p;
            p->next->prev = p->prev->next = p;

            mincol->depth += 1;
            mincol->count[DASH_CODE] += 1;

            mincol = mincol->pred;
            minpos -= 1;
          }

        if (cbck[minpos] == INS)
          { if (s->letter == '-')
              { re_symbol *p;

                /* Remove the '-' from the read (rather
                   than align it against a new column of '-'s)  */

                p = s;
                s = s->next;
                s->prev = p->prev;
                p->prev->next = s;
                free_symbol(p);
              }
            else
              { re_column *c;
                re_symbol *u, *t;

                /* Add a new column between mincol and its successor that
                   contains the non-dash char of the read and a column of
                   '-'s for each read not ending at the boundary          */

                c = new_column();
                c->pred = mincol;
                c->succ = mincol->succ;
                c->pred->succ = c->succ->pred = c;
                c->up = c->down = (re_symbol *) c;
                { int i;
                  for (i = 0; i <= MAX_CODE; i++)
                    c->count[i] = 0;
                }
                c->depth = 1;
                c->count[encode[s->letter]] += 1;

                /* Add a '-' to fragments that span mincol and its successor */

                for (t = mincol->up; t != (re_symbol *) mincol; t = t->up)
                  if (t->next->letter != '\0')
                    { u = new_symbol();
                      u->letter = '-';
                      u->prev   = t;
                      u->next   = t->next;
                      u->down   = c->down;
                      u->up     = (re_symbol *) c;
                      u->prev->next = u->next->prev = u;
                      u->up->down = u->down->up = u;

                      c->depth += 1;
                      c->count[DASH_CODE] += 1;
                    }

                s->up   = c->up;
                s->down = (re_symbol *) c;
                s->down->up = s->up->down = s;
              }
            minpos += 1;
          }
        else

          /* Weave the char of the read into column mincol */

          { s->up   = mincol->up;
            s->down = (re_symbol *) mincol;
            s->down->up = s->up->down = s;

            mincol->depth += 1;
            mincol->count[encode[s->letter]] += 1;

            mincol = mincol->pred;
          }
        cbck -= (bandwidth+1);
      }

    read->scol = mincol->succ;
  }
}

int Re_Consensus_Score(Re_Contig *contig)
{ int        i, max, score;
  re_column *c;

  score = 0;
  for (c = ((re_contig *) contig)->first->succ; c != NULL; c = c->succ)
    { max = c->count[DASH_CODE];
      for (i = 0; i < DASH_CODE; i++)
        if (c->count[i] > max)
          max = c->count[i];
      score += (c->depth-max);
    }
  return (score);
}

#ifdef STATS
static int num_contigs;
static int total_iterations;
#endif

/* Return how much the consensus score was improved by */

int Re_Align_Contig(Re_Contig *contig, int bandsize)
{ static re_column* bprelft = NULL;
  static re_column *bprergt, *bsufrgt, *bsuflft;

  re_contig *ctg = (re_contig *) contig;

  int score, oldscore, original;

  /* If first time, create column padding for each end of consensus */

  if (bprelft == NULL)
    { int        j;
      re_column *c;

      bprelft = (re_column *) malloc(sizeof(re_column)*2*bandsize);
      bprergt = bprelft + (bandsize-1);
      bsuflft = bprergt + 1;
      bsufrgt = bsuflft + (bandsize-1);
      for (c = bprelft; c <= bsufrgt; c++)
        { c->succ = c+1;
          c->pred = c-1;
          c->down = c->up = (re_symbol *) c;
          for (j = 0; j <= MAX_CODE; j++)
            c->count[j] = 0;
          c->depth = 0;
        }
   }

  /* Splice in the border padding (Bandwidth empty columns to simplify boundary cases) */

  bprergt->succ = ctg->first->succ;
  bsuflft->pred = ctg->last->pred;
  bprergt->succ->pred = bprergt;
  bsuflft->pred->succ = bsuflft;
  ctg->first->succ = bprelft;
  bprelft->pred = ctg->first;
  ctg->last->pred = bsufrgt;
  bsufrgt->succ = ctg->last;

  /* While score improves do realignment passes */

  original = Re_Consensus_Score(ctg);
  score    = original;
  oldscore = score+1;
#ifdef DEBUG
  printf("\nInitial score %d\n\n",score);
#endif
  while (oldscore > score)
    {
      /* Realign every fragment (in reverse order of list) */

      { re_fragment *f;

        for (f = ctg->frags; f != NULL; f = f->link)
          if (f->next != (re_symbol *) f)
            { Re_Align_Fragment(f,bandsize);

#ifdef DEBUG_REALIGN
              printf("\nFragment = %p\n",f);
              Re_Print_Structure(ctg,stdout);
#endif
            }
      }

      /* If fragments got realigned with part of the border then re-establish it */

      { re_column *a, *b, *c;
        re_symbol *s;
        int        j;

        c = bprergt->succ;
        for (b = bprergt; b->depth > 0; b = b->pred)  /* Initial border */
          { a = new_column();
            *a = *b;
            a->down->up = a->up->down = (re_symbol *) a;
            a->succ = c;
            c->pred = a;
            c = a;
            b->up = b->down = (re_symbol *) b;
            for (j = 0; j <= MAX_CODE; j++)
              b->count[j] = 0;
            b->depth = 0;
            for (s = a->down; s != (re_symbol *) a; s = s->down)
              if (s->prev->letter == '\0')
                ((re_fragment *) (s->prev))->scol = a;
          }
        bprergt->succ = c;
        c->pred = bprergt;

        c = bsuflft->pred;
        for (b = bsuflft; b->depth > 0; b = b->succ)  /* Tail border */
          { a = new_column();
            *a = *b;
            a->down->up = a->up->down = (re_symbol *) a;
            a->pred = c;
            c->succ = a;
            c = a;
            b->up = b->down = (re_symbol *) b;
            for (j = 0; j <= MAX_CODE; j++)
              b->count[j] = 0;
            b->depth = 0;
          }
        bsuflft->pred = c;
        c->succ = bsuflft;
      }

      /* Remove blank columns */

      { re_column *c, *d;

        for (c = bprergt->succ; c != bsuflft; c = d)
          { d = c->succ;
            if (c->depth == c->count[DASH_CODE])
              { re_symbol *s;

                c->pred->succ = d;
                d->pred = c->pred;
                for (s = c->down; s != (re_symbol *) c; s = s->down)
                  { if (s->prev->letter == '\0')
                      ((re_fragment *) (s->prev))->scol = d;
                    s->prev->next = s->next;
                    s->next->prev = s->prev;
                    free_symbol(s);
                  }
                free_column(c);
              }
          }
      }

      oldscore = score;
      score    = Re_Consensus_Score(ctg);

#ifdef DEBUG
      printf("\nCycle: new score = %d\n",score);
      Re_Print_Structure(ctg,stdout);
#endif

#ifdef STATS
      total_iterations += 1;
#endif
    }

  /* Restore regular column boundary */

  bprergt->succ->pred = ctg->first;
  bsuflft->pred->succ = ctg->last;
  ctg->first->succ = bprergt->succ;
  ctg->last->pred  = bsuflft->pred;

  return (original - score);
}
