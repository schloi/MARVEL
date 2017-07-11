/*****************************************************************************************\
*                                                                                         *
*  This is the interface library for a restructured version of realigner                  *
*                                                                                         *
*  Author:  Gene Myers                                                                    *
*  Date  :  March 2007                                                                    *
*                                                                                         *
\*****************************************************************************************/

#ifndef _REALIGN_MODULE

#define _REALIGN_MODULE

#include "dalign/align.h"

typedef void Re_Contig;

  // Read and write a contig in vertical format without consensus or dots (see converter)

Re_Contig *Re_Read_Contig(FILE *input, FILE *comments);
void       Re_Print_Contig(Re_Contig *contig, FILE *output, int samerows);

  // Debug routines that print a representation of the model (symbols only)
  //   and the entire data structure

void Re_Print_Model(Re_Contig *contig, FILE *output);
void Re_Print_Structure(Re_Contig *contig, FILE *output);

  // Return the consensus score of a contig

int Re_Consensus_Score(Re_Contig *contig);

  // Optimize the alignment of a contig, using round-robin realignment in a band of size bandsize

int Re_Align_Contig(Re_Contig *contig, int bandsize);

  // Free the storage for a contig

void Re_Free_Contig(Re_Contig *contig);

  /* Routines for building multi-alignments from layout trees of overlaps.
       Start by calling re_start_contig with the sequence, seq, and id, aread, of the root.
       Then if you want the multi-alignment of aread with every read segment that overlaps
         it, call re_add_overlap with each overlap o and the sequence of the bread, bseq.
       If you want an intial alignment of a layout tree, then call re_add_read with a preorder
         traversal of the overlaps in the tree.  Because the tree is a layout tree, it is true
         that the bread of overlap o, either is contained within the aread, or the aread extends
         to the current right end of the growing multi-alignment.  If reads are not added
         in such an order, then the routine emits an error message and terminates execution.    */

Re_Contig *Re_Start_Contig(int aread, char *aseq);
void       Re_Add_Overlap(Re_Contig *contig, Overlap *o, char *bseq, int bandsize);
void       Re_Add_Read   (Re_Contig *contig, Overlap *o, char *bseq, int bandsize);

  /* Routines for getting columns or dashed-reads out of a multi-alignment.
     re_start_scan sets starts a "scan" of a given contig, each subsequent call
     to re_next_read or re_next_column gets the next read or column of the
     multi-alignment.  re_next_read returns a pointer to the sequence of a read
     wherein the dashes used to align it in the multi-alignment *are present*.  The
     integer pointed at by id is set to the id stored for the read (or row # if it
     was read in from a vertical multi-alignment).  re_next_column returns a pointer the
     sequence of the next column, including dash symbols.  The integer array pointed at
     by ids gives the id of each read in the return sequence and is of the same length as
     that sequencee.  When the end of a read is reached, a '\n' is output for it in the
     column just to the right of the end of the sequence.  When a read is introduced into
     a column it is always introduced at the end.                                           */

void  Re_Start_Scan (Re_Contig *contig);
char *Re_Next_Read  (int *id);
char *Re_Next_Column(int **ids);

#endif  // _REALIGN_MODULE
