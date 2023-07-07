
#ifndef _A_MODULE

#define _A_MODULE

#include "db/DB.h"
#include "types.h"
#include <inttypes.h>

#define TRACE_XOVR 125   //  If the trace spacing is not more than this value, then can
                         //    and do compress traces pts to 8-bit unsigned ints



/*** ALIGNMENT ABSTRACTION:

     An alignment is modeled by an Alignment record, which in addition to a *pointer* to a
     'path', gives pointers to the A and B sequences, their lengths, and indicates whether
     the B-sequence needs to be complemented ('comp' non-zero if so).  The 'trace' pointer
     of the 'path' subrecord can be either NULL, a list of pass-through points, or an exact
     trace depending on what routines have been called on the record.

     One can (1) compute a trace, with Compute_Trace, either from scratch if 'path.trace' = NULL,
     or using the sequence of pass-through points in trace, (2) print an ASCII representation
     of an alignment, or (3) reverse the roles of A and B, and (4) complement a sequence
     (which is a reversible process).

     If the alignment record shows the B sequence as complemented, *** THEN IT IS THE
     RESPONSIBILITY OF THE CALLER *** to make sure that bseq points at a complement of
     the sequence before calling Compute_Trace or Print_Alignment.  Complement_Seq complements
     the sequence a of length n.  The operation does the complementation/reversal in place.
     Calling it a second time on a given fragment restores it to its original state.

     With the introduction of the DAMAPPER, we need to code chains of alignments between a
     pair of sequences.  The alignments of a chain are expected to be found in order either on
     a file or in memory, where the START_FLAG marks the first alignment and the NEXT_FLAG all
     subsequent alignmenst in a chain.  A chain of a single LA is marked with the START_FLAG.
     The BEST_FLAG marks one of the best chains for a pair of sequences.  The convention is
     that either every record has either a START- or NEXT-flag, or none of them do (e.g. as
     produced by daligner), so one can always check the flags of the first alignment to see
     whether or not the chain concept applies to a given collection or not.
***/

#define COMP_FLAG  0x1
#define ACOMP_FLAG 0x2   //  A-sequence is complemented, not B !  Only Local_Alignment notices

#define COMP(x)   ((x) & COMP_FLAG)
#define ACOMP(x)  ((x) & ACOMP_FLAG)

#define START_FLAG 0x4   //  LA is the first of a chain of 1 or more la's
#define NEXT_FLAG  0x8   //  LA is the next segment of a chain.
#define BEST_FLAG  0x10  //  This is the start of the best chain

#define CHAIN_START(x)  ((x) & START_FLAG)
#define CHAIN_NEXT(x)   ((x) & NEXT_FLAG)
#define BEST_CHAIN(x)   ((x) & BEST_FLAG)

#define ELIM_FLAG  0x20  //  This LA should be ignored

#define ELIM(x)  ((x) & ELIM_FLAG)

typedef struct
  { Path   *path;
    uint32_t  flags;        /* Pipeline status and complementation flags          */
    char   *aseq;         /* Pointer to A sequence                              */
    char   *bseq;         /* Pointer to B sequence                              */
    int     alen;         /* Length of A sequence                               */
    int     blen;         /* Length of B sequence                               */

    size_t pathBufferOffset;
    int aread;
    int bread;
  } Alignment;

void Complement_Seq(char *a, int n);

  /* Many routines like Local_Alignment, Compute_Trace, and Print_Alignment need working
     storage that is more efficiently reused with each call, rather than being allocated anew
     with each call.  Each *thread* can create a Work_Data object with New_Work_Data and this
     object holds and retains the working storage for routines of this module between calls
     to the routines.  If enough memory for a Work_Data is not available then NULL is returned.
     Free_Work_Data frees a Work_Data object and all working storage held by it.
  */

  Work_Data *New_Work_Data();

  void       Free_Work_Data(Work_Data *work);

  /* Local_Alignment seeks local alignments of a quality determined by a number of parameters.
     These are coded in an Align_Spec object that can be created with New_Align_Spec and
     freed with Free_Align_Spec when no longer needed.  There are 4 essential parameters:

     ave_corr:    the average correlation (1 - 2*error_rate) for the sought alignments.  For Pacbio
                    data we set this to .70 assuming an average of 15% error in each read.
     trace_space: the spacing interval for keeping trace points and segment differences (see
                    description of 'trace' for Paths above)
     freq[4]:     a 4-element vector where afreq[0] = frequency of A, f(A), freq[1] = f(C),
                    freq[2] = f(G), and freq[3] = f(T).  This vector is part of the header
                    of every DAZZ database (see db.h).
     reach:       a boolean, if set alignment extend to the boundary when reasonable, otherwise
                    the terminate only at suffix-positive points.

     If an alignment cannot reach the boundary of the d.p. matrix with this condition (i.e.
     overlap), then the last/first 30 columns of the alignment are guaranteed to be
     suffix/prefix positive at correlation ave_corr * g(freq) where g is an empirically
     measured function that increases from 1 as the entropy of freq decreases.  If memory is
     unavailable or the freq distribution is too skewed then NULL is returned.

     You can get back the original parameters used to create an Align_Spec with the simple
     utility functions below.
  */

  Align_Spec *New_Align_Spec(double ave_corr, int trace_space, float *freq, int reach, int nthreads);

  void        Free_Align_Spec(Align_Spec *spec);

  int    Trace_Spacing      (Align_Spec *spec);
  double Average_Correlation(Align_Spec *spec);
  float *Base_Frequencies   (Align_Spec *spec);
  int    Overlap_If_Possible(Align_Spec *spec);

  /* Local_Alignment finds the longest significant local alignment between the sequences in
     'align' subject to:

       (a) the alignment criterion given by the Align_Spec 'spec',
       (b) it passes through one of the points (anti+k)/2,(anti-k)/2 for k in [low,hgh] within
             the underlying dynamic programming matrix (i.e. the points on diagonals low to hgh
             on anti-diagonal anti or anti-1 (depending on whether the diagonal is odd or even)),
       (c) if lbord >= 0, then the alignment is always above diagonal low-lbord, and
       (d) if hbord >= 0, then the alignment is always below diagonal hgh+hbord.

     The path record of 'align' has its 'trace' filled from the point of view of an overlap
     between the aread and the bread.  In addition a Path record from the point of view of the
     bread versus the aread is returned by the function, with this Path's 'trace' filled in
     appropriately.  The space for the returned path and the two 'trace's are in the working
     storage supplied by the Work_Data packet and this space is reused with each call, so if
     one wants to retain the bread-path and the two trace point sequences, then they must be
     copied to user-allocated storage before calling the routine again.  NULL is returned in
     the event of an error.

     Find_Extension is a variant of Local_Alignment that simply finds a local alignment that
     either ends (if prefix is non-zero) or begins (if prefix is zero) at the point
     (anti+diag)/2,(anti-diag)/2).  All other parameters are as before.  It returns a non-zero
     value only when INTERACTIVE is on and it cannot allocate the memory it needs.
     Only the path and trace with respect to the aread is returned.  This routine is experimental
     and may not persist in later versions of the code.
  */

  Path *Local_Alignment(Alignment *align, Work_Data *work, Align_Spec *spec,
                        int low, int hgh, int anti, int lbord, int hbord);

  int   Find_Extension(Alignment *align, Work_Data *work, Align_Spec *spec,    //  experimental !!
                       int diag, int anti, int lbord, int hbord, int prefix);

  /* Given a legitimate Alignment object and associated trace point vector in 'align->path.trace',
     Compute_Trace_X, computes an exact trace for the alignment and resets 'align->path.trace'
     to point at an integer array within the storage of the Work_Data packet encoding an
     exact optimal trace from the start to end points.  If the trace is needed beyond the
     next call to a routine that sets it, then it should be copied to an array allocated
     and managed by the caller.

     Compute_Trace_PTS computes a trace by computing the trace between successive trace points.
     It is much, much faster than Compute_Alignment below but at the tradeoff of not necessarily
     being optimal as pass-through points are not all perfect.  Compute_Trace_MID computes a trace
     by computing the trace between the mid-points of alignments between two adjacent pairs of trace
     points.  It is generally twice as slow as Compute_Trace_PTS, but it produces nearer optimal
     alignments.  Both these routines return 1 if an error occurred and 0 otherwise.
  */

#define LOWERMOST -1   //   Possible modes for "mode" parameter below)
#define GREEDIEST  0
#define UPPERMOST  1

  int Compute_Trace_PTS(Alignment *align, Work_Data *work, int trace_spacing, int mode);
  int Compute_Trace_MID(Alignment *align, Work_Data *work, int trace_spacing, int mode);

  /* Compute_Trace_IRR (IRR for IRRegular) computes a trace for the given alignment where
     it assumes the spacing between trace points between both the A and B read varies, and
     futher assumes that the A-spacing is given in the short integers normally occupied by
     the differences in the alignment between the trace points.  This routine is experimental
     and may not persist in later versions of the code.
  */

  int Compute_Trace_IRR(Alignment *align, Work_Data *work, int mode);   //  experimental !!

  /* Compute Alignment determines the best alignment between the substrings specified by align.
     If the task is DIFF_ONLY, then only the difference of this alignment is computed and placed
     in the "diffs" field of align's path.  If the task is PLUS_TRACE or DIFF_TRACE, then
     'path.trace' is set to point at an integer array within the storage of the Work_Data packet
     encoding a trace point sequence for an optimal alignment, whereas if the task is PLUS_ALIGN
     or DIFF_ALIGN, then it points to an optimal trace of an optimatl alignment.  The PLUS
     tasks can only be called if the immmediately proceeding call was a DIFF_ONLY on the same
     alignment record and sequences, in which case a little efficiency is gained by avoiding
     the repetition of the top level search for an optimal mid-point.
  */

#define PLUS_ALIGN   0
#define PLUS_TRACE   1
#define DIFF_ONLY    2
#define DIFF_ALIGN   3
#define DIFF_TRACE   4

  int Compute_Alignment(Alignment *align, Work_Data *work, int task, int trace_spacing);

  /* Alignment_Cartoon prints an ASCII representation of the overlap relationhip between the
     two reads of 'align' to the given 'file' indented by 'indent' space.  Coord controls
     the display width of numbers, it must be not less than the width of any number to be
     displayed.

     If the alignment trace is an exact trace, then one can ask Print_Alignment to print an
     ASCII representation of the alignment 'align' to the file 'file'.  Indent the display
     by "indent" spaces and put "width" columns per line in the display.  Show "border"
     characters of sequence on each side of the aligned region.  If upper is non-zero then
     display bases in upper case.  If coord is greater than 0, then the positions of the
     first character in A and B in the given row is displayed with a field width given by
     coord's value.

     Print_Reference is like Print_Alignment but rather than printing exaclty "width" columns
     per segment, it prints "block" characters of the A sequence in each segment.  This results
     in segments of different lengths, but is convenient when looking at two alignments involving
     A as segments are guaranteed to cover the same interval of A in a segment.

     Both Print routines return 1 if an error occurred (not enough memory), and 0 otherwise.

     Flip_Alignment modifies align so the roles of A and B are reversed.  If full is off then
     the trace is ignored, otherwise the trace must be to a full alignment trace and this trace
     is also appropriately inverted.
  */

  void Alignment_Cartoon(FILE *file, Alignment *align, int indent, int coord);

  int  Print_Alignment(FILE *file, Alignment *align, Work_Data *work,
                       int indent, int width, int border, int upper, int coord);

  int  Print_Reference(FILE *file, Alignment *align, Work_Data *work,
                       int indent, int block, int border, int upper, int coord);

  void Flip_Alignment(Alignment *align, int full);


  /* Read_Overlap reads the next Overlap record from stream 'input', not including the trace
     (if any), and without modifying 'ovl's trace pointer.  Read_Trace reads the ensuing trace
     into the memory pointed at by the trace field of 'ovl'.  It is assumed to be big enough to
     accommodate the trace where each value take 'tbytes' bytes (1 if uint8 or 2 if uint16).

     Write_Overlap write 'ovl' to stream 'output' followed by its trace vector (if any) that
     occupies 'tbytes' bytes per value.  It returns non-zero if there was an error writing.

     Print_Overlap prints an ASCII version of the contents of 'ovl' to stream 'output'
     where the trace occupes 'tbytes' per value and the print out is indented from the left
     margin by 'indent' spaces.

     Compress_TraceTo8 converts a trace fo 16-bit values to 8-bit values in place, and
     Decompress_TraceTo16 does the reverse conversion.  If check is set in a call to Compress
     then it checks whether the values fit in 8-bits, and if not returns a non-zero result
     in interactive mode, or exits with an error message in batch mode.

     Check_Trace_Points checks that the number of trace points is correct and that the sum
     of the b-read displacements equals the b-read alignment interval, assuming the trace
     spacing is 'tspace'.  It reports an error message if there is a problem and 'verbose'
     is non-zero.  The 'ovl' came from the file names 'fname'.
  */

  int Read_Overlap(FILE *input, Overlap *ovl);
  int Read_Trace(FILE *innput, Overlap *ovl, int tbytes);

  int  Write_Overlap(FILE *output, Overlap *ovl, int tbytes);
  void Print_Overlap(FILE *output, Overlap *ovl, int tbytes, int indent);

  int  Compress_TraceTo8(Overlap *ovl, int check);
  void Decompress_TraceTo16(Overlap *ovl);

  int  Check_Trace_Points(Overlap *ovl, int tspace, int verbose, char *fname);

#endif // _A_MODULE
