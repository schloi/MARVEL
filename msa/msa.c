
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include "lib/colors.h"
#include "msa.h"

#define CONS_MIN_COV_FRACTION 0.5 // base calling threshold

typedef struct
{
    char* B;              // sequence that gets aligned to
    msa_profile_entry* A; // the profile B

    int abpos, aepos; // regions to be aligned in A
    int bbpos, bepos; // in B

    int* ptp;
    int ptpmax;
    int ptpcur;

    int* pPtPoints; // pairs of pass-through points <profile, sequence>
    int nPtPoints;  // number of pass-through points ( pairs * 2 )

    int diffs_a; // number of insertions in A
    int diffs_b; // number of insertions in B

    msa_alignment_ctx* pCtx; // points to context, working storage
} msa_alignment;

static int match( char* a, int ia, msa_profile_entry* p, int ip )
{
    msa_profile_entry* pEntry = ( p + ip );

    char nMax = pEntry->counts[ 0 ];
    char cMax = 0;

    char nCount;

    // loop unrolled

    nCount = pEntry->counts[ 1 ];

    if ( nCount > nMax )
    {
        nMax = nCount;
        cMax = 1;
    }

    nCount = pEntry->counts[ 2 ];

    if ( nCount > nMax )
    {
        nMax = nCount;
        cMax = 2;
    }

    nCount = pEntry->counts[ 3 ];

    if ( nCount > nMax )
    {
        nMax = nCount;
        cMax = 3;
    }

    nCount = pEntry->counts[ 4 ];

    if ( nCount >= nMax )
    {
        return 0;
    }

    return ( cMax == a[ ia ] );
}

static int v3_align_onp( msa_alignment* pAlign,
                         msa_profile_entry* A, int M, char* B, int N )
{
    int** PVF = pAlign->pCtx->PVF; // wave->PVF;
    int** PHF = pAlign->pCtx->PHF; // wave->PHF;
    int D;
    int del = M - N;

    {
        int *F0, *F1, *F2;
        int* HF;
        int low, hgh;

        if ( del >= 0 )
        {
            low = 0;
            hgh = del;
        }
        else
        {
            low = del;
            hgh = 0;
        }

        F1 = PVF[ -2 ];
        F0 = PVF[ -1 ];

        for ( D = low - 1; D <= hgh + 1; D++ )
        {
            F1[ D ] = F0[ D ] = -2;
        }

        F0[ 0 ] = -1;

        low += 1;
        hgh -= 1;

        for ( D = 0; 1; D += 1 )
        {
            int k, i, j;
            int am, ac, ap;
            msa_profile_entry* a;

            F2 = F1;
            F1 = F0;
            F0 = PVF[ D ];
            HF = PHF[ D ];

            if ( ( D & 0x1 ) == 0 )
            {
                hgh += 1;
                low -= 1;
            }

            F0[ hgh + 1 ] = F0[ low - 1 ] = -2;

#define FS_MOVE( mdir, pdir )                  \
    ac = F1[ k ] + 1;                          \
    if ( ac < am )                             \
        if ( ap < am )                         \
        {                                      \
            HF[ k ] = mdir;                    \
            j       = am;                      \
        }                                      \
        else                                   \
        {                                      \
            HF[ k ] = pdir;                    \
            j       = ap;                      \
        }                                      \
    else if ( ap < ac )                        \
    {                                          \
        HF[ k ] = 0;                           \
        j       = ac;                          \
    }                                          \
    else                                       \
    {                                          \
        HF[ k ] = pdir;                        \
        j       = ap;                          \
    }                                          \
                                               \
    if ( N < i )                               \
        while ( j < N && match( B, j, a, j ) ) \
            j += 1;                            \
    else                                       \
        while ( j < i && match( B, j, a, j ) ) \
            j += 1;                            \
    F0[ k ] = j;

            j = -2;
            a = A + hgh;
            i = M - hgh;

            for ( k = hgh; k > del; k-- )
            {
                ap = j + 1;
                am = F2[ k - 1 ];
                FS_MOVE( -1, 4 )
                a -= 1;
                i += 1;
            }

            j = -2;
            a = A + low;
            i = M - low;

            for ( k = low; k < del; k++ )
            {
                ap = F2[ k + 1 ] + 1;
                am = j;
                FS_MOVE( 2, 1 )
                a += 1;
                i -= 1;
            }

            ap = F0[ del + 1 ] + 1;
            am = j;
            FS_MOVE( 2, 4 )

            if ( F0[ del ] >= N )
            {
                break;
            }
        }
    }

    {
        int k, h, m, e, c;
        msa_profile_entry* a;
        int ap = ( pAlign->pCtx->Aabs - A ) - 1; // (wave->Aabs - A) - 1;
        int bp = ( B - pAlign->pCtx->Babs ) + 1; // (B - wave->Babs) + 1;

        PHF[ 0 ][ 0 ] = 3;

        c             = N;
        k             = del;
        e             = PHF[ D ][ k ];
        PHF[ D ][ k ] = 3;

        while ( e != 3 )
        {
            h = k + e;

            if ( e > 1 )
            {
                h -= 3;
            }
            else if ( e == 0 )
            {
                D -= 1;
            }
            else
            {
                D -= 2;
            }

            if ( h < k ) // => e = -1 or 2
            {
                a = A + k;

                if ( k < 0 )
                {
                    m = -k;
                }
                else
                {
                    m = 0;
                }

                if ( PVF[ D ][ h ] <= c )
                {
                    c = PVF[ D ][ h ] - 1;
                }

                while ( c >= m && match( B, c, a, c ) ) // a[c] == B[c])
                {
                    c -= 1;
                }

                if ( e < 1 ) //  => edge is 2, others are 1, and 0
                {
                    if ( c <= PVF[ D + 2 ][ k + 1 ] )
                    {
                        e = 4;
                        h = k + 1;
                        D = D + 2;
                    }
                    else if ( c == PVF[ D + 1 ][ k ] )
                    {
                        e = 0;
                        h = k;
                        D = D + 1;
                    }
                    else
                    {
                        PVF[ D ][ h ] = c + 1;
                    }
                }
                else //   => edge is 0, others are 1, and 2 (if k != del), 0 (otherwise)
                {
                    if ( k == del )
                    {
                        m = D;
                    }
                    else
                    {
                        m = D - 2;
                    }

                    if ( c <= PVF[ m ][ k + 1 ] )
                    {
                        if ( k == del )
                        {
                            e = 4;
                        }
                        else
                        {
                            e = 1;
                        }

                        h = k + 1;
                        D = m;
                    }
                    else if ( c == PVF[ D - 1 ][ k ] )
                    {
                        e = 0;
                        h = k;
                        D = D - 1;
                    }
                    else
                    {
                        PVF[ D ][ h ] = c + 1;
                    }
                }
            }

            m             = PHF[ D ][ h ];
            PHF[ D ][ h ] = e;
            e             = m;

            k = h;
        }

        k = D = 0;
        e     = PHF[ D ][ k ];

        while ( e != 3 )
        {
            h = k - e;
            c = PVF[ D ][ k ];

            if ( e > 1 )
            {
                h += 3;
            }
            else if ( e == 0 )
            {
                D += 1;
            }
            else
            {
                D += 2;
            }

            if ( h > k )
            {
                *( pAlign->pCtx->Stop ) = bp + c;
                ( pAlign->pCtx->Stop )++;

                pAlign->diffs_b++;

                // *wave->Stop++ = bp + c;
            }
            else if ( h < k )
            {
                *( pAlign->pCtx->Stop ) = ap - ( c + k );
                ( pAlign->pCtx->Stop )++;

                pAlign->diffs_a++;

                // *wave->Stop++ = ap - (c + k);
            }

            k = h;
            e = PHF[ D ][ h ];
        }
    }

    return ( D + abs( del ) );
}

static int v3_align( msa_alignment* align )
{
    msa_alignment_ctx* pCtx = align->pCtx;
    char* bseq;
    msa_profile_entry* aseq;
    int *points, tlen;
    int ab, bb;
    int ae, be;
    int diffs;

    aseq = align->A;
    bseq = align->B;

    tlen   = align->nPtPoints;
    points = align->pPtPoints;

    {
        int d, s;
        int M, N;
        int dmax, nmax, mmax;
        int **PVF, **PHF;

        M = align->aepos - align->abpos;
        N = align->bepos - align->bbpos;

        s = M > N ? M : N;

        if ( s > pCtx->ntrace )
        {
            pCtx->ntrace = 1.2 * pCtx->ntrace + s;
            pCtx->trace  = (int*)realloc( pCtx->trace, sizeof( int ) * pCtx->ntrace );
        }

        mmax = 0;
        nmax = 0;
        dmax = 0;
        ab   = align->abpos;
        bb   = align->bbpos;

        for ( d = 0; d < tlen; d += 2 )
        {
            ae = points[ d ];
            be = points[ d + 1 ];

            M = ae - ab;
            N = be - bb;

            if ( M < N )
            {
                diffs = M;
            }
            else
            {
                diffs = N;
            }

            if ( diffs > dmax )
            {
                dmax = diffs;
            }

            if ( M > mmax )
            {
                mmax = M;
            }

            if ( N > nmax )
            {
                nmax = N;
            }

            ab = ae;
            bb = be;
        }

        ae = align->aepos;
        be = align->bepos;
        M  = ae - ab;
        N  = be - bb;

        if ( M < N )
        {
            diffs = M;
        }
        else
        {
            diffs = N;
        }

        if ( diffs > dmax )
        {
            dmax = diffs;
        }

        if ( M > mmax )
        {
            mmax = M;
        }

        if ( N > nmax )
        {
            nmax = N;
        }

        s = ( dmax + 3 ) * 2 * ( ( mmax + nmax + 3 ) * sizeof( int ) + sizeof( int* ) );

        if ( s > pCtx->vecmax )
        {
            if ( s > 256 * 1024 * 1024 )
            {
                printf( "v3_align> skipping alignment. excessive amount of memory needed.\n" );
                printf( "dmax = %d mmax = %d nmax = %d s = %d\n", dmax, mmax, nmax, s );

                return 0;
            }

            pCtx->vecmax = s * 1.2 + 10000;

            pCtx->vector = (int*)realloc( pCtx->vector, pCtx->vecmax );
        }

        pCtx->PVF = PVF = ( (int**)( pCtx->vector ) ) + 2;
        pCtx->PHF = PHF = PVF + ( dmax + 3 );

        // wave.PVF = PVF = ((int**) (work->vec.data())) + 2;
        // wave.PHF = PHF = PVF + (dmax + 3);

        s         = mmax + nmax + 3;
        PVF[ -2 ] = ( (int*)( PHF + ( dmax + 1 ) ) ) + ( nmax + 1 );

        for ( d = -1; d <= dmax; d++ )
        {
            PVF[ d ] = PVF[ d - 1 ] + s;
        }

        PHF[ -2 ] = PVF[ dmax ] + s;

        for ( d = -1; d <= dmax; d++ )
        {
            PHF[ d ] = PHF[ d - 1 ] + s;
        }
    }

    pCtx->Stop = pCtx->trace;
    pCtx->Aabs = aseq;
    pCtx->Babs = bseq;

    {
        int i;

        diffs = 0;
        ab    = align->abpos;
        bb    = align->bbpos;

        for ( i = 0; i < tlen; i += 2 )
        {
            be = points[ i + 1 ];
            ae = points[ i ];

            diffs += v3_align_onp( align, aseq + ab, ae - ab, bseq + bb, be - bb );

            ab = ae;
            bb = be;
        }

        ae = align->aepos;
        be = align->bepos;

        diffs += v3_align_onp( align, aseq + ab, ae - ab, bseq + bb, be - bb );
    }

    pCtx->ntrace = pCtx->Stop - pCtx->trace;

    // path->diffs = diffs;

    return 1;
}

static char decode_base( unsigned char c )
{
    assert( c <= 4 );

    static const char* pcTable = "ACGT-";
    return pcTable[ c ];
}

msa* msa_init()
{
    msa* m = (msa*)malloc( sizeof( msa ) );

    m->track = NULL;
    m->tmax  = 0;

    m->curprof = 0;
    m->profile = NULL;
    m->maxprof = 0;
    m->added   = 0;

    m->seq  = NULL;
    m->nseq = 0;

    m->msa_len   = NULL;
    m->msa_seq   = NULL;
    m->msa_max   = 0;
    m->msa_smax  = NULL;
    m->msa_lgaps = NULL;
    m->msa_ids   = NULL;

    m->aln_ctx         = (msa_alignment_ctx*)malloc( sizeof( msa_alignment_ctx ) );
    m->aln_ctx->ntrace = 0;
    m->aln_ctx->trace  = NULL;
    m->aln_ctx->vecmax = 0;
    m->aln_ctx->vector = NULL;

    m->ptp    = NULL;
    m->ptpmax = 0;

    return m;
}

void msa_free( msa* m )
{
    free( m->profile );

    free( m->ptp );

    int i;

    for ( i = 0; i < m->added; i++ )
    {
        free( m->msa_seq[ i ] );
    }

    free( m->msa_len );
    free( m->msa_smax );
    free( m->msa_seq );
    free( m->msa_lgaps );

    free( m->aln_ctx->trace );
    free( m->aln_ctx->vector );

    free( m->aln_ctx );

    free( m->track );

    free( m );
}

void msa_print( msa* m, FILE* fileOut, int b, int e )
{
    int i;

    b = ( b == -1 ? 0 : m->track[b] );
    e = ( e == -1 ? m->curprof : m->track[e] );

    for ( i = 0; i < m->added; i++ )
    {
        /*
        int j;

        for ( j = 0; j < m->msa_lgaps[ i ]; j++ )
        {
            putc( '-', fileOut );
        }

        fprintf( fileOut, "%s\n", m->msa_seq[ i ] );
        */

        fprintf( fileOut, "%-8d %-5d ", m->msa_ids[ i ], m->msa_len[ i ] );

        int j;
        for ( j = b; j < m->msa_lgaps[ i ]; j++ )
        {
            putc( '-', fileOut );
        }

        char* beg = m->msa_seq[ i ] + MAX( b - m->msa_lgaps[ i ], 0 );
        int len   = ( e - b ) - MAX( m->msa_lgaps[ i ] - b, 0 );

        fprintf( fileOut, "%.*s\n", len, beg );
    }
}

void msa_print_simple( msa* m, FILE* filemsa, FILE* filerids, int b, int e )
{
    int i;

    b = m->track[ b ];
    e = m->track[ e ];

    for ( i = 0; i < m->added; i++ )
    {
        int j;
        for ( j = b; j < m->msa_lgaps[ i ]; j++ )
        {
            putc( '-', filemsa );
        }

        char* beg = m->msa_seq[ i ] + MAX( b - m->msa_lgaps[ i ], 0 );
        int len   = ( e - b ) - MAX( m->msa_lgaps[ i ] - b, 0 );

        fprintf( filemsa, "%.*s\n", len, beg );
        fprintf( filerids, "%d\n", m->msa_ids[ i ] );
    }
}

void msa_print_v( msa* m, FILE* fileOut )
{
    char* cons = msa_consensus( m, 1 );

    int col, row;

    for ( col = 0; col < m->curprof; col++ )
    {
        putc( cons[ col ], fileOut );

        // printf("%*s", indent, "");

        for ( row = 0; row < m->added; row++ )
        {
            int abscol = col - m->msa_lgaps[ row ];

            if ( abscol >= 0 && abscol < m->msa_len[ row ] )
            {
                putc( m->msa_seq[ row ][ abscol ], fileOut );
            }
            else
            {
                putc( ' ', fileOut );
            }
        }

        putc( '\n', fileOut );
    }
}

void msa_print_profile( msa* m, FILE* fileOut, int b, int e, int colorize )
{
    msa_profile_entry* pEntry;

    b = ( b == -1 ? 0 : m->track[ b ] );
    e = ( e == -1 ? m->curprof : m->track[ e ] );

    // counts

    int i;

    if ( colorize )
    {
        printf( ANSI_COLOR_BLUE );
    }

    printf( "  " );

    for ( i = b; i < e; i++ )
    {
        pEntry = m->profile + i;
        fprintf( fileOut, "%7d", i % 100 );
    }

    if ( colorize )
    {
        printf( ANSI_COLOR_RESET );
    }

    fprintf( fileOut, "\n" );

    int nBase;

    for ( nBase = 0; nBase < 5; nBase++ )
    {
        fprintf( fileOut, "%c ", decode_base( nBase ) );

        for ( i = b; i < e; i++ )
        {
            pEntry = m->profile + i;
            fprintf( fileOut, "%7lu", pEntry->counts[ nBase ] );
        }

        fprintf( fileOut, "\n" );
    }

    // consensus

    if ( colorize )
    {
        printf( ANSI_COLOR_RED );
    }

    char* pcCons = msa_consensus( m, 1 );

    printf( "# " );

    for ( i = b; i < e; i++ )
    {
        fprintf( fileOut, "%7c", pcCons[ i ] );
    }

    if ( colorize )
    {
        printf( ANSI_COLOR_RESET );
    }

    fprintf( fileOut, "\n" );
}

void msa_reset( msa* m )
{
    m->curprof = 0;
    m->added   = 0;
}

static void msa_add_first( msa* m, char* seq, int len )
{
    if ( len > m->maxprof )
    {
        m->maxprof = 1.2 * m->maxprof + len;
        m->profile = realloc( m->profile, sizeof( msa_profile_entry ) * m->maxprof );
    }

    if ( len > m->tmax )
    {
        m->tmax  = 1.2 * m->tmax + len;
        m->track = realloc( m->track, sizeof( int ) * m->tmax );
    }

    m->curprof = len;
    bzero( m->profile, sizeof( msa_profile_entry ) * m->maxprof );

    int i;

    for ( i = 0; i < len; i++ )
    {
        m->profile[ i ].counts[ (unsigned char)seq[ i ] ] = 1;

        m->track[ i ] = i;
    }
}

static void apply_gap( msa* m, int pos, int seq_from, int seq_to )
{
    int i;
    int abspos;

    for ( i = seq_from; i < seq_to; i++ )
    {
        if ( pos <= m->msa_lgaps[ i ] )
        {
            m->msa_lgaps[ i ]++;
            continue;
        }

        abspos = pos - m->msa_lgaps[ i ];

        if ( abspos > m->msa_len[ i ] )
        {
            continue;
        }

        if ( m->msa_len[ i ] + 1 >= m->msa_smax[ i ] )
        {
            m->msa_smax[ i ] = m->msa_smax[ i ] * 1.2 + 100;
            m->msa_seq[ i ]  = realloc( m->msa_seq[ i ], m->msa_smax[ i ] );
        }

        memmove( m->msa_seq[ i ] + abspos + 1,
                 m->msa_seq[ i ] + abspos,
                 strlen( m->msa_seq[ i ] + abspos ) + 1 );

        m->msa_seq[ i ][ abspos ] = '-';

        m->msa_len[ i ]++;
    }
}

void msa_add( msa* m, char* seq, int pb, int pe, int sb, int se, ovl_trace* trace, int tlen, int id )
{
    int i;

    /*
    printf("add ");

    for (i = sb; i < se; i++)
    {
        printf("%c", decode_base(seq[i]));
    }

    printf("\n");
    */

    /*
    if ( m->added > 0 && m->added % 10 == 0 )
    {
        for ( i = 0; i < m->curprof; i++ )
        {
            msa_profile_entry* pe = m->profile + i;
            unsigned long sum = pe->counts[ 0 ] + pe->counts[ 1 ] + pe->counts[ 2 ] + pe->counts[ 3 ];

            if ( sum < pe->counts[ 4 ] )
            {
                pe->counts[0] = pe->counts[1] = pe->counts[2] = pe->counts[3] = 0;
                pe->counts[4] += sum;
            }
        }
    }
    */

    if ( m->msa_max <= m->added )
    {
        m->msa_max = m->msa_max * 1.2 + 10;

        m->msa_seq   = (char**)realloc( m->msa_seq, sizeof( char* ) * m->msa_max );
        m->msa_smax  = (int*)realloc( m->msa_smax, sizeof( int ) * m->msa_max );
        m->msa_lgaps = (int*)realloc( m->msa_lgaps, sizeof( int ) * m->msa_max );
        m->msa_len   = (int*)realloc( m->msa_len, sizeof( int ) * m->msa_max );
        m->msa_ids   = (int*)realloc( m->msa_ids, sizeof( int ) * m->msa_max );
    }

    int smax                = ( se - sb ) * 2;
    m->msa_seq[ m->added ]  = (char*)malloc( smax );
    m->msa_smax[ m->added ] = smax;

    for ( i = sb; i < se; i++ )
    {
        m->msa_seq[ m->added ][ i - sb ] = decode_base( seq[ i ] );
    }

    m->msa_seq[ m->added ][ i - sb ] = '\0';

    m->msa_len[ m->added ] = i - sb; // se - sb;
    m->msa_ids[ m->added ] = id;

    if ( m->curprof == 0 )
    {
        m->msa_lgaps[ m->added ] = 0;

        m->alen = se - sb;

        msa_add_first( m, seq + sb, se - sb );
        m->added++;

        // consensus_print_profile(cns, stdout);

        return;
    }

    msa_alignment aln;
    aln.pCtx = m->aln_ctx;

    aln.diffs_a = 0;
    aln.diffs_b = 0;

    aln.B     = seq;
    aln.bbpos = sb;
    aln.bepos = se;

    aln.A = m->profile;

    // convert to profile relative coordinates

    if ( pb != -1 )
    {
        aln.abpos = m->track[ pb ];
    }
    else
    {
        aln.abpos = 0;
    }

    if ( pe != -1 )
    {
        aln.aepos = m->track[ pe - 1 ] + 1;
    }
    else
    {
        aln.aepos = m->curprof;
    }

    m->msa_lgaps[ m->added ] = aln.abpos;

    assert( aln.bbpos >= 0 );
    assert( aln.bbpos < aln.bepos );

    // prepare pass through points

    if ( tlen > 0 )
    {
        int ptp = tlen - 2;
        assert( ( ptp % 2 ) == 0 );

        if ( ptp > m->ptpmax )
        {
            m->ptpmax = ptp * 1.2 + 100;
            m->ptp    = (int*)realloc( m->ptp, sizeof( int ) * m->ptpmax );
        }

        int ptp_p = ( pb / m->twidth ) * m->twidth;
        int ptp_s = sb;

        aln.nPtPoints = tlen - 2;
        aln.pPtPoints = m->ptp;

        for ( i = 1; i < tlen - 2; i += 2 )
        {
            ptp_p += m->twidth;
            ptp_s += trace[ i ];

            aln.pPtPoints[ i - 1 ] = m->track[ ptp_p ];
            aln.pPtPoints[ i ]     = ptp_s;
        }
    }
    else
    {
        aln.nPtPoints = 0;
    }

    // align

    if ( !v3_align( &aln ) )
    {
        return;
    }

    int c, p, b, n;

    // enlarge profile if necessary
    if ( m->curprof + aln.diffs_a > m->maxprof )
    {
        m->maxprof = m->maxprof * 2 + aln.diffs_a;

        m->profile = (msa_profile_entry*)realloc( m->profile, sizeof( msa_profile_entry ) * m->maxprof );
        bzero( m->profile + m->curprof, sizeof( msa_profile_entry ) * aln.diffs_a );
    }

    // apply trace to profile and sequence in reverse order
    // thereby allowing us to perform the update in place

    // profile overhang right
    n = m->curprof - 1 + aln.diffs_a;
    p = m->curprof - 1;

    while ( p > aln.aepos - 1 )
    {
        m->profile[ n ] = m->profile[ p ];
        p--;
        n--;
    }

    b = aln.bepos - 1;

    for ( i = aln.pCtx->ntrace - 1; i >= 0; i-- )
    {
        c = aln.pCtx->trace[ i ];

        if ( c > 0 ) // dash before B[-c]
        {
            c--;

            while ( c <= b )
            {
                m->profile[ n ] = m->profile[ p ];
                m->profile[ n ].counts[ (unsigned char)aln.B[ b ] ]++;

                n--;

                b--;
                p--;
            }

            apply_gap( m, c + aln.abpos - sb, m->added, m->added + 1 );

            m->profile[ n ] = m->profile[ p ];
            m->profile[ n ].counts[ 4 ]++;
            n--;

            p--;
        }
        else // dash before P[c]
        {
            c = -c;
            c--;

            while ( c <= p )
            {
                m->profile[ n ] = m->profile[ p ];
                m->profile[ n ].counts[ (unsigned char)aln.B[ b ] ]++;

                n--;

                b--;
                p--;
            }

            apply_gap( m, c, 0, m->added );

            memset( m->profile + n, 0, sizeof( msa_profile_entry ) );
            m->profile[ n ].counts[ (unsigned char)aln.B[ b ] ]++;
            m->profile[ n ].counts[ 4 ] = m->profile[ n + 1 ].counts[ 0 ] +
                                          m->profile[ n + 1 ].counts[ 1 ] +
                                          m->profile[ n + 1 ].counts[ 2 ] +
                                          m->profile[ n + 1 ].counts[ 3 ] +
                                          m->profile[ n + 1 ].counts[ 4 ];

            n--;

            b--;
        }
    }

    // aligned leftovers
    while ( b >= aln.bbpos )
    {
        m->profile[ n ].counts[ (unsigned char)aln.B[ b ] ]++;

        n--;

        b--;
        p--;
    }

    m->curprof += aln.diffs_a;

    // keep track of where the bases of the main read have ended up in the profile

    if ( aln.diffs_a > 0 )
    {
        int nPosIncr = 0;
        int nPos     = 0;

        int t;
        for ( t = 0; t < aln.pCtx->ntrace; t++ )
        {
            c = aln.pCtx->trace[ t ];

            if ( c > 0 )
            {
                continue;
            }

            c = -c;

            // c--;

            while ( nPos < m->alen && m->track[ nPos ] < c - 1 )
            {
                m->track[ nPos ] += nPosIncr;
                nPos++;
            }

            nPosIncr++;
        }

        while ( nPos < m->alen )
        {
            m->track[ nPos ] += nPosIncr;
            nPos++;
        }
    }

    m->added++;

    // consensus_print_profile(cns, stdout, 1);
}

char* msa_consensus( msa* m, int dashes )
{
    msa_profile_entry* pEntry;

    if ( m->nseq <= m->curprof )
    {
        m->nseq = m->curprof + 1;
        m->seq  = (char*)realloc( m->seq, m->nseq );
    }

    int curseq = 0;
    unsigned char nMax, cMax;
    unsigned short nCovGaps, nCov;

    int i, b;

    for ( i = 0; i < m->curprof; i++ )
    {
        pEntry = m->profile + i;
        nMax = cMax = 0;

        // find out whether ACTG or - received the most "votes"
        for ( b = 0; b < 5; b++ )
        {
            if ( pEntry->counts[ b ] > nMax )
            {
                nMax = pEntry->counts[ b ];
                cMax = b;
            }
        }

        nCov     = pEntry->counts[ 0 ] + pEntry->counts[ 1 ] + pEntry->counts[ 2 ] + pEntry->counts[ 3 ];
        nCovGaps = nCov + pEntry->counts[ 4 ];

        if ( nCov < nCovGaps * CONS_MIN_COV_FRACTION )
        {
            cMax = 4;
        }

        // if it is a dash, but we don't have to add "-" to the consensus sequence
        if ( cMax == 4 && !dashes )
        {
            continue;
        }

        m->seq[ curseq++ ] = decode_base( cMax );
    }

    m->seq[ curseq ] = '\0';

    return m->seq;
}
