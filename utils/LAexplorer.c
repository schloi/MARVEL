
#include <gtk/gtk.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <string.h>

#include "lib/oflags.h"
#include "lib/utils.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/lasidx.h"

#include "db/DB.h"
#include "dalign/align.h"

// drawing area padding

#define PADDING 20

#define HIGHLIGHT_COLORS 32

// defaults

#define DEF_LINEWIDTH 4

// track type

#define TRACK_INTERVAL          1
#define TRACK_POINT             2

#undef DEBUG

// sort type
typedef enum
{
    sort_abpos = 0, sort_aepos = 1, sort_length = 2, sort_readID = 3, sort_qual = 4
} sortType;

typedef enum
{
    show_q = 0, show_distinct = 1, show_same = 2
} showType;

typedef enum
{
    filter_include = 0, filter_exclude = 1, filter_highlight = 2
} filterType;

typedef struct
{
    Overlap ovl;
    GdkRGBA color;
} OverlapDetails;

typedef struct
{
    int rid;
    gdouble scroll_x;
    gdouble scroll_y;

    int filter_ab;
    int filter_ae;

    int hzoom;
} CurrentView;

// global status

typedef struct
{
    HITS_DB db;
    // FILE* fileLas;
    HITS_TRACK* qtrack;
    HITS_TRACK* srctrack;

    lasidx* lasIndex;
    int* lasIndexFile;
    FILE** lasFiles;

    char* pathDb;
    char* pathLas;

    // Overlap* ovls;
    // GdkRGBA* ovl_color;

    OverlapDetails* ovls;
    OverlapDetails** ovls_sorted;

    int omax;
    int ocur;

    int* ovl_display;
    int novl_display;

    int tmax;
    int tcur;
    ovl_trace* trace;

    size_t tbytes;
    int twidth;

    GdkRGBA highlight_color[HIGHLIGHT_COLORS];

    int* highlight;
    int hmax;
    int hcur;

    int rid_hover_highlight;

    GtkScrolledWindow* scrolled_wnd;        // container for drawing_area
    GtkWidget* drawing_area;
    GtkWidget* drawing_tracks;

    float hscale;
    int pad;

    HITS_TRACK** tracks;
    int* track_type;
    int maxtracks;
    int ntracks;

    // toolbar

    GtkWidget* edit_read;
    GtkWidget* edit_linewidth;
    GtkWidget* check_revSort;
    GtkWidget* edit_hzoom;
    GtkWidget* combo_show;

    GtkWidget* edit_filter_ab;
    GtkWidget* edit_filter_ae;

    // overlap details bar
    GtkWidget* overlap_details;
    int idx_details;

    // filter panel
    GtkWidget* filter_panel;
    int filter_mask;
    filterType filter;

    int line_width;
    int revSort;
    sortType sort;
    showType show;

    int ovl_y_start;        // y offset of first overlap

    CurrentView* viewstack;
    int viewstack_cur;
    int viewstack_fill;
    int viewstack_max;

} ExplorerContext;

static ExplorerContext g_ectx;

// oflags.c

extern OverlapFlag2Label oflag2label[];

// overlap sorting

static CurrentView* view_current()
{
    if (g_ectx.viewstack_cur >= 0)
    {
        return g_ectx.viewstack + g_ectx.viewstack_cur;
    }

    return NULL;
}

static CurrentView* view_prev()
{
    if (g_ectx.viewstack_cur > 0)
    {
        return g_ectx.viewstack + (g_ectx.viewstack_cur - 1);
    }

    return NULL;
}

static void view_new(int rid)
{
    CurrentView* prev = view_current();

    if ( prev != NULL )
    {
        prev->scroll_x = gtk_adjustment_get_value( gtk_scrolled_window_get_hadjustment(g_ectx.scrolled_wnd) );
        prev->scroll_y = gtk_adjustment_get_value( gtk_scrolled_window_get_vadjustment(g_ectx.scrolled_wnd) );
    }

    g_ectx.viewstack_cur += 1;
    g_ectx.viewstack_fill = g_ectx.viewstack_cur;

    if ( g_ectx.viewstack_cur >= g_ectx.viewstack_max )
    {
        g_ectx.viewstack_max = g_ectx.viewstack_max * 1.2 + 20;
        g_ectx.viewstack = realloc( g_ectx.viewstack, sizeof(CurrentView) * g_ectx.viewstack_max );
    }

    CurrentView* view = view_current();
    view->rid = rid;
    view->scroll_x = 0;
    view->scroll_y = 0;

    view->filter_ab = 0;
    view->filter_ae = -1;

    if ( prev != NULL )
    {
        view->hzoom = prev->hzoom;
    }
    else
    {
        view->hzoom = 1;
    }
}

static int view_back()
{
    if ( g_ectx.viewstack_cur < 1 )
    {
        return FALSE;
    }

    g_ectx.viewstack_cur -= 1;

    return TRUE;
}

static int view_forward()
{
    if (g_ectx.viewstack_cur == g_ectx.viewstack_fill)
    {
        return FALSE;
    }

    g_ectx.viewstack_cur += 1;

    return TRUE;
}

static int cmp_ovls_abpos(const void* a, const void* b)
{
    OverlapDetails* o1 = *(OverlapDetails**) a;
    OverlapDetails* o2 = *(OverlapDetails**) b;

    int cmp = o1->ovl.path.abpos - o2->ovl.path.abpos;

    if (!cmp)
    {
        cmp = (o1->ovl.path.aepos - o1->ovl.path.abpos) - (o2->ovl.path.aepos - o2->ovl.path.abpos);
    }

    return cmp;
}

static int cmp_ovls_id(const void* a, const void* b)
{
    OverlapDetails* o1 = *(OverlapDetails**) a;
    OverlapDetails* o2 = *(OverlapDetails**) b;

    int cmp = o1->ovl.bread - o2->ovl.bread;

    if (cmp == 0)
    {
        cmp = o1->ovl.path.abpos - o2->ovl.path.abpos;
    }

    if (cmp == 0)
    {
        cmp = o1->ovl.path.aepos - o2->ovl.path.aepos;
    }

    return cmp;
}

static int cmp_ovls_aepos(const void* a, const void* b)
{
    OverlapDetails* o1 = *(OverlapDetails**) a;
    OverlapDetails* o2 = *(OverlapDetails**) b;

    int cmp = o1->ovl.path.aepos - o2->ovl.path.aepos;

    if (!cmp)
    {
        cmp = (o1->ovl.path.aepos - o1->ovl.path.abpos) - (o2->ovl.path.aepos - o2->ovl.path.abpos);
    }

    return cmp;
}

static int cmp_ovls_length(const void* a, const void* b)
{
    OverlapDetails* o1 = *(OverlapDetails**) a;
    OverlapDetails* o2 = *(OverlapDetails**) b;

    return (o1->ovl.path.aepos - o1->ovl.path.abpos) - (o2->ovl.path.aepos - o2->ovl.path.abpos);
}

static int cmp_ovls_qual(const void* a, const void* b)
{
    OverlapDetails* o1 = *(OverlapDetails**) a;
    OverlapDetails* o2 = *(OverlapDetails**) b;

    return ((100 - (o1->ovl.path.diffs * 100.0 / (o1->ovl.path.aepos - o1->ovl.path.abpos))) * 10 - (100 - (o2->ovl.path.diffs * 100.0 / (o2->ovl.path.aepos - o2->ovl.path.abpos))) * 10);
}

static void adjust_drawing_area_size()
{
    GtkWidget* p = gtk_widget_get_parent(g_ectx.drawing_area);
    int pwidth = gtk_widget_get_allocated_width(p) * view_current()->hzoom;

    // gtk_widget_set_size_request(g_ectx.drawing_tracks, gtk_widget_get_allocated_width(p),
    //                            PADDING + (g_ectx.ntracks + 1) * DEF_LINEWIDTH + PADDING);

    gtk_widget_set_size_request(g_ectx.drawing_area, pwidth,
                                (g_ectx.novl_display + 1) * g_ectx.line_width * (g_ectx.pad + 1) + PADDING);
}

static void redraw()
{
    g_ectx.idx_details = -1;

    adjust_drawing_area_size();

    gtk_widget_queue_draw(g_ectx.drawing_area);
    gtk_widget_queue_draw(g_ectx.drawing_tracks);
}

// convert HSL to RGB
//

static double hue2rgb(double p, double q, double t)
{
    if (t < 0)
    {
        t += 1;
    }

    if (t > 1)
    {
        t -= 1;
    }

    if (t < 1. / 6)
    {
        return p + (q - p) * 6. * t;
    }

    if (t < 1. / 2)
    {
        return q;
    }

    if (t < 2. / 3)
    {
        return p + (q - p) * (2. / 3 - t) * 6.;
    }

    return p;
}

static void hsl2rgb(double h, double s, double l, GdkRGBA* rgb)
{
    double r, g, b;

    if (s == 0)
    {
        r = g = b = l; // achromatic
    }
    else
    {
        double q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        double p = 2. * l - q;

        r = hue2rgb(p, q, h + 1. / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1. / 3);
    }

    rgb->red = r;
    rgb->green = g;
    rgb->blue = b;

    rgb->alpha = 0;
}

static double rand2()
{
    return (double) rand() / (double) RAND_MAX;
}

static void get_unique_color(int color, int ncolors, GdkRGBA* rgb)
{
    double hue = color * 360.0 / ncolors;

    hsl2rgb(hue / 360.0, (50 + rand2() * 20) / 100.0, (50 + rand2() * 20) / 100.0, rgb);
}

//
// ///

// assign unique colors to b's occuring more than once

static void assign_colors()
{
    OverlapDetails* ovls = g_ectx.ovls;

    // how many colors do we need

    int i;
    int colors = 0;
    int beg = 0;

    for (i = 1; i < g_ectx.ocur; i++)
    {
        if (ovls[beg].ovl.bread != ovls[i].ovl.bread)
        {
            if (i - beg > 1)
            {
                colors++;
            }

            beg = i;
        }
    }

    if (i - beg > 1)
    {
        colors++;
    }

    // go through the HSL cylinder

    int col = 0;

    GdkRGBA rgb_default;
    rgb_default.red = 0.3;
    rgb_default.green = 0.3;
    rgb_default.blue = 0.3;
    rgb_default.alpha = 0.0;

    beg = 0;

    for (i = 1; i < g_ectx.ocur; i++)
    {
        if (ovls[beg].ovl.bread != ovls[i].ovl.bread)
        {
            if (i - beg > 1)
            {
                get_unique_color(col, colors, &(ovls[beg].color));
                col++;

                while (beg != i)
                {
                    ovls[beg + 1].color = ovls[beg].color;
                    beg++;
                }
            }
            else
            {
                ovls[beg].color = rgb_default;
                beg++;
            }

            beg = i;
        }
    }

    if (i - beg > 1)
    {
        get_unique_color(col, colors, &(ovls[beg].color));

        while (beg != i)
        {
            ovls[beg + 1].color = ovls[beg].color;
            beg++;
        }
    }
    else
    {
        ovls[i - 1].color = rgb_default;
        beg++;
    }
}

static int ypos2index(int y)
{
    return (y - PADDING - g_ectx.line_width) / ((1 + g_ectx.pad) * g_ectx.line_width);
}

static int ypos2overlap(int y)
{
    int ovl = (y - PADDING - g_ectx.line_width) / ((1 + g_ectx.pad) * g_ectx.line_width);

    if (ovl >= 0 && ovl < g_ectx.novl_display)
    {
        return g_ectx.ovl_display[ovl];
    }

    return -1;
}

static int overlap2ypos(int i)
{
    return g_ectx.ovl_y_start + (i + 1) * g_ectx.line_width * (1 + g_ectx.pad);
}

static int xpos2apos(int x)
{
    x -= PADDING;

    if (x < 0)
    {
        return 0;
    }

    x /= g_ectx.hscale;

    int ovlALen = DB_READ_LEN(&g_ectx.db, g_ectx.ovls->ovl.aread);

    if (x > ovlALen)
    {
        x = ovlALen;
    }

    return x;
}

static void sort_overlaps()
{
    int i;
    OverlapDetails** ovls_sorted = g_ectx.ovls_sorted;
    int n = g_ectx.ocur;

    for (i = 0; i < n; i++)
    {
        ovls_sorted[i] = g_ectx.ovls + i;
    }

    switch (g_ectx.sort)
    {
        case sort_abpos:
            qsort(ovls_sorted, n, sizeof(OverlapDetails*), cmp_ovls_abpos);
            break;

        case sort_aepos:
            qsort(ovls_sorted, n, sizeof(OverlapDetails*), cmp_ovls_aepos);
            break;

        case sort_length:
            qsort(ovls_sorted, n, sizeof(OverlapDetails*), cmp_ovls_length);
            break;

        case sort_readID:
            qsort(ovls_sorted, n, sizeof(OverlapDetails*), cmp_ovls_id);
            break;

        case sort_qual:
            qsort(ovls_sorted, n, sizeof(OverlapDetails*), cmp_ovls_qual);
            break;

        default:
            break;
    }

    if (g_ectx.revSort)
    {
        int j = n - 1;
        OverlapDetails* temp;

        for ( i = 0 ; i < n / 2 ; i++, j-- )
        {
            temp = ovls_sorted[i];
            ovls_sorted[i] = ovls_sorted[j];
            ovls_sorted[j] = temp;
        }

        if ( i != j )
        {
            temp = ovls_sorted[i];
            ovls_sorted[i] = ovls_sorted[j];
            ovls_sorted[j] = temp;
        }
    }

}

static void filter_overlaps(void)
{
    OverlapDetails** ovls_sorted = g_ectx.ovls_sorted;
    int ocur = g_ectx.ocur;
    int i;
    g_ectx.novl_display = 0;
    int filter_ab = view_current()->filter_ab;
    int filter_ae = view_current()->filter_ae;

    for (i = 0; i < ocur; i++)
    {
        Overlap* ovl = &(ovls_sorted[i]->ovl);

        switch (g_ectx.filter)
        {
            case filter_exclude:
                if ( ovl->flags & g_ectx.filter_mask )
                {
                    continue;
                }
                break;

            case filter_include:
                if ( !(ovl->flags & g_ectx.filter_mask) )
                {
                    continue;
                }
                break;

            default:
                break;
        }

        if (ovl->path.aepos < filter_ab || ovl->path.abpos > filter_ae)
        {
            continue;
        }

        g_ectx.ovl_display[g_ectx.novl_display++] = i;
    }
}


// load overlaps for read g_ectx.rid from las file

static void load_overlaps()
{
    int rid = view_current()->rid;
    int fileIdx = g_ectx.lasIndexFile[rid];
    off_t lasOffset = g_ectx.lasIndex[rid];

    int n = 0;
    int a = rid;

    if (lasOffset > 0)
    {
        OverlapDetails* ovls = g_ectx.ovls;
        int omax = g_ectx.omax;
        FILE* fileLas = g_ectx.lasFiles[fileIdx];
        g_ectx.tcur = 0;

        fseeko(fileLas, lasOffset, SEEK_SET);

        while (1)
        {
            if (Read_Overlap(fileLas, &(ovls[n].ovl)) || ovls[n].ovl.aread != a)
            {
                break;
            }

            if (ovls[n].ovl.path.tlen + g_ectx.tcur > g_ectx.tmax)
            {
                g_ectx.tmax = 1.2 * g_ectx.tmax + ovls[n].ovl.path.tlen;

                ovl_trace* trace = realloc(g_ectx.trace, g_ectx.tmax * sizeof(ovl_trace));

                int j;

                for (j = 0; j < n; j++)
                {
                    ovls[j].ovl.path.trace = trace + ((ovl_trace*) (ovls[j].ovl.path.trace) - g_ectx.trace);
                }

                g_ectx.trace = trace;
            }

            ovls[n].ovl.path.trace = g_ectx.trace + g_ectx.tcur;
            Read_Trace(fileLas, &(ovls[n].ovl), g_ectx.tbytes);

            g_ectx.tcur += ovls[n].ovl.path.tlen;

            if (g_ectx.tbytes == sizeof(uint8))
            {
                Decompress_TraceTo16(&(ovls[n].ovl));
            }

            n += 1;

            if (n >= omax)
            {
                omax = 1.2 * n + 100;
                ovls = realloc(ovls, sizeof(OverlapDetails) * omax);
                g_ectx.ovls_sorted = realloc(g_ectx.ovls_sorted, sizeof(OverlapDetails*) * omax);

                g_ectx.ovl_display = realloc(g_ectx.ovl_display, sizeof(int) * omax);
            }
        }

        g_ectx.ovls = ovls;
        g_ectx.omax = omax;
        g_ectx.ocur = n;
    }
    else
    {
        g_ectx.ocur = n = 0;
    }

    assign_colors();

    sort_overlaps();

    int ovlALen = DB_READ_LEN(&g_ectx.db, a);

    view_current()->filter_ab = 0;
    view_current()->filter_ae = ovlALen;

    gtk_spin_button_set_range(GTK_SPIN_BUTTON(g_ectx.edit_filter_ab), 0, ovlALen);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_ectx.edit_filter_ab), 0);

    gtk_spin_button_set_range(GTK_SPIN_BUTTON(g_ectx.edit_filter_ae), 0, ovlALen);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_ectx.edit_filter_ae), ovlALen);

    filter_overlaps();

    redraw();
}

static void set_segment_color(cairo_t* cr, int diffs, int len, int bHighlight)
{
    const int q_colors[] =
    { 22, 178, 0, 122, 195, 0, 211, 183, 0, 228, 82, 0, 245, 0, 34 };

    int q = ((double) diffs / len) * g_ectx.twidth;

    q = q / 10;

    if (q > 4)
    {
        q = 4;
    }

    float shift = 1.0;

    if (bHighlight)
    {
        shift -= 0.3;
    }

    cairo_set_source_rgb(cr, (q_colors[q * 3] * shift) / 255.0, (q_colors[q * 3 + 1] * shift) / 255.0, (q_colors[q * 3 + 2] * shift) / 255.0);
}

static int highlight_has(int rid)
{
    int i;

    for (i = 0; i < g_ectx.hcur; i++)
    {
        int h = g_ectx.highlight[i];

        if (h > rid)
        {
            return -1;
        }
        else if (g_ectx.highlight[i] == rid)
        {
            return i;
        }
    }

    return -1;
}

static int cmp_int(const void* x, const void* y)
{
    int* a = (int*) x;
    int* b = (int*) y;

    return (*a) - (*b);
}

static void highlight_add(int rid)
{
    if (g_ectx.hcur + 1 >= g_ectx.hmax)
    {
        g_ectx.hmax += 20;
        g_ectx.highlight = realloc(g_ectx.highlight, sizeof(int) * g_ectx.hmax);
    }

    g_ectx.highlight[g_ectx.hcur] = rid;
    g_ectx.hcur++;

    qsort(g_ectx.highlight, g_ectx.hcur, sizeof(int), cmp_int);
}

static void highlight_remove(int rid)
{
    int i;

    for (i = 0; i < g_ectx.hcur; i++)
    {
        if (g_ectx.highlight[i] == rid)
        {
            g_ectx.highlight[i] = INT_MAX;
            i = -1;

            break;
        }
    }

    if (i == -1)
    {
        qsort(g_ectx.highlight, g_ectx.hcur, sizeof(int), cmp_int);
        g_ectx.hcur--;
    }
}

static void draw_ovl(cairo_t* cr, OverlapDetails* ovld, int y, float scale, int clip_xb, int clip_xe, int bHighlight)
{
    Overlap* ovl = &(ovld->ovl);

    if (g_ectx.show == show_q)
    {
        int x = ovl->path.abpos;

        if (ovl->path.tlen == 0)
        {
            set_segment_color(cr, ovl->path.diffs, ovl->path.aepos - ovl->path.abpos, bHighlight);

            cairo_move_to(cr, PADDING + ovl->path.abpos * scale, y + 0.5);
            cairo_line_to(cr, PADDING + ovl->path.aepos * scale, y + 0.5);
            cairo_stroke(cr);
        }
        else
        {
            ovl_trace* trace = ovl->path.trace;
            int j;

            for (j = 0; j < ovl->path.tlen - 2; j += 2)
            {
                int x_next = (x / g_ectx.twidth + 1) * g_ectx.twidth;

                if (x_next * scale + PADDING < clip_xb)
                {
                    x = x_next;
                    continue;
                }

                cairo_move_to(cr, PADDING + x * scale, y + 0.5);

                set_segment_color(cr, trace[j], trace[j + 1], bHighlight);
                cairo_line_to(cr, PADDING + x_next * scale, y + 0.5);
                cairo_stroke(cr);

                x = x_next;

                if (x * scale + PADDING > clip_xe)
                {
                    break;
                }
            }

            if (x * scale + PADDING <= clip_xe)
            {
                set_segment_color(cr, trace[j], trace[j + 1], bHighlight);

                cairo_move_to(cr, PADDING + x * scale, y + 0.5);
                cairo_line_to(cr, PADDING + ovl->path.aepos * scale, y + 0.5);
                cairo_stroke(cr);
            }
        }
    }
    else
    {
        if (g_ectx.show == show_same)
        {
            int base = 0;

            if (bHighlight)
            {
                base = 1;
            }

            cairo_set_source_rgb(cr, fabs(base - 0.3), fabs(base - 0.3), fabs(base - 0.8));
        }
        else
        {
            if (bHighlight)
            {
                cairo_set_source_rgb(cr, 1.0 - ovld->color.red, 1.0 - ovld->color.green, 1.0 - ovld->color.blue);
            }
            else
            {
                cairo_set_source_rgb(cr, ovld->color.red, ovld->color.green, ovld->color.blue);
            }
        }

        cairo_move_to(cr, PADDING + ovl->path.abpos * scale, y + 0.5);
        cairo_line_to(cr, PADDING + ovl->path.aepos * scale, y + 0.5);
        cairo_stroke(cr);
    }

    int hidx = highlight_has(ovl->bread);
    CurrentView* prev = view_prev();

    if (prev != NULL && prev->rid == ovl->bread)
    {
        cairo_set_source_rgb(cr, 0, 0, 0);

        cairo_move_to(cr, PADDING + ovl->path.abpos * scale - 10, y + 0.5);
        cairo_line_to(cr, PADDING + ovl->path.abpos * scale - 5, y + 0.5);

        cairo_move_to(cr, PADDING + ovl->path.aepos * scale + 5, y + 0.5);
        cairo_line_to(cr, PADDING + ovl->path.aepos * scale + 10, y + 0.5);

        cairo_stroke(cr);
    }
    else if (hidx != -1)
    {
        hidx = hidx % HIGHLIGHT_COLORS;
        GdkRGBA* color = g_ectx.highlight_color + hidx;

        cairo_set_source_rgb(cr, color->red, color->green, color->blue);

        cairo_move_to(cr, PADDING + ovl->path.abpos * scale - 10, y + 0.5);
        cairo_line_to(cr, PADDING + ovl->path.abpos * scale - 5, y + 0.5);

        cairo_move_to(cr, PADDING + ovl->path.aepos * scale + 5, y + 0.5);
        cairo_line_to(cr, PADDING + ovl->path.aepos * scale + 10, y + 0.5);

        cairo_stroke(cr);
    }
}

static gboolean track_draw_callback(GtkWidget* widget, cairo_t* cr, gpointer data)
{
    UNUSED(data);

    int rid = view_current()->rid;
    int line_width = DEF_LINEWIDTH;

    int width = gtk_widget_get_allocated_width(widget);

    GdkRectangle clip;
    gdk_cairo_get_clip_rectangle(cr, &clip);

    int clip_xb = clip.x;
    int clip_xe = clip.x + clip.width;
    int clip_yb = clip.y;
    // int clip_ye = clip.y + clip.height;

    int alen = DB_READ_LEN(&g_ectx.db, rid);
    int hzoom = view_current()->hzoom;
    float scale = (width * hzoom - 2.0 * PADDING) / alen;

    int x_offset = gtk_adjustment_get_value( gtk_scrolled_window_get_hadjustment(g_ectx.scrolled_wnd) );

    cairo_set_line_width(cr, line_width);

    // background

    cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
    cairo_rectangle(cr, clip_xb, clip_xe, clip.width, clip.height);
    cairo_fill(cr);

    int y = PADDING;

    // tracks

    int i;

    for (i = 0; i < g_ectx.ntracks; i++, y += line_width * 2)
    {
        track_anno* anno = g_ectx.tracks[i]->anno;
        track_data* data = g_ectx.tracks[i]->data;

        track_anno b = anno[rid] / sizeof(track_data);
        track_anno e = anno[rid + 1] / sizeof(track_data);

        cairo_set_source_rgb(cr, i % 2, (i % 3) / 2.0, (i % 4) / 3.0);

        switch (g_ectx.track_type[i])
        {
            case TRACK_INTERVAL:
            {
                for ( ; b < e ; b += 2 )
                {
                    track_data rb = data[b];
                    track_data re = data[b + 1];

                    int xb = MIN(width, MAX(0, PADDING + rb * scale - x_offset));
                    int xe = MIN(width, MAX(0, PADDING + re * scale - x_offset));

                    if ( xb >= xe )
                    {
                        continue;
                    }

                    cairo_move_to(cr, xb, y); // PADDING + rb * scale, y);
                    cairo_line_to(cr, xe, y); // PADDING + re * scale, y);
                }

                break;
            }

            case TRACK_POINT:
            {
                // TODO
                break;
            }

            default:
                break;
        }

        cairo_stroke(cr);

        // y += line_width * 2;

    }

    if (y > clip_yb)
    {

        // a read

        cairo_set_line_width(cr, line_width);
        cairo_set_source_rgb(cr, 1, 0, 0);

        int xb = MIN(width, MAX(0, PADDING - x_offset));
        int xe = MIN(width, MAX(0, width * hzoom - PADDING - x_offset));

        cairo_move_to(cr, xb, y + 0.5);
        cairo_line_to(cr, xe, y + 0.5);

        cairo_stroke(cr);

        // a read ticks

        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_line_width(cr, 2.5);

        for (i = 500; i < alen; i += 500)
        {
            int x = PADDING + i * scale - x_offset;

            if ( x < 0 )
            {
                continue;
            }

            if ( x > width )
            {
                break;
            }

            cairo_move_to(cr, x, y - line_width);
            cairo_line_to(cr, x, y - line_width / 2);
        }

        cairo_stroke(cr);

    }


    return FALSE;
}

static gboolean read_draw_callback(GtkWidget* widget, cairo_t* cr, gpointer data)
{
    UNUSED(data);

    guint width; // , height;
    OverlapDetails** ovls_sorted = g_ectx.ovls_sorted;
    int rid = view_current()->rid;

    width = gtk_widget_get_allocated_width(widget);
    // height = gtk_widget_get_allocated_height(widget);

    GdkRectangle clip;
    gdk_cairo_get_clip_rectangle(cr, &clip);

    int clip_xb = clip.x;
    int clip_xe = clip.x + clip.width;
    int clip_yb = clip.y;
    int clip_ye = clip.y + clip.height;

#ifdef DEBUG
    printf("draw clip rect %4d..%4d -> %4d..%4d\n", clip_xb, clip_yb, clip_xe, clip_ye);
#endif

    // background

    cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
    cairo_rectangle(cr, clip_xb, clip_xe, clip.width, clip.height);
    cairo_fill(cr);

    cairo_set_line_width(cr, g_ectx.line_width);
    int alen = DB_READ_LEN(&g_ectx.db, rid);
    float scale = g_ectx.hscale = (width - 2.0 * PADDING) / alen;

    // printf("drawing %d overlaps @ scale %.2f\n", ocur, scale);

    // b reads

    int y = PADDING;
    int i;

    cairo_set_line_width(cr, g_ectx.line_width);
    // y += g_ectx.line_width;

    g_ectx.ovl_y_start = y;

    int draw_start = MAX(0, ypos2index( clip_yb ) - 1);
    int draw_end = ypos2index( clip_ye );

    if (draw_end == -1 || draw_end > g_ectx.novl_display)
    {
        draw_end = g_ectx.novl_display;
    }
    else
    {
        draw_end = MIN(g_ectx.novl_display, draw_end + 1);
    }

#ifdef DEBUG
    printf("draw_start %d .. draw_end %d\n", draw_start, draw_end);
#endif

    y += draw_start * g_ectx.line_width * (1 + g_ectx.pad);

    for ( i = draw_start ; i < draw_end ; i++ )
    {
        int idx = g_ectx.ovl_display[i];
        Overlap* ovl = &(ovls_sorted[idx]->ovl);

        y += g_ectx.line_width * (1 + g_ectx.pad);

#ifdef DEBUG
        printf("drawing %d @ y %4d .. bread %8d .. highlight %8d\n", i, y, ovl->bread, g_ectx.rid_hover_highlight);
#endif

        int highlight = 0;

        if ( ovl->bread == g_ectx.rid_hover_highlight ||
             ( g_ectx.filter == filter_highlight && ovl->flags & g_ectx.filter_mask ) )
        {
            highlight = 1;
        }

        draw_ovl(cr, ovls_sorted[idx], y, scale, clip_xb, clip_xe, highlight);
    }

    return FALSE;
}

static gboolean on_key_press(GtkWidget* widget, GdkEventKey* event, gpointer user_data)
{
    UNUSED(widget);
    UNUSED(user_data);

    switch (event->keyval)
    {
        case GDK_KEY_plus:
            gtk_spin_button_spin(GTK_SPIN_BUTTON(g_ectx.edit_read), GTK_SPIN_STEP_FORWARD, 1);
            break;

        case GDK_KEY_minus:
            gtk_spin_button_spin(GTK_SPIN_BUTTON(g_ectx.edit_read), GTK_SPIN_STEP_BACKWARD, 1);
            break;

        case GDK_KEY_f:
            if (view_forward())
            {
                gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_ectx.edit_read), view_current()->rid);
            }
            break;

        case GDK_KEY_b:
            if (view_back())
            {
                gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_ectx.edit_read), view_current()->rid);
            }
            break;

        case GDK_KEY_q:
            g_ectx.show = show_q;
            gtk_combo_box_set_active_id(GTK_COMBO_BOX(g_ectx.combo_show), "q");
            redraw();
            break;

        case GDK_KEY_w:
            g_ectx.show = show_distinct;
            gtk_combo_box_set_active_id(GTK_COMBO_BOX(g_ectx.combo_show), "distinct");
            redraw();
            break;

        case GDK_KEY_e:
            g_ectx.show = show_same;
            gtk_combo_box_set_active_id(GTK_COMBO_BOX(g_ectx.combo_show), "same");
            redraw();
            break;

        case GDK_KEY_h:
        {
            int rid = g_ectx.rid_hover_highlight;

            if (rid != -1)
            {
                if (highlight_has(rid) != -1)
                {
                    highlight_remove(rid);
                }
                else
                {
                    highlight_add(rid);
                }

                redraw();
            }
        }

            break;

        case GDK_KEY_H:
            {
                int rid = view_current()->rid;

                if (rid != -1)
                {
                    if (highlight_has(rid) != -1)
                    {
                        highlight_remove(rid);
                    }
                    else
                    {
                        highlight_add(rid);
                    }
                }

            }
            break;

        case GDK_KEY_c:
            g_ectx.hcur = 0;
            redraw();

            break;

        case GDK_KEY_l:
            gtk_spin_button_spin(GTK_SPIN_BUTTON(g_ectx.edit_linewidth), GTK_SPIN_STEP_BACKWARD, 1);
            break;

        case GDK_KEY_L:
            gtk_spin_button_spin(GTK_SPIN_BUTTON(g_ectx.edit_linewidth), GTK_SPIN_STEP_FORWARD, 1);
            break;
    }

    return FALSE;
}

static gboolean frame_callback(GtkWindow* window, GdkEvent* event, gpointer data)
{
    UNUSED(window);
    UNUSED(event);
    UNUSED(data);

    adjust_drawing_area_size();

    return FALSE;
}

static void edit_read_value_changed(GtkSpinButton* spin_button, gpointer user_data)
{
    UNUSED(user_data);

    int rid = gtk_spin_button_get_value_as_int(spin_button);

    if ( view_current()->rid != rid )
    {
        view_new(rid);
    }

    load_overlaps();

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));
}

static void edit_hzoom_changed(GtkSpinButton* spin_button, gpointer user_data)
{
    UNUSED(spin_button);
    UNUSED(user_data);

    view_current()->hzoom = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(g_ectx.edit_hzoom));

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));

    redraw();
}

static void edit_filter_pos_changed(GtkSpinButton* spin_button, gpointer user_data)
{
    UNUSED(spin_button);
    UNUSED(user_data);

    int filter_ab = view_current()->filter_ab = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(g_ectx.edit_filter_ab));
    int filter_ae = view_current()->filter_ae = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(g_ectx.edit_filter_ae));

    if (filter_ab >= filter_ae)
    {
        view_current()->filter_ab = 0;
        view_current()->filter_ae = DB_READ_LEN(&g_ectx.db, g_ectx.ovls->ovl.aread);
    }

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));

    filter_overlaps();
    redraw();
}

static void edit_linewidth_value_changed(GtkSpinButton* sbtn, gpointer user_data)
{
    UNUSED(user_data);

    g_ectx.line_width = gtk_spin_button_get_value_as_int(sbtn);
    adjust_drawing_area_size();

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));
}

static void check_revSort_toggled(GtkToggleButton* togglebutton, gpointer user_data)
{
    UNUSED(user_data);

    g_ectx.revSort = gtk_toggle_button_get_active(togglebutton);

    sort_overlaps();
    filter_overlaps();
    redraw();
}

static void check_pad_toggled(GtkToggleButton* togglebutton, gpointer user_data)
{
    UNUSED(user_data);

    g_ectx.pad = gtk_toggle_button_get_active(togglebutton);
    adjust_drawing_area_size();

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));

}

static void combo_filter_changed(GtkComboBox* widget, gpointer user_data)
{
    UNUSED(user_data);

    if (strcmp(gtk_combo_box_get_active_id(widget), "exclude") == 0)
    {
        g_ectx.filter = filter_exclude;
    }
    else if (strcmp(gtk_combo_box_get_active_id(widget), "include") == 0)
    {
        g_ectx.filter = filter_include;
    }
    else if (strcmp(gtk_combo_box_get_active_id(widget), "highlight") == 0)
    {
        g_ectx.filter = filter_highlight;
    }

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));

    filter_overlaps();
    redraw();
}

static void combo_show_changed(GtkComboBox* widget, gpointer user_data)
{
    UNUSED(user_data);

    if (strcmp(gtk_combo_box_get_active_id(widget), "q") == 0)
    {
        g_ectx.show = show_q;
    }
    else if (strcmp(gtk_combo_box_get_active_id(widget), "distinct") == 0)
    {
        g_ectx.show = show_distinct;
    }
    else
    {
        g_ectx.show = show_same;
    }

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));

    redraw();
}

static void combo_sort_changed(GtkComboBox* widget, gpointer user_data)
{
    UNUSED(user_data);

    if (strcmp(gtk_combo_box_get_active_id(widget), "abpos") == 0)
    {
        g_ectx.sort = sort_abpos;
    }
    else if (strcmp(gtk_combo_box_get_active_id(widget), "aepos") == 0)
    {
        g_ectx.sort = sort_aepos;
    }
    else if (strcmp(gtk_combo_box_get_active_id(widget), "length") == 0)
    {
        g_ectx.sort = sort_length;
    }
    else if (strcmp(gtk_combo_box_get_active_id(widget), "readID") == 0)
    {
        g_ectx.sort = sort_readID;
    }
    else
    {
        g_ectx.sort = sort_qual;
    }

    gtk_widget_grab_focus(GTK_WIDGET(g_ectx.scrolled_wnd));

    sort_overlaps();
    filter_overlaps();
    redraw();
}

static GtkWidget* setup_toolbar()
{
    GtkWidget* toolbar = gtk_grid_new();

    gtk_grid_set_column_spacing(GTK_GRID(toolbar), 10);

    // read
    gtk_grid_attach_next_to(GTK_GRID(toolbar), gtk_label_new("Read"), NULL, GTK_POS_RIGHT, 1, 1);
    GtkWidget* edit_read = g_ectx.edit_read = gtk_spin_button_new_with_range(0, g_ectx.db.nreads - 1, 1);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), edit_read, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(edit_read), "value-changed", G_CALLBACK(edit_read_value_changed), NULL);

    // sort mode
    GtkWidget* combo_show = g_ectx.combo_show = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_show), "q", "Q");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_show), "distinct", "Distinct");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_show), "same", "Same");
    gtk_combo_box_set_active_id(GTK_COMBO_BOX(combo_show), "same");
    g_ectx.show = show_same;
    g_signal_connect(G_OBJECT(combo_show), "changed", G_CALLBACK(combo_show_changed), NULL);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), combo_show, NULL, GTK_POS_RIGHT, 1, 1);

    // line width
    gtk_grid_attach_next_to(GTK_GRID(toolbar), gtk_label_new("Width"), NULL, GTK_POS_RIGHT, 1, 1);
    GtkWidget* edit_linewidth = g_ectx.edit_linewidth = gtk_spin_button_new_with_range(1, 10, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(edit_linewidth), DEF_LINEWIDTH);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), edit_linewidth, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(edit_linewidth), "value-changed", G_CALLBACK(edit_linewidth_value_changed), NULL);
    GtkWidget* check_pad = gtk_check_button_new();
    gtk_grid_attach_next_to(GTK_GRID(toolbar), check_pad, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(check_pad), "toggled", G_CALLBACK(check_pad_toggled), NULL);

    // h-zoom
    gtk_grid_attach_next_to(GTK_GRID(toolbar), gtk_label_new("H-Zoom"), NULL, GTK_POS_RIGHT, 1, 1);
    GtkWidget* edit_hzoom = g_ectx.edit_hzoom = gtk_spin_button_new_with_range(1, 100, 1);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), edit_hzoom, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(edit_hzoom), "value-changed", G_CALLBACK(edit_hzoom_changed), NULL);

    // begin/end filter
    gtk_grid_attach_next_to(GTK_GRID(toolbar), gtk_label_new("Range"), NULL, GTK_POS_RIGHT, 1, 1);
    GtkWidget* edit_filter_ab = g_ectx.edit_filter_ab = gtk_spin_button_new_with_range(0, 100, 100);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), edit_filter_ab, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(edit_filter_ab), "value-changed", G_CALLBACK(edit_filter_pos_changed), NULL);
    GtkWidget* edit_filter_ae = g_ectx.edit_filter_ae = gtk_spin_button_new_with_range(0, 100, 100);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), edit_filter_ae, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(edit_filter_ae), "value-changed", G_CALLBACK(edit_filter_pos_changed), NULL);

    // sort mode
    gtk_grid_attach_next_to(GTK_GRID(toolbar), gtk_label_new("Sort"), NULL, GTK_POS_RIGHT, 1, 1);
    GtkWidget* combo_sort = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_sort), "abpos", "begin");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_sort), "aepos", "end");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_sort), "readID", "read id");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_sort), "length", "length");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_sort), "qual", "quality");
    gtk_combo_box_set_active_id(GTK_COMBO_BOX(combo_sort), "abpos");
    g_ectx.sort = sort_abpos;
    g_signal_connect(G_OBJECT(combo_sort), "changed", G_CALLBACK(combo_sort_changed), NULL);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), combo_sort, NULL, GTK_POS_RIGHT, 1, 1);

    // reverse Sort
    GtkWidget* check_revSort = g_ectx.check_revSort = gtk_check_button_new();
    gtk_grid_attach_next_to(GTK_GRID(toolbar), check_revSort, NULL, GTK_POS_RIGHT, 1, 1);
    g_signal_connect(G_OBJECT(check_revSort), "toggled", G_CALLBACK(check_revSort_toggled), NULL);
    gtk_grid_attach_next_to(GTK_GRID(toolbar), gtk_label_new("reverse"), NULL, GTK_POS_RIGHT, 1, 1);

    return toolbar;
}

static void update_details_bar(int x, int idx)
{
    GtkWidget* bar = g_ectx.overlap_details;

    OverlapDetails* ovls = g_ectx.ovls;
    OverlapDetails** ovls_sorted = g_ectx.ovls_sorted;
    Overlap* ovl = &(ovls_sorted[idx]->ovl);

    int i;
    int cov = 0;

    for (i = 0; i < g_ectx.ocur; i++)
    {
        // TODO --- compute coverage stats in filter_overlaps

        if (ovls[i].ovl.path.abpos <= x && x <= ovls[i].ovl.path.aepos)
        {
            cov++;
        }
    }

    char flags[OVL_FLAGS + 1];
    flags2str(flags, ovl->flags);

    char q[8];
    if (g_ectx.qtrack)
    {
        track_anno* qanno = g_ectx.qtrack->anno;
        track_data* qdata = g_ectx.qtrack->data;
        int ob = qanno[ovl->aread] / sizeof(track_data);
        // int oe = qanno[ovl->aread + 1] / sizeof(track_data);

        int qvalue = qdata[ ob + x / g_ectx.twidth ];

        sprintf(q, " q %2d", qvalue);
    }
    else
    {
        q[0] = '\0';
    }

    char src[64];
    if (g_ectx.srctrack)
    {
        track_anno* srcanno = g_ectx.srctrack->anno;
        track_data* srcdata = g_ectx.srctrack->data;

        uint64 srca = srcdata[ srcanno[ovl->aread] / sizeof(track_data) ];
        uint64 srcb = srcdata[ srcanno[ovl->bread] / sizeof(track_data) ];

        sprintf(src, " source reads %8llu x %8llu", srca, srcb);
    }
    else
    {
        src[0] = '\0';
    }

    gchar* text = g_markup_printf_escaped(
            "<tt>@ %5dbp cov %3d%s <b>%8d</b> (%5d) x <b>%8d</b> (%5d) <b>%5d..%5d</b> (%5d) %c <b>%5d..%5d</b> (%5d)%s flags %s</tt>",
             x, cov, q,
             ovl->aread, DB_READ_LEN(&g_ectx.db, ovl->aread),
             ovl->bread, DB_READ_LEN(&g_ectx.db, ovl->bread),
             ovl->path.abpos, ovl->path.aepos, ovl->path.aepos - ovl->path.abpos,
             ovl->flags & OVL_COMP ? '<' : '>',
             ovl->path.bbpos, ovl->path.bepos, ovl->path.bepos - ovl->path.bbpos,
             src, flags);

    gtk_label_set_markup(GTK_LABEL(bar), text);

#if GTK_CHECK_VERSION(3, 14, 0)
    gtk_list_box_unselect_all(GTK_LIST_BOX(g_ectx.filter_panel));

    i = 0;

    while (oflag2label[i].mask)
    {
        if (oflag2label[i].mask & ovl->flags)
        {
            gtk_list_box_select_row(GTK_LIST_BOX(g_ectx.filter_panel),
                                    gtk_list_box_get_row_at_index(GTK_LIST_BOX(g_ectx.filter_panel), i));
        }

        i++;
    }

#endif

    g_free(text);
}

static gboolean read_mouse_move(GtkWidget* widget, GdkEvent* event, gpointer user_data)
{
    UNUSED(widget);
    UNUSED(user_data);

    int idx = ypos2overlap(event->button.y);
    int x = xpos2apos(event->button.x);

    if (idx != -1) //  && idx != g_ectx.idx_details)
    {
        g_ectx.idx_details = idx;
        update_details_bar(x, idx);
    }

    if (idx != -1)
    {
        int bread = g_ectx.ovls_sorted[idx]->ovl.bread;
        int bprev = g_ectx.rid_hover_highlight;

        if (bread != bprev)
        {
            int i;
            int width = gtk_widget_get_allocated_width(widget);

            g_ectx.rid_hover_highlight = bread;

            if (bprev != -1)
            {
#ifdef DEBUG
                printf("clear highlight bread %8d\n", bprev);
#endif

                for (i = 0; i < g_ectx.novl_display; i++)
                {
                    if (g_ectx.ovls_sorted[g_ectx.ovl_display[i]]->ovl.bread == bprev)
                    {
                        gtk_widget_queue_draw_area(g_ectx.drawing_area,
                                                   PADDING, overlap2ypos(i) - g_ectx.line_width, width - 2 * PADDING, g_ectx.line_width * (2 + g_ectx.pad));
                    }
                }
            }

#ifdef DEBUG
            printf("add highlight bread %8d\n", bread);
#endif

            for (i = 0; i < g_ectx.novl_display; i++)
            {
                if (g_ectx.ovls_sorted[g_ectx.ovl_display[i]]->ovl.bread == bread)
                {
                    gtk_widget_queue_draw_area(g_ectx.drawing_area,
                                               PADDING, overlap2ypos(i) - g_ectx.line_width, width - 2 * PADDING, g_ectx.line_width * (2 + g_ectx.pad));
                }
            }
        }

    }

    return GDK_EVENT_PROPAGATE;
}

static gboolean context_go_to(GtkWidget* widget, GdkEvent* event, gpointer user_data)
{
    UNUSED(widget);
    UNUSED(event);
    UNUSED(user_data);

    int rid = g_ectx.rid_hover_highlight;

    if (rid >= 0)
    {
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_ectx.edit_read), rid);

        view_new(rid);
        load_overlaps();
    }

    return TRUE;
}

static gboolean menuitem_track(GtkWidget* widget, GdkEvent* event, gpointer user_data)
{
    UNUSED(widget);
    UNUSED(event);
    UNUSED(user_data);

    int rid = g_ectx.rid_hover_highlight;

    if (highlight_has(rid) != -1)
    {
        highlight_remove(rid);
    }
    else
    {
        highlight_add(rid);
    }

    redraw();

    return TRUE;
}

static gint show_right_click_menu(GdkEvent* event)
{
    GtkMenu* menu = GTK_MENU(gtk_menu_new());
    // GdkEventButton* event_button = (GdkEventButton*) event;

    g_return_val_if_fail(event != NULL, FALSE);

    char label[255];
    int rid = g_ectx.rid_hover_highlight;

    sprintf(label, "READ %d", rid);
    GtkWidget* track_item = gtk_menu_item_new_with_label(label);
    int i = 0;
    gtk_menu_attach(menu, track_item, 0, 1, i, i + 1);
    i++;

    sprintf(label, "go to");
    track_item = gtk_menu_item_new_with_label(label);
    gtk_menu_attach(menu, track_item, 0, 1, i, i + 1);
    g_signal_connect(G_OBJECT(track_item), "activate", G_CALLBACK(context_go_to), NULL);

    i++;

    if (highlight_has(rid) != -1)
    {
        sprintf(label, "remove highlight");

        rid = -rid;
    }
    else
    {
        sprintf(label, "highlight");
    }

    track_item = gtk_menu_item_new_with_label(label);

    gtk_menu_attach(menu, track_item, 0, 1, i, i + 1);
    i++;

    g_signal_connect(G_OBJECT(track_item), "activate", G_CALLBACK(menuitem_track), NULL);

    gtk_widget_show_all(GTK_WIDGET(menu));

#if GTK_CHECK_VERSION(3, 22, 0)
    gtk_menu_popup_at_pointer(menu, event);
#endif

    return TRUE;
}

static gboolean read_button_press(GtkWidget* widget, GdkEvent* event, gpointer user_data)
{
    UNUSED(widget);
    UNUSED(user_data);

    if (event->type == GDK_BUTTON_PRESS)
    {
        GdkEventButton* event_button = (GdkEventButton*) event;

        int idx = ypos2overlap(event->button.y);

        if (event_button->button == GDK_BUTTON_PRIMARY)
        {

            if (idx != -1)
            {
                int rid = g_ectx.ovls_sorted[idx]->ovl.bread;

                gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_ectx.edit_read), rid);
            }

        }
        else if (event_button->button == GDK_BUTTON_SECONDARY)
        {
            show_right_click_menu(event);
        }
    }

    return GDK_EVENT_PROPAGATE;
}

static void check_filter_toggled(GtkToggleButton* togglebutton, gpointer user_data)
{
    UNUSED(togglebutton);

    OverlapFlag2Label* label = (OverlapFlag2Label*) user_data;

    int flag = label->mask;

    if (g_ectx.filter == filter_highlight)
    {
        g_ectx.rid_hover_highlight = -1;
    }

    g_ectx.filter_mask ^= flag;

    filter_overlaps();
    redraw();
}

#if GTK_CHECK_VERSION(3, 10, 0)

static GtkWidget* setup_filter_panel(GtkWidget** listbox)
{
    GtkWidget* right_panel = gtk_grid_new();

    // filter mode
    GtkWidget* combo_filter = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_filter), "exclude", "exclude selected");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_filter), "include", "only selected");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(combo_filter), "highlight", "highlight selected");
    gtk_combo_box_set_active_id(GTK_COMBO_BOX(combo_filter), "exclude");
    g_ectx.filter = filter_exclude;
    g_signal_connect(G_OBJECT(combo_filter), "changed", G_CALLBACK(combo_filter_changed), NULL);

    gtk_grid_attach( GTK_GRID(right_panel), combo_filter, 0, 0, 1, 1);

    GtkWidget* panel = gtk_list_box_new();
    gtk_widget_set_vexpand(panel, TRUE);
    gtk_list_box_set_selection_mode(GTK_LIST_BOX(panel), GTK_SELECTION_MULTIPLE);

    gtk_grid_attach_next_to( GTK_GRID(right_panel), panel, combo_filter, GTK_POS_BOTTOM, 1, 1);

    int i = 0;

    while (oflag2label[i].mask)
    {
        GtkWidget* box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

        gtk_widget_set_halign(box, GTK_ALIGN_START);

        gtk_box_pack_end(GTK_BOX(box), gtk_label_new(oflag2label[i].label), FALSE, FALSE, 0);

        GtkWidget* check = gtk_check_button_new();
        gtk_box_pack_end(GTK_BOX(box), check, FALSE, FALSE, 0);
        g_signal_connect(G_OBJECT(check), "toggled", G_CALLBACK(check_filter_toggled), oflag2label + i);

        gtk_list_box_insert(GTK_LIST_BOX(panel), box, i);
        i++;
    }

    *listbox = panel;

    return right_panel;
}

#endif

static gboolean on_scroll_event(GtkWidget* widget, GdkEvent* event, gpointer user_data)
{
    UNUSED(widget);
    UNUSED(event);
    UNUSED(user_data);

    // double dx, dy;
    // gdk_event_get_scroll_deltas(event, &dx, &dy);

    gtk_widget_queue_draw(g_ectx.drawing_tracks);

    return FALSE;
}

static void activate(GtkApplication* app, gpointer user_data)
{
    UNUSED(user_data);

    GtkWidget* window;

    // window title

    int needed = 30 + strlen(g_ectx.pathDb) + strlen(g_ectx.pathLas);
    int i;

    for (i = 0; i < g_ectx.ntracks; i++)
    {
        needed += strlen(g_ectx.tracks[i]->name);
    }

    char* title = malloc(needed);

    int tpos = sprintf(title, "LA Explorer - %s %s", g_ectx.pathDb, g_ectx.pathLas);

    for (i = 0; i < g_ectx.ntracks; i++)
    {
        tpos += sprintf(title + tpos, " %s", g_ectx.tracks[i]->name);
    }

    // main window

    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), title);

    free(title);

    gtk_widget_add_events(GTK_WIDGET(window), GDK_CONFIGURE);
    g_signal_connect(G_OBJECT(window), "configure-event", G_CALLBACK(frame_callback), NULL);
    g_signal_connect(G_OBJECT (window), "key_press_event", G_CALLBACK (on_key_press), NULL);

    GtkWidget* grid = gtk_grid_new();

    // options bar
    GtkWidget* bar = setup_toolbar();
    gtk_grid_attach(GTK_GRID(grid), bar, 0, 0, 2, 1);

    // overlap display
    g_ectx.scrolled_wnd = GTK_SCROLLED_WINDOW(gtk_scrolled_window_new(NULL, NULL));
    gtk_widget_set_hexpand(GTK_WIDGET(g_ectx.scrolled_wnd), TRUE);
    gtk_widget_set_vexpand(GTK_WIDGET(g_ectx.scrolled_wnd), TRUE);

    g_ectx.drawing_area = gtk_drawing_area_new();
    gtk_widget_add_events(g_ectx.drawing_area, GDK_BUTTON_PRESS_MASK | GDK_POINTER_MOTION_MASK);

    g_signal_connect(G_OBJECT(g_ectx.drawing_area), "draw", G_CALLBACK(read_draw_callback), NULL);
    g_signal_connect(G_OBJECT(g_ectx.drawing_area), "button-press-event", G_CALLBACK(read_button_press), NULL);
    g_signal_connect(G_OBJECT(g_ectx.drawing_area), "motion-notify-event", G_CALLBACK(read_mouse_move), NULL);

    g_signal_connect(G_OBJECT(g_ectx.scrolled_wnd), "scroll-event", G_CALLBACK(on_scroll_event), NULL);

    g_ectx.drawing_tracks = gtk_drawing_area_new();
    g_signal_connect(G_OBJECT(g_ectx.drawing_tracks), "draw", G_CALLBACK(track_draw_callback), NULL);
    gtk_widget_set_hexpand(GTK_WIDGET(g_ectx.drawing_tracks), TRUE);
    gtk_widget_set_size_request(GTK_WIDGET(g_ectx.drawing_tracks), -1, PADDING + (g_ectx.ntracks + 1) * DEF_LINEWIDTH + PADDING);
    gtk_grid_attach(GTK_GRID(grid), GTK_WIDGET(g_ectx.drawing_tracks), 0, 1, 1, 1);

    gtk_container_add(GTK_CONTAINER(g_ectx.scrolled_wnd), g_ectx.drawing_area);
    gtk_widget_set_can_focus(GTK_WIDGET(g_ectx.scrolled_wnd), TRUE);

    gtk_grid_attach(GTK_GRID(grid), GTK_WIDGET(g_ectx.scrolled_wnd), 0, 2, 1, 1);

    // overlap details bar

    bar = g_ectx.overlap_details = gtk_label_new("");
    gtk_widget_set_halign(bar, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), bar, 0, 3, 2, 1);

#if GTK_CHECK_VERSION(3, 10, 0)
    // flags / filter panel
    GtkWidget* panel = setup_filter_panel(&(g_ectx.filter_panel));
    gtk_grid_attach(GTK_GRID(grid), panel, 1, 1, 1, 3);
#endif

    gtk_container_add(GTK_CONTAINER(window), grid);

    gtk_widget_show_all(window);

    view_new(0);

    load_overlaps();
}

static int command_line(GApplication* application, GApplicationCommandLine* cmdline)
{
    gchar** argv;
    gint argc;

    argv = g_application_command_line_get_arguments(cmdline, &argc);

    gchar* qname = TRACK_Q;
    gchar* tname = TRACK_TRIM;
    gchar* sname = TRACK_SOURCE;
    gchar** inames = NULL;
    gchar** remaining = NULL;

    GOptionEntry entries[] = {
            {"quality", 'q', 0, G_OPTION_ARG_STRING, &qname, "name of the track containing the read quality annotation", NULL},
            {"trim", 't', 0, G_OPTION_ARG_STRING, &tname, "name of the track containing the read trim annotation", NULL},
            {"source", 's', 0, G_OPTION_ARG_STRING, &sname, "track containing a mapping of the read ids back to the pre-patching read ids", NULL},
            {"interval", 'i', 0, G_OPTION_ARG_STRING_ARRAY, &inames, "include the interval annotation track in the display", NULL},
            {G_OPTION_REMAINING, 0, 0, G_OPTION_ARG_STRING_ARRAY, &remaining, NULL, NULL},
            {NULL, 0, 0, 0, NULL, NULL, NULL}
    };

    GOptionContext* optctx = g_option_context_new("database input.las");
    g_option_context_add_main_entries(optctx, entries, NULL);
    GError* error = NULL;

    if ( ! g_option_context_parse(optctx, &argc, &argv, &error) )
    {
        g_print("failed to parse command line: %s\n", error->message);
        exit(1);
    }

    if (remaining == NULL || remaining[0] == NULL || remaining[1] == NULL)
    {
        gchar* help = g_option_context_get_help(optctx, TRUE, NULL);
        g_print("%s", help);
        g_free(help);

        exit(1);
    }

    g_ectx.pathDb = remaining[0];

    if (Open_DB(g_ectx.pathDb, &g_ectx.db))
    {
        printf("could not open '%s'\n", g_ectx.pathDb);
        exit(1);
    }

    int nreads = DB_NREADS(&(g_ectx.db));
    int blocks = DB_Blocks(g_ectx.pathDb);

    // open las files and their indices

    g_ectx.lasFiles = malloc(sizeof(FILE*) * blocks);
    bzero(g_ectx.lasFiles, sizeof(FILE*) * blocks);

    g_ectx.lasIndexFile = malloc(sizeof(int) * nreads);
    g_ectx.pathLas = remaining[1];

    int rid;

    for (rid = 0; rid < nreads; rid++)
    {
        g_ectx.lasIndexFile[rid] = -1;
    }

    char* num = strchr(g_ectx.pathLas, '#');

    if (num != NULL)
    {
        char* pathLas = malloc(strlen(g_ectx.pathLas) + 10);

        char* prefix = g_ectx.pathLas;
        char* suffix = num + 1;
        *num = '\0';

        int b;

        for (b = 1; b <= blocks; b++)
        {
            sprintf(pathLas, "%s%d%s", prefix, b, suffix);

            if ((g_ectx.lasFiles[b - 1] = fopen(pathLas, "r")) == NULL)
            {
                fprintf(stderr, "could not open '%s'\n", pathLas);
                exit(1);
            }

            if (b == 1)
            {
                g_ectx.lasIndex = lasidx_load(&(g_ectx.db), pathLas, 1);

                int j;

                for (j = 0; j < nreads; j++)
                {
                    if (g_ectx.lasIndex[j] != 0)
                    {
                        g_ectx.lasIndexFile[j] = 0;
                    }
                }
            }
            else
            {
                lasidx* lasIndex = lasidx_load(&(g_ectx.db), pathLas, 1);

                int j;

                for (j = 0; j < nreads; j++)
                {
                    if (lasIndex[j] == 0)
                    {
                        continue;
                    }

                    if (g_ectx.lasIndex[j] != 0)
                    {
                        fprintf(stderr, "error: read id %d used in two indices\n", j);

                        // printf("%lld %lld\n", lasIndex[j], g_ectx.lasIndex[j]);

                        exit(1);
                    }

                    g_ectx.lasIndex[j] = lasIndex[j];
                    g_ectx.lasIndexFile[j] = b - 1;
                }

                free(lasIndex);
            }

        }

        *num = '#';
    }
    else
    {
        if ((g_ectx.lasFiles[0] = fopen(g_ectx.pathLas, "r")) == NULL)
        {
            fprintf(stderr, "could not open '%s'\n", g_ectx.pathLas);
            exit(1);
        }

        g_ectx.lasIndex = lasidx_load(&(g_ectx.db), g_ectx.pathLas, 1);

        int j;

        for (j = 0; j < nreads; j++)
        {
            if (g_ectx.lasIndex[j] != 0)
            {
                g_ectx.lasIndexFile[j] = 0;
            }
        }
    }

    g_ectx.qtrack = track_load(&g_ectx.db, qname);
    g_ectx.srctrack = track_load(&g_ectx.db, sname);

    int i = 0;
    if (inames != NULL)
    {
        while ( inames[i] != NULL )
        {
            char* iname = inames[i];

            if (g_ectx.ntracks >= g_ectx.maxtracks)
            {
                g_ectx.maxtracks = 1.2 * g_ectx.maxtracks + 100;
                g_ectx.tracks = realloc(g_ectx.tracks, sizeof(HITS_TRACK*) * g_ectx.maxtracks);
                g_ectx.track_type = realloc(g_ectx.track_type, sizeof(int) * g_ectx.maxtracks);
            }

            g_ectx.track_type[g_ectx.ntracks] = TRACK_INTERVAL;

            if (!(g_ectx.tracks[g_ectx.ntracks] = track_load(&g_ectx.db, iname)))
            {
                printf("failed to load track %s\n", iname);
                exit(1);
            }

            g_ectx.ntracks++;
            i++;
        }
    }

    // init

    g_ectx.line_width = DEF_LINEWIDTH;
    g_ectx.viewstack_cur = -1;
    g_ectx.viewstack_fill = -1;
    g_ectx.viewstack_max = 0;
    g_ectx.viewstack = NULL;
    g_ectx.omax = 500;
    g_ectx.ovls = malloc(sizeof(OverlapDetails) * g_ectx.omax);
    g_ectx.ovls_sorted = malloc(sizeof(OverlapDetails*) * g_ectx.omax);
    g_ectx.ovl_display = malloc(sizeof(int) * g_ectx.omax);

    g_ectx.hmax = 0;
    g_ectx.hcur = 0;
    g_ectx.highlight = NULL;

    g_ectx.rid_hover_highlight = -1;

    // view_new(0);

    for (i = 0; i < HIGHLIGHT_COLORS; i++)
    {
        get_unique_color(i, HIGHLIGHT_COLORS, g_ectx.highlight_color + i);
    }

    // get trace info from header

    ovl_header_novl novl;

    ovl_header_read(g_ectx.lasFiles[0], &novl, &(g_ectx.twidth));
    g_ectx.tbytes = TBYTES(g_ectx.twidth);

    // enter even loop

    g_option_context_free(optctx);

    g_application_activate(application);

    return 0;
}

int main(int argc, char** argv)
{
    GtkApplication* app;
    int status;

    bzero(&g_ectx, sizeof(ExplorerContext));

    // g_object_set (gtk_settings_get_default (), "gtk-enable-animations", FALSE, NULL);

    app = gtk_application_new(NULL, G_APPLICATION_HANDLES_COMMAND_LINE);

    g_signal_connect(app, "command-line", G_CALLBACK(command_line), NULL);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);

    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);

    return status;
}
