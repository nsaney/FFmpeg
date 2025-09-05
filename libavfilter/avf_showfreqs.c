/*
 * Copyright (c) 2015 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <float.h>
#include <math.h>

#include "libavutil/mem.h"
#include "libavutil/tx.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "audio.h"
#include "filters.h"
#include "formats.h"
#include "video.h"
#include "avfilter.h"
#include "window_func.h"

enum DataMode       { MAGNITUDE, PHASE, DELAY, NB_DATA };
enum DisplayMode    { LINE, BAR, DOT, PIPE, NB_MODES };
enum ChannelMode    { COMBINED, SEPARATE, NB_CMODES };
enum FrequencyScale { FS_LINEAR, FS_LOG, FS_RLOG, NB_FSCALES };
enum AmplitudeScale { AS_LINEAR, AS_SQRT, AS_CBRT, AS_LOG, NB_ASCALES };

typedef struct RectangleBounds {
    int x_lo;
    int x_hi;
    int y_lo;
    int y_hi;
} RectangleBounds;

typedef struct ShowFreqsPipeModeContext {
    char *pipe_border_color;
    uint8_t bd[4];
    char *pipe_padding_color;
    uint8_t pg[4];
    int pipe_min_width;
    int pipe_curr_unused_min_y;
    int pipe_next_x0;
} ShowFreqsPipeModeContext;

typedef struct ShowFreqsContext {
    const AVClass *class;
    int w, h;
    int mode;
    int data_mode;
    int cmode;
    int fft_size;
    int ascale, fscale;
    int avg;
    int win_func;
    char *ch_layout_str;
    uint8_t *bypass;
    AVChannelLayout ch_layout;
    AVTXContext *fft;
    av_tx_fn tx_fn;
    AVComplexFloat **fft_input;
    AVComplexFloat **fft_data;
    AVFrame *window;
    float **avg_data;
    float *window_func_lut;
    float overlap;
    float minamp;
    int hop_size;
    int nb_channels;
    int nb_draw_channels;
    int nb_freq;
    int win_size;
    float scale;
    char *colors;
    int64_t pts;
    int64_t old_pts;
    AVRational frame_rate;
    ShowFreqsPipeModeContext pipe_mode_ctx;
} ShowFreqsContext;

#define OFFSET(x) offsetof(ShowFreqsContext, x)
#define PIPE_MODE_OFFSET(x) OFFSET(pipe_mode_ctx) + offsetof(ShowFreqsPipeModeContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption showfreqs_options[] = {
    { "size", "set video size", OFFSET(w), AV_OPT_TYPE_IMAGE_SIZE, {.str = "1024x512"}, 0, 0, FLAGS },
    { "s",    "set video size", OFFSET(w), AV_OPT_TYPE_IMAGE_SIZE, {.str = "1024x512"}, 0, 0, FLAGS },
    { "rate", "set video rate",  OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str = "25"}, 0, INT_MAX, FLAGS },
    { "r",    "set video rate",  OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str = "25"}, 0, INT_MAX, FLAGS },
    { "mode", "set display mode", OFFSET(mode), AV_OPT_TYPE_INT, {.i64=BAR}, 0, NB_MODES-1, FLAGS, .unit = "mode" },
        { "line", "show lines",  0, AV_OPT_TYPE_CONST, {.i64=LINE},   0, 0, FLAGS, .unit = "mode" },
        { "bar",  "show bars",   0, AV_OPT_TYPE_CONST, {.i64=BAR},    0, 0, FLAGS, .unit = "mode" },
        { "dot",  "show dots",   0, AV_OPT_TYPE_CONST, {.i64=DOT},    0, 0, FLAGS, .unit = "mode" },
        { "pipe", "show pipes",  0, AV_OPT_TYPE_CONST, {.i64=PIPE},   0, 0, FLAGS, .unit = "mode" },
    { "ascale", "set amplitude scale", OFFSET(ascale), AV_OPT_TYPE_INT, {.i64=AS_LOG}, 0, NB_ASCALES-1, FLAGS, .unit = "ascale" },
        { "lin",  "linear",      0, AV_OPT_TYPE_CONST, {.i64=AS_LINEAR}, 0, 0, FLAGS, .unit = "ascale" },
        { "sqrt", "square root", 0, AV_OPT_TYPE_CONST, {.i64=AS_SQRT},   0, 0, FLAGS, .unit = "ascale" },
        { "cbrt", "cubic root",  0, AV_OPT_TYPE_CONST, {.i64=AS_CBRT},   0, 0, FLAGS, .unit = "ascale" },
        { "log",  "logarithmic", 0, AV_OPT_TYPE_CONST, {.i64=AS_LOG},    0, 0, FLAGS, .unit = "ascale" },
    { "fscale", "set frequency scale", OFFSET(fscale), AV_OPT_TYPE_INT, {.i64=FS_LINEAR}, 0, NB_FSCALES-1, FLAGS, .unit = "fscale" },
        { "lin",  "linear",              0, AV_OPT_TYPE_CONST, {.i64=FS_LINEAR}, 0, 0, FLAGS, .unit = "fscale" },
        { "log",  "logarithmic",         0, AV_OPT_TYPE_CONST, {.i64=FS_LOG},    0, 0, FLAGS, .unit = "fscale" },
        { "rlog", "reverse logarithmic", 0, AV_OPT_TYPE_CONST, {.i64=FS_RLOG},   0, 0, FLAGS, .unit = "fscale" },
    { "win_size", "set window size", OFFSET(fft_size), AV_OPT_TYPE_INT, {.i64=2048}, 16, 65536, FLAGS },
    WIN_FUNC_OPTION("win_func", OFFSET(win_func), FLAGS, WFUNC_HANNING),
    { "overlap",  "set window overlap", OFFSET(overlap), AV_OPT_TYPE_FLOAT, {.dbl=1.}, 0., 1., FLAGS },
    { "averaging", "set time averaging", OFFSET(avg), AV_OPT_TYPE_INT, {.i64=1}, 0, INT32_MAX, FLAGS },
    { "colors", "set channels colors", OFFSET(colors), AV_OPT_TYPE_STRING, {.str = "red|green|blue|yellow|orange|lime|pink|magenta|brown" }, 0, 0, FLAGS },
    { "cmode", "set channel mode", OFFSET(cmode), AV_OPT_TYPE_INT, {.i64=COMBINED}, 0, NB_CMODES-1, FLAGS, .unit = "cmode" },
        { "combined", "show all channels in same window",  0, AV_OPT_TYPE_CONST, {.i64=COMBINED}, 0, 0, FLAGS, .unit = "cmode" },
        { "separate", "show each channel in own window",   0, AV_OPT_TYPE_CONST, {.i64=SEPARATE}, 0, 0, FLAGS, .unit = "cmode" },
    { "minamp",  "set minimum amplitude", OFFSET(minamp), AV_OPT_TYPE_FLOAT, {.dbl=1e-6}, FLT_MIN, 1e-6, FLAGS },
    { "data", "set data mode", OFFSET(data_mode), AV_OPT_TYPE_INT, {.i64=MAGNITUDE}, 0, NB_DATA-1, FLAGS, .unit = "data" },
        { "magnitude", "show magnitude",  0, AV_OPT_TYPE_CONST, {.i64=MAGNITUDE}, 0, 0, FLAGS, .unit = "data" },
        { "phase",     "show phase",      0, AV_OPT_TYPE_CONST, {.i64=PHASE},     0, 0, FLAGS, .unit = "data" },
        { "delay",     "show group delay",0, AV_OPT_TYPE_CONST, {.i64=DELAY},     0, 0, FLAGS, .unit = "data" },
    { "channels", "set channels to draw", OFFSET(ch_layout_str), AV_OPT_TYPE_STRING, {.str="all"}, 0, 0, FLAGS },
    { "pipe_border_color",  "set pipe_border_color",  PIPE_MODE_OFFSET(pipe_border_color),  AV_OPT_TYPE_STRING, {.str="0x3f3f3f"},  0,  0, FLAGS },
    { "pipe_padding_color", "set pipe_padding_color", PIPE_MODE_OFFSET(pipe_padding_color), AV_OPT_TYPE_STRING, {.str="0xdfdfdf"},  0,  0, FLAGS },
    { "pipe_min_width",     "set pipe_min_width",     PIPE_MODE_OFFSET(pipe_min_width),     AV_OPT_TYPE_INT,    {.i64=14},       0, 65536, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(showfreqs);

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    AVFilterFormats *formats = NULL;
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE };
    static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE };
    int ret;

    /* set input audio formats */
    formats = ff_make_format_list(sample_fmts);
    if ((ret = ff_formats_ref(formats, &cfg_in[0]->formats)) < 0)
        return ret;

    /* set output video format */
    formats = ff_make_format_list(pix_fmts);
    if ((ret = ff_formats_ref(formats, &cfg_out[0]->formats)) < 0)
        return ret;

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    FilterLink *l = ff_filter_link(outlink);
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    ShowFreqsContext *s = ctx->priv;
    float overlap, scale = 1.f;
    int i, ret;

    s->old_pts = AV_NOPTS_VALUE;
    s->nb_freq = s->fft_size / 2;
    s->win_size = s->fft_size;
    av_tx_uninit(&s->fft);
    ret = av_tx_init(&s->fft, &s->tx_fn, AV_TX_FLOAT_FFT, 0, s->fft_size, &scale, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Unable to create FFT context. "
               "The window size might be too high.\n");
        return ret;
    }

    /* FFT buffers: x2 for each (display) channel buffer.
     * Note: we use free and malloc instead of a realloc-like function to
     * make sure the buffer is aligned in memory for the FFT functions. */
    for (i = 0; i < s->nb_channels; i++) {
        av_freep(&s->fft_input[i]);
        av_freep(&s->fft_data[i]);
        av_freep(&s->avg_data[i]);
    }
    av_freep(&s->bypass);
    av_freep(&s->fft_input);
    av_freep(&s->fft_data);
    av_freep(&s->avg_data);
    s->nb_channels = inlink->ch_layout.nb_channels;

    s->bypass = av_calloc(s->nb_channels, sizeof(*s->bypass));
    if (!s->bypass)
        return AVERROR(ENOMEM);
    s->fft_input = av_calloc(s->nb_channels, sizeof(*s->fft_input));
    if (!s->fft_input)
        return AVERROR(ENOMEM);
    s->fft_data = av_calloc(s->nb_channels, sizeof(*s->fft_data));
    if (!s->fft_data)
        return AVERROR(ENOMEM);
    s->avg_data = av_calloc(s->nb_channels, sizeof(*s->avg_data));
    if (!s->avg_data)
        return AVERROR(ENOMEM);
    for (i = 0; i < s->nb_channels; i++) {
        s->fft_input[i] = av_calloc(FFALIGN(s->win_size, 512), sizeof(**s->fft_input));
        s->fft_data[i] = av_calloc(FFALIGN(s->win_size, 512), sizeof(**s->fft_data));
        s->avg_data[i] = av_calloc(s->nb_freq, sizeof(**s->avg_data));
        if (!s->fft_data[i] || !s->avg_data[i] || !s->fft_input[i])
            return AVERROR(ENOMEM);
    }

    /* pre-calc windowing function */
    s->window_func_lut = av_realloc_f(s->window_func_lut, s->win_size,
                                      sizeof(*s->window_func_lut));
    if (!s->window_func_lut)
        return AVERROR(ENOMEM);
    generate_window_func(s->window_func_lut, s->win_size, s->win_func, &overlap);
    if (s->overlap == 1.)
        s->overlap = overlap;
    s->hop_size = (1. - s->overlap) * s->win_size;
    if (s->hop_size < 1) {
        av_log(ctx, AV_LOG_ERROR, "overlap %f too big\n", s->overlap);
        return AVERROR(EINVAL);
    }

    for (s->scale = 0, i = 0; i < s->win_size; i++) {
        s->scale += s->window_func_lut[i] * s->window_func_lut[i];
    }

    s->window = ff_get_audio_buffer(inlink, s->win_size * 2);
    if (!s->window)
        return AVERROR(ENOMEM);

    l->frame_rate = s->frame_rate;
    outlink->time_base = av_inv_q(l->frame_rate);
    outlink->sample_aspect_ratio = (AVRational){1,1};
    outlink->w = s->w;
    outlink->h = s->h;

    ret = av_channel_layout_copy(&s->ch_layout, &inlink->ch_layout);
    if (ret < 0)
        return ret;
    s->nb_draw_channels = s->nb_channels;

    if (strcmp(s->ch_layout_str, "all")) {
        int nb_draw_channels = 0;
        av_channel_layout_from_string(&s->ch_layout,
                                      s->ch_layout_str);

        for (int ch = 0; ch < s->nb_channels; ch++) {
            const enum AVChannel channel = av_channel_layout_channel_from_index(&inlink->ch_layout, ch);

            s->bypass[ch] = av_channel_layout_index_from_channel(&s->ch_layout, channel) < 0;
            nb_draw_channels += s->bypass[ch] == 0;
        }

        s->nb_draw_channels = nb_draw_channels;
    }

    return 0;
}

static inline void draw_dot(AVFrame *out, int x, int y, const uint8_t fg[4])
{

    uint32_t color = AV_RL32(out->data[0] + y * out->linesize[0] + x * 4);

    if ((color & 0xffffff) != 0)
        AV_WL32(out->data[0] + y * out->linesize[0] + x * 4, AV_RL32(fg) | color);
    else
        AV_WL32(out->data[0] + y * out->linesize[0] + x * 4, AV_RL32(fg));
}

static inline void fill_rectangle(AVFrame *out, const uint8_t fg[4],
                                  RectangleBounds *rb,
                                  int x0, int xf, int y0, int yf)
{
    int x;
    if (xf < x0) { x = xf; xf = x0; x0 = x; }
    if (x0 < rb->x_lo) x0 = rb->x_lo;
    if (xf > rb->x_hi) xf = rb->x_hi;
    int y;
    if (yf < y0) { y = yf; yf = y0; y0 = y; }
    if (y0 < rb->y_lo) y0 = rb->y_lo;
    if (yf > rb->y_hi) yf = rb->y_hi;
    for (x = x0; x < xf; x++)
        for (y = y0; y < yf; y++)
            draw_dot(out, x, y, fg);
}

static inline void fill_and_mirror_rectangle(AVFrame *out, const uint8_t fg[4],
                                             RectangleBounds *rb,
                                             int x0, int xf, int y0, int yf)
{
    int x_md = (rb->x_hi + rb->x_lo) / 2 + 1;
    int rx0 = rb->x_hi - (x0 - rb->x_lo);
    int rxf = rb->x_hi - (xf - rb->x_lo);
    int y_md = (rb->y_hi + rb->y_lo) / 2 + 1;
    int ry0 = rb->y_hi - (y0 - rb->y_lo);
    int ryf = rb->y_hi - (yf - rb->y_lo);
    fill_rectangle(out, fg, &(RectangleBounds){ rb->x_lo,     x_md, rb->y_lo,     y_md },  x0,  xf,  y0,  yf);
    fill_rectangle(out, fg, &(RectangleBounds){     x_md, rb->x_hi, rb->y_lo,     y_md }, rx0, rxf,  y0,  yf);
    fill_rectangle(out, fg, &(RectangleBounds){ rb->x_lo,     x_md,     y_md, rb->y_hi },  x0,  xf, ry0, ryf);
    fill_rectangle(out, fg, &(RectangleBounds){     x_md, rb->x_hi,     y_md, rb->y_hi }, rx0, rxf, ry0, ryf);
}

static int get_sx(ShowFreqsContext *s, int f)
{
    switch (s->fscale) {
    case FS_LINEAR:
        return (s->w/(float)s->nb_freq)*f;
    case FS_LOG:
        return s->w-pow(s->w, (s->nb_freq-f-1)/(s->nb_freq-1.));
    case FS_RLOG:
        return pow(s->w, f/(s->nb_freq-1.));
    }

    return 0;
}

static float get_bsize(ShowFreqsContext *s, int f)
{
    switch (s->fscale) {
    case FS_LINEAR:
        return s->w/(float)s->nb_freq;
    case FS_LOG:
        return pow(s->w, (s->nb_freq-f-1)/(s->nb_freq-1.))-
               pow(s->w, (s->nb_freq-f-2)/(s->nb_freq-1.));
    case FS_RLOG:
        return pow(s->w, (f+1)/(s->nb_freq-1.))-
               pow(s->w,  f   /(s->nb_freq-1.));
    }

    return 1.;
}

static inline void plot_freq(ShowFreqsContext *s, int ch,
                             double a, int f, uint8_t fg[4], int *prev_y,
                             AVFrame *out, AVFilterLink *outlink)
{
    FilterLink *outl = ff_filter_link(outlink);
    const int w = s->w;
    const float min = s->minamp;
    const float avg = s->avg_data[ch][f];
    const float bsize = get_bsize(s, f);
    const int sx = get_sx(s, f);
    int end = outlink->h;
    RectangleBounds rb;
    int top;
    ShowFreqsPipeModeContext *p = &s->pipe_mode_ctx;
    int x_md, y_md;
    int pipe_width, pipe_gap;
    int x, y, i;

    switch(s->ascale) {
    case AS_SQRT:
        a = 1.0 - sqrt(a);
        break;
    case AS_CBRT:
        a = 1.0 - cbrt(a);
        break;
    case AS_LOG:
        a = log(av_clipd(a, min, 1)) / log(min);
        break;
    case AS_LINEAR:
        a = 1.0 - a;
        break;
    }

    switch (s->cmode) {
    case COMBINED:
        top = 0;
        end = outlink->h;
        y = a * outlink->h - 1;
        break;
    case SEPARATE:
        top = (outlink->h / s->nb_draw_channels) * ch;
        end = (outlink->h / s->nb_draw_channels) * (ch + 1);
        y = top + a * (outlink->h / s->nb_draw_channels) - 1;
        break;
    default:
        av_assert0(0);
    }
    if (y < 0)
        return;

    switch (s->avg) {
    case 0:
        y = s->avg_data[ch][f] = !outl->frame_count_in ? y : FFMIN(0, y);
        break;
    case 1:
        break;
    default:
        s->avg_data[ch][f] = avg + y * (y - avg) / (FFMIN(outl->frame_count_in + 1, s->avg) * (float)y);
        y = av_clip(s->avg_data[ch][f], 0, outlink->h - 1);
        break;
    }

    switch(s->mode) {
    case LINE:
        if (*prev_y == -1) {
            *prev_y = y;
        }
        if (y <= *prev_y) {
            for (x = sx + 1; x < sx + bsize && x < w; x++)
                draw_dot(out, x, y, fg);
            for (i = y; i <= *prev_y; i++)
                draw_dot(out, sx, i, fg);
        } else {
            for (i = *prev_y; i <= y; i++)
                draw_dot(out, sx, i, fg);
            for (x = sx + 1; x < sx + bsize && x < w; x++)
                draw_dot(out, x, i - 1, fg);
        }
        *prev_y = y;
        break;
    case BAR:
        for (x = sx; x < sx + bsize && x < w; x++)
            for (i = y; i < end; i++)
                draw_dot(out, x, i, fg);
        break;
    case DOT:
        for (x = sx; x < sx + bsize && x < w; x++)
            draw_dot(out, x, y, fg);
        break;
    case PIPE:
        rb.x_lo = sx;
        rb.x_hi = sx + bsize - 1;
        pipe_width = rb.x_hi - rb.x_lo;
        if (pipe_width < p->pipe_min_width && y < p->pipe_curr_unused_min_y)
            p->pipe_curr_unused_min_y = y;
        if (p->pipe_min_width + 1 <= rb.x_lo - p->pipe_next_x0) {
            y = p->pipe_curr_unused_min_y;
            rb.x_lo = p->pipe_next_x0;
            rb.x_hi = rb.x_lo + p->pipe_min_width;
            pipe_width = p->pipe_min_width;
        }
        if (pipe_width < p->pipe_min_width)
            break;
        if (rb.x_hi > w)
            break;
        p->pipe_curr_unused_min_y = end;
        p->pipe_next_x0 = rb.x_hi + 1;
        x_md = (rb.x_lo + rb.x_hi) / 2 + 1;
        //      top  - - - y - -|- - - - - -  end
        // BAR   [........]<====|==============>
        // PIPE  [....<=========|=========>....]
        pipe_gap = (y - top) / 2;
        rb.y_lo = y - pipe_gap;
        rb.y_hi = end - pipe_gap;
        y_md = (rb.y_lo + rb.y_hi) / 2 + 1;
        if (22 <= pipe_width) {
            // A  B  C  D  E  E  ... E  E  D  C  B  A
            // .  .  .  .  ## ## ...
            // .  .  ## ## [] []
            // .  ## [] [] fg fg
            // .  ## [] fg fg fg
            // ## [] fg fg fg fg
            // ## [] fg fg fg fg ...
            // ...            ...
            // cols A
            x = rb.x_lo;
            y = rb.y_lo + 8;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y_md);
            // cols B
            x = rb.x_lo + 2;
            y = rb.y_lo + 4;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y + 4);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x + 2, y + 4, y_md );
            // cols C
            x = rb.x_lo + 4;
            y = rb.y_lo + 2;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x + 2, y + 2, y + 6);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x + 2, y + 6, y_md );
            // cols D
            x = rb.x_lo + 6;
            y = rb.y_lo + 2;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x + 2, y + 2, y + 4);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x + 2, y + 4, y_md );
            // cols E
            x = rb.x_lo + 8;
            y = rb.y_lo;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x_md , y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x_md , y + 2, y + 4);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x_md , y + 4, y_md );
        } else if (18 <= pipe_width) {
            // A  B  C  D  D  ... D  D  C  B  A
            // .  .  .  ## ## ...
            // .  .  ## [] []
            // .  ## [] fg fg
            // ## [] fg fg fg ...
            // ## [] fg fg fg ...
            // ...      ...
            // cols A
            x = rb.x_lo;
            y = rb.y_lo + 6;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y_md );
            // cols B
            x = rb.x_lo + 2;
            y = rb.y_lo + 4;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x + 2, y + 2, y_md );
            // cols C
            x = rb.x_lo + 4;
            y = rb.y_lo + 2;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x + 2, y + 2, y + 4);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x + 2, y + 4, y_md );
            // cols D
            x = rb.x_lo + 6;
            y = rb.y_lo;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x_md , y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x_md , y + 2, y + 4);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x_md , y + 4, y_md );
        } else if (14 <= pipe_width) {
            // A  B  C  C  ... C  C  B  A
            // .  .  ## ## ...
            // .  ## [] []
            // ## [] fg fg
            // ## [] fg fg ...
            // ...      ...
            // cols A
            x = rb.x_lo;
            y = rb.y_lo + 4;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y_md );
            // cols B
            x = rb.x_lo + 2;
            y = rb.y_lo + 2;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x + 2, y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x + 2, y + 2, y_md );
            // cols C
            x = rb.x_lo + 4;
            y = rb.y_lo;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x_md , y    , y + 2);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x_md , y + 2, y + 4);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x_md , y + 4, y_md );
        } else {
            if (bsize < 2)
                x_md = rb.x_hi = rb.x_lo + 1;
            x = rb.x_lo;
            y = rb.y_lo;
            fill_and_mirror_rectangle(out, p->bd, &rb, x, x_md , y    , y + 1);
            fill_and_mirror_rectangle(out, p->pg, &rb, x, x_md , y + 1, y + 2);
            fill_and_mirror_rectangle(out,    fg, &rb, x, x_md , y + 2, y_md );
        }
        break;
    }
}

static int plot_freqs(AVFilterLink *inlink, int64_t pts)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    ShowFreqsContext *s = ctx->priv;
    ShowFreqsPipeModeContext *p = &s->pipe_mode_ctx;
    AVFrame *in = s->window;
    const int win_size = s->win_size;
    char *colors, *color, *saveptr = NULL;
    AVFrame *out;
    int ch, n;

    /* fill FFT input with the number of samples available */
    for (ch = 0; ch < s->nb_channels; ch++) {
        const float *p = (float *)in->extended_data[ch];

        if (s->bypass[ch])
            continue;

        for (n = 0; n < win_size; n++) {
            s->fft_input[ch][n].re = p[n] * s->window_func_lut[n];
            s->fft_input[ch][n].im = 0;
        }
    }

    /* run FFT on each samples set */
    for (ch = 0; ch < s->nb_channels; ch++) {
        if (s->bypass[ch])
            continue;

        s->tx_fn(s->fft, s->fft_data[ch], s->fft_input[ch], sizeof(AVComplexFloat));
    }

    s->pts = av_rescale_q(pts, inlink->time_base, outlink->time_base);
    if (s->old_pts >= s->pts)
        return 0;
    s->old_pts = s->pts;

#define RE(x, ch) s->fft_data[ch][x].re
#define IM(x, ch) s->fft_data[ch][x].im
#define M(a, b) (sqrt((a) * (a) + (b) * (b)))
#define P(a, b) (atan2((b), (a)))

    colors = av_strdup(s->colors);
    if (!colors)
        return AVERROR(ENOMEM);

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_free(colors);
        return AVERROR(ENOMEM);
    }

    for (n = 0; n < outlink->h; n++)
        memset(out->data[0] + out->linesize[0] * n, 0, outlink->w * 4);

    for (ch = 0; ch < s->nb_channels; ch++) {
        uint8_t fg[4] = { 0xff, 0xff, 0xff, 0xff };
        int prev_y = -1, f;
        double a;

        color = av_strtok(ch == 0 ? colors : NULL, " |", &saveptr);
        if (color)
            av_parse_color(fg, color, -1, ctx);

        if (s->mode == PIPE) {
            if (p->pipe_border_color)
                av_parse_color(p->bd, p->pipe_border_color, -1, ctx);
            if (p->pipe_padding_color)
                av_parse_color(p->pg, p->pipe_padding_color, -1, ctx);
            p->pipe_curr_unused_min_y = outlink->h;
            p->pipe_next_x0 = 0;
        }

        if (s->bypass[ch])
            continue;

        switch (s->data_mode) {
        case MAGNITUDE:
            for (f = 0; f < s->nb_freq; f++) {
                a = av_clipd(M(RE(f, ch), IM(f, ch)) / s->scale, 0, 1);

                plot_freq(s, ch, a, f, fg, &prev_y, out, outlink);
            }
            break;
        case PHASE:
            for (f = 0; f < s->nb_freq; f++) {
                a = av_clipd((M_PI + P(RE(f, ch), IM(f, ch))) / (2. * M_PI), 0, 1);

                plot_freq(s, ch, a, f, fg, &prev_y, out, outlink);
            }
            break;
        case DELAY:
            for (f = 0; f < s->nb_freq; f++) {
                a = av_clipd((M_PI - P(IM(f, ch) * RE(f-1, ch) - IM(f-1, ch) * RE(f, ch),
                                       RE(f, ch) * RE(f-1, ch) + IM(f, ch) * IM(f-1, ch))) / (2. * M_PI), 0, 1);

                plot_freq(s, ch, a, f, fg, &prev_y, out, outlink);
            }
            break;
        }
    }

    av_free(colors);
    out->pts = s->pts;
    out->duration = 1;
    out->sample_aspect_ratio = (AVRational){1,1};
    return ff_filter_frame(outlink, out);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    ShowFreqsContext *s = ctx->priv;
    const int offset = s->win_size - s->hop_size;
    int64_t pts = in->pts;

    for (int ch = 0; ch < in->ch_layout.nb_channels; ch++) {
        float *dst = (float *)s->window->extended_data[ch];

        memmove(dst, &dst[s->hop_size], offset * sizeof(float));
        memcpy(&dst[offset], in->extended_data[ch], in->nb_samples * sizeof(float));
        memset(&dst[offset + in->nb_samples], 0, (s->hop_size - in->nb_samples) * sizeof(float));
    }

    av_frame_free(&in);

    return plot_freqs(inlink, pts);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    ShowFreqsContext *s = ctx->priv;
    AVFrame *in;
    int ret;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = ff_inlink_consume_samples(inlink, s->hop_size, s->hop_size, &in);
    if (ret < 0)
        return ret;

    if (ret > 0)
        ret = filter_frame(inlink, in);
    if (ret < 0)
        return ret;

    if (ff_inlink_queued_samples(inlink) >= s->hop_size) {
        ff_filter_set_ready(ctx, 10);
        return 0;
    }

    FF_FILTER_FORWARD_STATUS(inlink, outlink);
    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ShowFreqsContext *s = ctx->priv;
    int i;

    av_channel_layout_uninit(&s->ch_layout);
    av_tx_uninit(&s->fft);
    for (i = 0; i < s->nb_channels; i++) {
        if (s->fft_input)
            av_freep(&s->fft_input[i]);
        if (s->fft_data)
            av_freep(&s->fft_data[i]);
        if (s->avg_data)
            av_freep(&s->avg_data[i]);
    }
    av_freep(&s->bypass);
    av_freep(&s->fft_input);
    av_freep(&s->fft_data);
    av_freep(&s->avg_data);
    av_freep(&s->window_func_lut);
    av_frame_free(&s->window);
}

static const AVFilterPad showfreqs_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const FFFilter ff_avf_showfreqs = {
    .p.name        = "showfreqs",
    .p.description = NULL_IF_CONFIG_SMALL("Convert input audio to a frequencies video output."),
    .p.priv_class  = &showfreqs_class,
    .uninit        = uninit,
    .priv_size     = sizeof(ShowFreqsContext),
    .activate      = activate,
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(showfreqs_outputs),
    FILTER_QUERY_FUNC2(query_formats),
};
