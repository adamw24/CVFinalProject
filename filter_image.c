#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

// Collaborators: Adam Wang

void l1_normalize(image im)
{
    float norm_val = 0.0;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                norm_val += get_pixel(im, k, j, i);
            }
        }
    }
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                set_pixel(im, k, j, i, get_pixel(im, k, j, i) / norm_val);
            }
        }
    }
}

image make_box_filter(int w)
{
    image box_filter = make_image(w, w, 1);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            set_pixel(box_filter, i, j, 0, 1.0);
        }
    }
    l1_normalize(box_filter);
    return box_filter;
}

float convolve_pixel(image im, image filter, int x, int y, int c) {
    assert(im.c == filter.c || filter.c == 1);
    float sum = 0.0;
    for (int j = -filter.h / 2; j <= filter.h / 2; j++) {
        for (int k = -filter.w / 2; k <= filter.w / 2; k++) {
            float cur_im_pix = get_pixel(im, x + k, y + j, c);
            float cur_filter_pix = get_pixel(filter, k + filter.w / 2, j + filter.h / 2,
                                             MIN(filter.c - 1, c));
            sum += cur_im_pix * cur_filter_pix;
        }
    }
    return sum;
}

image convolve_image(image im, image filter, int preserve)
{
    assert(im.c == filter.c || filter.c == 1);
    image convolved;
    if (preserve == 1) {
        convolved = make_image(im.w, im.h, im.c);
        for (int cha = 0; cha < im.c; cha++) {
            for (int row = 0; row < im.h; row++) {
                for (int col = 0; col < im.w; col++) {
                    float sum = convolve_pixel(im, filter, col, row, cha);
                    set_pixel(convolved, col, row, cha, sum);
                }
            }
        }
    } else {
        convolved = make_image(im.w, im.h, 1);
        for (int row = 0; row < im.h; row++) {
            for (int col = 0; col < im.w; col++) {
                float sum = 0.0;
                for (int cha = 0; cha < im.c; cha++) {
                    sum += convolve_pixel(im, filter, col, row, cha);
                }
                set_pixel(convolved, col, row, 0, sum);
            }
        }
    }
    return convolved;
}

image make_highpass_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, 0.0);
    set_pixel(filter, 1, 0, 0, -1.0);
    set_pixel(filter, 2, 0, 0, 0.0);
    set_pixel(filter, 0, 1, 0, -1.0);
    set_pixel(filter, 1, 1, 0, 4.0);
    set_pixel(filter, 2, 1, 0, -1.0);
    set_pixel(filter, 0, 2, 0, 0.0);
    set_pixel(filter, 1, 2, 0, -1.0);
    set_pixel(filter, 2, 2, 0, 0.0);
    return filter;
}

image make_sharpen_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, 0.0);
    set_pixel(filter, 1, 0, 0, -1.0);
    set_pixel(filter, 2, 0, 0, 0.0);
    set_pixel(filter, 0, 1, 0, -1.0);
    set_pixel(filter, 1, 1, 0, 5.0);
    set_pixel(filter, 2, 1, 0, -1.0);
    set_pixel(filter, 0, 2, 0, 0.0);
    set_pixel(filter, 1, 2, 0, -1.0);
    set_pixel(filter, 2, 2, 0, 0.0);
    return filter;
}

image make_emboss_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -2.0);
    set_pixel(filter, 1, 0, 0, -1.0);
    set_pixel(filter, 2, 0, 0, 0.0);
    set_pixel(filter, 0, 1, 0, -1.0);
    set_pixel(filter, 1, 1, 0, 1.0);
    set_pixel(filter, 2, 1, 0, 1.0);
    set_pixel(filter, 0, 2, 0, 0.0);
    set_pixel(filter, 1, 2, 0, 1.0);
    set_pixel(filter, 2, 2, 0, 2.0);
    return filter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and
// which ones should we not? Why?
// Answer: Because color (which uses multiple channels) matters in the output of box, sharpen,
//         and emboss filters, we should use preserve when applying them. For highpass,
//         however, when we use preserve the filter finds edges for each color channel seperately.
//         This means that highpass will not be able to find edges accross colors (e.g. when
//         a smaller green box is inside of a larger red box) even when they're easily spotted
//         by humans. Similarly, because the output of highpass is just edges that
//         arent necessarily the same color as the original picture, we should not use preserve.

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: We need to clamp the results between 0.0 and 1.0 after convolving with highpass, emboss,
//         and sharpen filters becasue summing the negative values in each of their kernels means
//         we may have a negative pixel value in the resulting image -- which is not possible for
//         RGB.

float twoD_gaussian(float x, float y, float sigma) {
    return (1 / (TWOPI * sigma * sigma)) * expf(-(x * x + y * y) / (2 * sigma * sigma));
}

image make_gaussian_filter(float sigma)
{
    int filter_size = (int) ceilf(6 * sigma);
    if (filter_size % 2 == 0) { filter_size += 1; }
    image filter = make_image(filter_size, filter_size, 1);
    for (int row = -filter_size / 2; row <= filter_size / 2; row++) {
        for (int col = -filter_size / 2; col <= filter_size / 2; col++) {
            set_pixel(filter, col + filter_size / 2, row + filter_size / 2, 0,
                      twoD_gaussian(col, row, sigma));
        }
    }
    l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
    assert(a.c == b.c && a.h == b.h && a.w == b.w);
    image sum = make_image(a.w, a.h, a.c);
    for (int i = 0; i < a.c; i++) {
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                float a_pix = get_pixel(a, k, j, i);
                float b_pix = get_pixel(b, k, j, i);
                set_pixel(sum, k, j, i, a_pix + b_pix);
            }
        }
    }
    return sum;
}

image sub_image(image a, image b)
{
    assert(a.c == b.c && a.h == b.h && a.w == b.w);
    image difference = make_image(a.w, a.h, a.c);
    for (int i = 0; i < a.c; i++) {
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                float a_pix = get_pixel(a, k, j, i);
                float b_pix = get_pixel(b, k, j, i);
                set_pixel(difference, k, j, i, a_pix - b_pix);
            }
        }
    }
    return difference;
}

image make_gx_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -1.0);
    set_pixel(filter, 1, 0, 0, 0.0);
    set_pixel(filter, 2, 0, 0, 1.0);
    set_pixel(filter, 0, 1, 0, -2.0);
    set_pixel(filter, 1, 1, 0, 0.0);
    set_pixel(filter, 2, 1, 0, 2.0);
    set_pixel(filter, 0, 2, 0, -1.0);
    set_pixel(filter, 1, 2, 0, 0.0);
    set_pixel(filter, 2, 2, 0, 1.0);
    return filter;
}

image make_gy_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -1.0);
    set_pixel(filter, 1, 0, 0, -2.0);
    set_pixel(filter, 2, 0, 0, -1.0);
    set_pixel(filter, 0, 1, 0, 0.0);
    set_pixel(filter, 1, 1, 0, 0.0);
    set_pixel(filter, 2, 1, 0, 0.0);
    set_pixel(filter, 0, 2, 0, 1.0);
    set_pixel(filter, 1, 2, 0, 2.0);
    set_pixel(filter, 2, 2, 0, 1.0);
    return filter;
}

void feature_normalize(image im)
{
    float min = __FLT_MAX__;
    float max = __FLT_MIN__;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                float cur_pix = get_pixel(im, k, j, i);
                min = fminf(min, cur_pix);
                max = fmaxf(max, cur_pix);
            }
        }
    }
    float range = max - min;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                float new_pix;
                if (range <= 0) {
                    new_pix = 0;
                } else {
                    new_pix = (get_pixel(im, k, j, i) - min) / range;
                }
                set_pixel(im, k, j, i, new_pix);
            }
        }
    }
}

image *sobel_image(image im)
{
    image magnitude = make_image(im.w, im.h, 1);
    image direction = make_image(im.w, im.h, 1);
    image gx_filter = make_gx_filter();
    image gy_filter = make_gy_filter();
    image gx = convolve_image(im, gx_filter, 0);
    image gy = convolve_image(im, gy_filter, 0);
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                float gx_pix = get_pixel(gx, k, j, i);
                float gy_pix = get_pixel(gy, k, j, i);
                set_pixel(magnitude, k, j, i, sqrtf(gx_pix * gx_pix + gy_pix * gy_pix));
                set_pixel(direction, k, j, i, atan2f(gy_pix, gx_pix));
            }
        }
    }
    image *results = calloc(2, sizeof(image));
    results[0] = magnitude;
    results[1] = direction;
    return results;
}

image colorize_sobel(image im)
{
    image gaussian_filter = make_gaussian_filter(2.0);
    image smoothed = convolve_image(im, gaussian_filter, 1);
    image colorized = make_image(im.w, im.h, im.c);
    image *sobel = sobel_image(smoothed);
    image magnitude = sobel[0];
    image direction = sobel[1];
    for (int i = 0; i < im.h; i++) {
        for (int j = 0; j < im.w; j++) {
            float cur_mag = get_pixel(magnitude, j, i, 0);
            float cur_dir = get_pixel(direction, j, i, 0);
            set_pixel(colorized, j, i, 0, cur_dir);
            set_pixel(colorized, j, i, 1, 1.0);
            set_pixel(colorized, j, i, 2, cur_mag);
        }
    }
    feature_normalize(colorized);
    hsv_to_rgb(colorized);
    return colorized;
}
