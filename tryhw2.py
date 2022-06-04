from uwimg import *
im = load_image("data/dog.jpg")
f = make_box_filter(7)
blur = convolve_image(im, f, 1)
save_image(blur, "dog-box7")

f = make_sharpen_filter();
blur = convolve_image(im, f, 1);
clamp_image(blur);
save_image(blur, "dog-sharpen")

f = make_highpass_filter();
blur = convolve_image(im, f, 1);
clamp_image(blur);
save_image(blur, "dog-highpass")

f = make_emboss_filter();
blur = convolve_image(im, f, 1);
clamp_image(blur);
save_image(blur, "dog-emboss")

f = make_gaussian_filter(2);
blur = convolve_image(im, f, 1);
clamp_image(blur);
save_image(blur, "dog-gaussian2")

res = sobel_image(im)
mag = res[0]
direction = res[1]
feature_normalize(mag)
feature_normalize(direction)
save_image(mag, "dog-magnitude")
save_image(direction, "dog-direction")

colorized = colorize_sobel(im)
save_image(colorized, "dog-color-sobel")