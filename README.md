# Video to ASCII: Displaying Live Video without a GUI
### Computer Vision (CSE455) Final Project built by Adam Wang and Kai Nylund

---
## Problem Description:
This project explores converting images and live video into ASCII. This allows for images and videos to be displayed without a GUI, for example, in Command Prompt. We wanted to make the features stand out while also making the computations fast.

![Setup](./imgs/misc/setup.png)
---
## Approach:
We implemented our ASCII conversion in several steps:

- `ascii_shading_mapping`: Maps pixels of an image directly to ascii characters from dark to light (e.g., " .\':;o*O#@") based on their grayscale value.
![Shading Image](./imgs/misc/shading.png)

- `ascii_edge_mapping`: Maps the edges in an image to a limited set of ascii characters (e.g., "|/—\") based on their direction. We found edge magnitude and directions using a sobel filter.Then, we display the ascii edges over the ascii shading to create our final image. 
![Edges Combined Image](./imgs/misc/edges.png)
- `ascii_corner_mapping`: Used Harris Corner detection to represent corners of objects as a +. We learned that it does not have a very large effect on the resulting image.
![Harris Image](./imgs/misc/Harris.png)

- `img_to_mini_hog_ascii`: Created a different style of ASCII art using HOG features without shading.
![HOG Image](./imgs/misc/HOG.png)


- `combined_live.py`: Uses `ascii_shading_mapping` and `ascii_edge_mapping` to produce a live ASCII conversion using camera input.

---
## Details and Resources Used:
We mainly used the OpenCV library to perform tasks such as resizing and video capture. For edge detection, we used the idea of Sobel filters from Homework 2, but used the built in OpenCV function instead. We used SciPy to map the max pixel values to the number of characters we were using. We used the os library in python to set the terminal size for optimal viewing of the combined live version (for Windows).

We used Joeseph Redmons dog photo as well as other images to test our conversions (can be found in `img/test_imgs`).

---

## Summary:
Since ASCII text is taller than it is wide, we scaled the width and height of the images/ video frames differently so the ASCII image would look more proportional. Converting the shading into ASCII makes the resulting art look very realistic. To make it more of something a human would have generated, we reduced the image size and compressed it so that there were less details, and were much more realistic for a human to generate.

![Resize Image](./imgs/misc/resize.png)

We found that when doing live conversions based on camera input, using too many characters to convert into the ASCII image created muddled images because small differences in  brightness would cause a different character to be used. We ended up using ~14 shading character encodings. We also noticed that doing the Harris Corner detection did not have a very large impact on the ASCII art, so we omitted it in the live version to decrease computation time and improve responsiveness.

One of the issues with live conversions is that the ASCII is text, not an image, and to display it we have to write to a file as opposed to showing the image. We used Notepad ++ and it prompted us to refesh every time for a 'updated' frame, so it was more of a camera snapshot than a live ASCII video. How we fixed this was printing the ASCII as a string to the terminal, and fixing the terminal size so that it would display correctly.

![Terminal Image](./imgs/misc/terminal.png)


---
## Results:

Simple 'Live' Conversion (Adam + Roommates + Cat):
![Simple Live Conversion Image](./imgs/result_imgs/epic_live.conversion.png)

Simple Pixel Conversion of dog:
![Simple Pixel Dog Image](./imgs/result_imgs/ascii_dog.png)

Sobel Edge Conversion of dog:
![Sobel Edge Conversion Image](./imgs/result_imgs/ascii_dog_edges.png)

Combined image of Edges + ASCII Conversion of dog:
![Live ASCII Conversion Image](./imgs/result_imgs/combined.png)

HOG Results on test images:
![HOG Conversion Image](./imgs/result_imgs/HOGResults.png)


Live Demo Video:

https://user-images.githubusercontent.com/55294835/172263666-38a6daac-11f4-4a50-8d17-40976569046e.mp4

---
## Video Presentation:

https://youtu.be/rKsnCe0fgdY

---

## Next Steps:
After resizing out input, we are doing an almost pixel by pixel conversion to ASCII characters. It would be interesting building some network or model that would be able to smooth the images, especially when capturing live video.

--- 

## References:
https://towardsdatascience.com/convert-pictures-to-ascii-art-ece89582d65b

https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/

https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html 

https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html 

https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html 
