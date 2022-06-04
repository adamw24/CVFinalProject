import cv2
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dark_to_light1 = "()1\{\}[]?$@B\%8&WMZO0QLCJUYX#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "
dark_to_light2 = " .\':;o*O#@"[::-1]

def int_to_ascii(i):
  return dark_to_light2[i]

def resize_image_to_scale(img, scale):
  width = int(img.shape[1] * scale / 100)
  height = int(img.shape[0] * scale / 50)
  dim = (height, width)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

range_mapping = interp1d([0, 255], [0, len(dark_to_light2) - 1])
ascii_mapping = np.vectorize(int_to_ascii)

image = cv2.imread('imgs/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = resize_image_to_scale(image, 20)

mapped_image = range_mapping(image)
mapped_image = np.array(mapped_image, dtype=int)
ascii_image = ascii_mapping(mapped_image)

# print(mapped_image.shape)
# print(ascii_image)

plt.imshow(mapped_image, cmap='gray')
plt.colorbar()
plt.savefig('imgs/dog_grayscale_compressed.jpg')

with open("imgs/ascii_dog.txt", "w") as file:
    for row in ascii_image:
      file.write(' '.join(row.tolist())+ '\n')


#np.savetxt("mapped.txt", ascii_image)