import cv2
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dark_to_light1 = "()1\{\}[]?$@B\%8&WMZO0QLCJUYX#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "
dark_to_light2 = " .\':;o*O#@"[::-1]

edges = "|/—\\"
deg_increment = 180 / len(edges)

def int_to_ascii(i):
  return dark_to_light2[i]

def resize_image_to_scale(img, scale):
  width = int(img.shape[1] * scale / 100)
  height = int(img.shape[0] * scale / 25)
  dim = (height, width)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

range_mapping = interp1d([0, 255], [0, len(dark_to_light2) - 1])
ascii_mapping = np.vectorize(int_to_ascii)

image = cv2.imread('imgs/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = resize_image_to_scale(image, 20)

mapped_image = range_mapping(image)
mapped_image = np.array(mapped_image, dtype=int)

gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

magnitude = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

print(np.max(magnitude))
print(np.min(orientation), np.max(orientation))

plt.imshow(magnitude)
plt.savefig('imgs/sobel_dog_mag.jpg')
#plt.imshow(orientation)
#plt.savefig('imgs/sobel_dog_deg.jpg')

def map_edges_to_ascii(magnitude, orientation):
  if (magnitude > 2500):
    edge_num = round(orientation / deg_increment) % len(edges)
    return edges[edge_num]
  else:
    return " "

edge_mapping = np.vectorize(map_edges_to_ascii)
edge_ascii = edge_mapping(magnitude, orientation)
ascii_image = ascii_mapping(mapped_image)

# Override ascii in the original mapping with non-space characters in the edge image
edge_idxs = np.where(edge_ascii != " ")
ascii_image[edge_idxs] = edge_ascii[edge_idxs]

with open("imgs/combined.txt", "w") as file:
  for row in ascii_image:
    file.write(''.join(row.tolist())+ "\n")
