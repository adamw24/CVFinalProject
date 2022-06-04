import cv2
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# VS code characters are roughly 40:17 height to width ratio
char_h_to_w_ratio = 40.0 / 17.0

dark_to_light1 = "?$@B\%8&#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "[::-1]
dark_to_light2 = " .\':;o*O#@"
dark_to_light3 = "@#B&$\%?*o+~;:\"\'`. "[::-1]

dark_edges = "|/—\\"
light_edges = "I/=\\"
deg_increment = 180 / len(dark_edges)

chars_used = dark_to_light2
img_name = "apple.jpg"

def int_to_ascii(i):
  return chars_used[i]

def resize_image_to_scale(img, w_scale, h_scale):
  width = int(img.shape[1] * w_scale)
  height = int(img.shape[0] * h_scale)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

range_mapping = interp1d([0, 255], [0, len(chars_used) - 1])
ascii_mapping = np.vectorize(int_to_ascii)

image = cv2.imread("imgs/" + img_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = resize_image_to_scale(image, char_h_to_w_ratio * 0.1, 0.1)

mapped_image = range_mapping(image)
mapped_image = np.array(mapped_image, dtype=int)

gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

magnitude = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

print(np.max(magnitude))
print(np.min(orientation), np.max(orientation))

plt.imshow(magnitude)
plt.savefig('imgs/sobel_' + img_name)
#plt.imshow(orientation)
#plt.savefig('imgs/sobel_dog_deg.jpg')

def map_edges_to_ascii(image, magnitude, orientation):
  if (magnitude > 2500):
    edge_num = round(orientation / deg_increment) % len(light_edges)
    if image > 15:
      return light_edges[edge_num]
    return dark_edges[edge_num]
  else:
    return " "

edge_mapping = np.vectorize(map_edges_to_ascii)
edge_ascii = edge_mapping(image, magnitude, orientation)
ascii_image = ascii_mapping(mapped_image)

# Override ascii in the original mapping with non-space characters in the edge image
edge_idxs = np.where(edge_ascii != " ")
ascii_image[edge_idxs] = edge_ascii[edge_idxs]

with open("ascii_imgs/combined_" + img_name.split(".")[0] + ".txt", "w") as file:
  for row in ascii_image:
    file.write(''.join(row.tolist())+ "\n")
