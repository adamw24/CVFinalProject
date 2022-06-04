import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

##################################### helper functions #####################################

def resize_image_to_scale(img, w_scale, h_scale):
  width = int(img.shape[1] * w_scale)
  height = int(img.shape[0] * h_scale)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def img_sobel(image):
  gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
  gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
  magnitude = np.sqrt((gX ** 2) + (gY ** 2))
  orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
  return magnitude, orientation

##################################### img to ascii functions #####################################

def img_to_ascii_shading(resized_image, shading_chars):
  def int_to_ascii(i):
    return shading_chars[i]
  range_mapping = interp1d([0, 255], [0, len(shading_chars) - 1])
  ascii_mapping = np.vectorize(int_to_ascii)
  mapped_image = np.array(range_mapping(resized_image), dtype=int)
  ascii_shading = ascii_mapping(mapped_image)
  return ascii_shading

def img_to_ascii_edges(resized_image, light_edge_chars, dark_edge_chars,
                       mag_threshold, light_threshold):
  num_edges = len(light_edge_chars)
  deg_increment = 180 / num_edges

  def map_edges_to_ascii(image, magnitude, orientation):
    if (magnitude > mag_threshold):
      edge_num = round(orientation / deg_increment) % num_edges
      if image > light_threshold:
        return light_edge_chars[edge_num]
      return dark_edge_chars[edge_num]
    else:
      return " "

  edge_mapping = np.vectorize(map_edges_to_ascii)
  magnitude, orientation = img_sobel(resized_image)
  edge_ascii = edge_mapping(resized_image, magnitude, orientation)
  return edge_ascii

def img_to_ascii_combined(resized_image, shading_chars, light_edges, dark_edges,
                          mag_threshold=2500, light_threshold=15):
  ascii_shading = img_to_ascii_shading(resized_image, shading_chars)
  ascii_edges = img_to_ascii_edges(resized_image, light_edges, dark_edges,
                                    mag_threshold, light_threshold)
  # Overwrite ascii in the original mapping with non-space characters in the edge image
  edge_idxs = np.where(ascii_edges != " ")
  ascii_shading[edge_idxs] = ascii_edges[edge_idxs]
  return ascii_shading

def write_img_to_ascii(img_path, outpath, scale, char_h_to_w_ratio, shading_chars, light_edges,
                       dark_edges, mag_threshold=2500, light_threshold=15):
  image = cv2.imread(img_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = resize_image_to_scale(image, char_h_to_w_ratio * scale, scale)
  ascii_image = img_to_ascii_combined(resized_image, shading_chars, light_edges, dark_edges,
                                      mag_threshold, light_threshold)
  with open(outpath, "w") as ascii_file:
    for row in ascii_image:
      ascii_file.write(''.join(row.tolist())+ "\n")

##################################### Example #####################################

# VS code characters are roughly 40:17 height to width ratio
char_h_to_w_ratio = 40.0 / 17.0

dark_to_light1 = "?$@B\%8&#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "[::-1]
dark_to_light2 = " .\':;o*O#@"
dark_to_light3 = "@#B&$\%?*o+~;:\"\'`. "[::-1]

dark_edges = "|/—\\"
light_edges = "I/=\\"

img_path = "imgs/apple.jpg"
outpath = "ascii_imgs/combined_apple.txt"

write_img_to_ascii(img_path, outpath, 0.1, char_h_to_w_ratio, dark_to_light2, light_edges, dark_edges)