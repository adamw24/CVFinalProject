import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

##################################### helper functions #####################################

def resize_image_to_scale(img, w_scale, h_scale, blur):
  if blur:
    img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0)
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

def ascii_range_mapping(shading_chars):
  range_mapping = interp1d([0, 255], [0, len(shading_chars) - 1])
  return range_mapping

def ascii_shading_mapping(shading_chars):
  def int_to_ascii(i):
    return shading_chars[i]
  return np.vectorize(int_to_ascii)

def ascii_edge_mapping(edges_chars, mag_threshold=2500):
  num_edges = len(edges_chars)
  deg_increment = 180 / num_edges

  def map_edges_to_ascii(magnitude, orientation):
    if (magnitude > mag_threshold):
      edge_num = round(orientation / deg_increment) % num_edges
      return edges_chars[edge_num]
    return " "
  return np.vectorize(map_edges_to_ascii)

def ascii_corner_mapping(corner_char, threshold=0.03):
  def map_corners_to_ascii(harris_img):
    if harris_img > threshold:
      return corner_char
    return " "
  return np.vectorize(map_corners_to_ascii)

# Overwrite ascii in the each layer with non-space characters in the next layer
# layers[0] is lowest priority, layers[-1] is highest
def combine_ascii_layers(layers):
  layered_ascii = layers[0]
  for i in range(1, len(layers)):
    cur_layer = layers[i]
    new_idxs = np.where(cur_layer != " ")
    layered_ascii[new_idxs] = cur_layer[new_idxs]
  return layered_ascii

def img_to_ascii(img, scale, char_h_to_w_ratio, range_mapping, shading_mapping, edge_mapping=None,
                 corner_mapping=None, blur=False):
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resized_image = resize_image_to_scale(image, char_h_to_w_ratio * scale, scale, blur)
  mapped_image = np.array(range_mapping(resized_image), dtype=int)
  ascii_layers = []
  ascii_shading = shading_mapping(mapped_image)
  ascii_layers.append(ascii_shading)

  if edge_mapping:
    magnitude, orientation = img_sobel(resized_image)
    ascii_edges = edge_mapping(magnitude, orientation)
    ascii_layers.append(ascii_edges)

  if corner_mapping:
    corners = cv2.cornerHarris(resized_image, blockSize=2, ksize=3, k=0.04)
    ascii_corners = corner_mapping(corners)
    ascii_layers.append(ascii_corners)

  ascii_image = combine_ascii_layers(ascii_layers)
  return ascii_image

def write_ascii_to_file(ascii_image, outpath):
  with open(outpath, "w") as ascii_file:
    for row in ascii_image:
      ascii_file.write(''.join(row.tolist()) + "\n")

##################################### Example #####################################

# VS code characters are roughly 40:17 height to width ratio
char_h_to_w_ratio = 40.0 / 17.0

dark_to_light1 = "?$@B%8&#*oahkbdpqwmzcvunxrjft~<>i!lI;:,\"^`\'. "
dark_to_light2 = " .\':;o*O#@"[::-1]
dark_to_light3 = "@#B&$%?*o~;:\"\'`. "

edges = "|/-\\"

range_mapping = ascii_range_mapping(dark_to_light2)
shading_mapping = ascii_shading_mapping(dark_to_light2)
edge_mapping = ascii_edge_mapping(edges, mag_threshold=7000)
corner_mapping = ascii_corner_mapping("+", threshold=0.005)

def main():
  img_path = "imgs/flower_frames.jpg"
  outpath = "ascii_imgs/flower_frames.txt"
  image = cv2.imread(img_path)
  ascii_image = img_to_ascii(image, 0.1, char_h_to_w_ratio, range_mapping,
                            shading_mapping, edge_mapping, corner_mapping, blur = True)
  write_ascii_to_file(ascii_image, outpath)


if __name__ == '__main__':
  main()