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
  ascii_mapping = np.vectorize(int_to_ascii)
  return ascii_mapping

def ascii_edge_mapping(edges, mag_threshold=2500):
  num_edges = len(edges)
  deg_increment = 180 / num_edges

  def map_edges_to_ascii(magnitude, orientation):
    if (magnitude > mag_threshold):
      edge_num = round(orientation / deg_increment) % num_edges
      return edges[edge_num]
    else:
      return " "

  return np.vectorize(map_edges_to_ascii)

# Overwrite ascii in the original mapping with non-space characters in the edge image
def combine_shading_and_edges(ascii_shading, ascii_edges):
  edge_idxs = np.where(ascii_edges != " ")
  ascii_shading[edge_idxs] = ascii_edges[edge_idxs]
  return ascii_shading

def img_to_ascii(img, scale, char_h_to_w_ratio, range_mapping, shading_mapping, edge_mapping, blur = False):
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resized_image = resize_image_to_scale(image, char_h_to_w_ratio * scale, scale, blur)
  magnitude, orientation = img_sobel(resized_image)
  mapped_image = np.array(range_mapping(resized_image), dtype=int)
  ascii_shading = shading_mapping(mapped_image)
  ascii_edges = edge_mapping(magnitude, orientation)
  ascii_image = combine_shading_and_edges(ascii_shading, ascii_edges)
  return ascii_image                                    

def write_ascii_to_file(ascii_image, outpath):
  with open(outpath, "w") as ascii_file:
    for row in ascii_image:
      ascii_file.write(''.join(row.tolist()) + "\n")

##################################### Example #####################################

# VS code characters are roughly 40:17 height to width ratio
char_h_to_w_ratio = 40.0 / 17.0

dark_to_light1 = "?$@B\%8&#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "
dark_to_light2 = " .\':;o*O#@"[::-1]
dark_to_light3 = "@#B&$\%?*o+~;:\"\'`. "

edges = "|/-\\"

range_mapping = ascii_range_mapping(dark_to_light2)                    
shading_mapping = ascii_shading_mapping(dark_to_light2)
edge_mapping = ascii_edge_mapping(edges, mag_threshold=7000)

def main():
  img_path = "imgs/plant.jpg"
  outpath = "ascii_imgs/plant.txt"
  image = cv2.imread(img_path)
  ascii_image = img_to_ascii(image, 0.1, char_h_to_w_ratio, range_mapping,
                            shading_mapping, edge_mapping, blur = True)                         
  write_ascii_to_file(ascii_image, outpath)


if __name__ == '__main__':
  main()