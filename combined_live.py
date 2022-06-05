import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import img_to_ascii as asc

# ASCII characters to use
dark_to_light1 = "?$@B\%8&#*oahkbdpqwmzcvunxrjft+~<>i!lI:. "[::-1]
dark_to_light2 = " .\':;o*O#"
dark_to_light3 = "@#$\%?*o+~;:\"\'`. "[::-1]
dark_to_light4 = "@ #.+\'\%:?o*~;$"
light_edges = "|/-\\"


scale = 0.15
# For terminal height to width ratio
h_to_w_ratio = 2.1

def main():
  range_mapping = asc.ascii_range_mapping(dark_to_light4)
  shading_mapping = asc.ascii_shading_mapping(dark_to_light4)
  edge_mapping = asc.ascii_edge_mapping(light_edges)

  cap = cv2.VideoCapture(0)

  # Check if the webcam is opened correctly
  if not cap.isOpened():
      raise IOError("Cannot open webcam")

  # Take a sample frame and adjust Terminal window accordingly.
  _, frame = cap.read()
  frame = asc.resize_image_to_scale(frame, scale*h_to_w_ratio, scale, False)
  x,y,z = frame.shape
  os.system("mode {},{}".format(y,x))

  while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Displays the grayscale video next to the ASCII art
    cv2.imshow("grayscale feed", grayframe)
    ascii_frame = asc.img_to_ascii(frame, scale, h_to_w_ratio, range_mapping,shading_mapping,edge_mapping)
    print(''.join(list(ascii_frame.flatten()))+'\r', end = '' )

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ ==  "__main__":
    main()
