import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dark_to_light1 = "()1\{\}[]?$@B\%8&WMZO0QLCJUYX#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "
dark_to_light2 = " .\':;o*O#@"[::-1]
edges = "|/—\\"
deg_increment = 180 / len(edges)
scale_factor = 11

def int_to_ascii(i):
  return dark_to_light2[i]

def resize_image_to_scale(img, scale):
  width = int(img.shape[1] * scale / 100)
  height = int(img.shape[0] * scale / 25)
  dim = (height, width)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def map_edges_to_ascii(magnitude, orientation):
    if (magnitude > 2500):
      edge_num = round(orientation / deg_increment) % len(edges)
      return edges[edge_num]
    else:
      return " "


def main():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    range_mapping = interp1d([0, 255], [0, len(dark_to_light2) - 1])
    ascii_mapping = np.vectorize(int_to_ascii)
    edge_mapping = np.vectorize(map_edges_to_ascii)

    _, frame = cap.read()
    frame = resize_image_to_scale(frame, scale_factor)
    x,y,z = frame.shape
    os.system("mode {},{}".format(y,x))

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = resize_image_to_scale(frame,scale_factor)

        mapped_frame = range_mapping(frame)
        mapped_frame = np.array(mapped_frame, dtype=int)
        ascii_frame = ascii_mapping(mapped_frame)

        gX = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        gY = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = np.sqrt((gX ** 2) + (gY ** 2))
        orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

        edge_ascii = edge_mapping(magnitude, orientation)

        # Override ascii in the original mapping with non-space characters in the edge image
        edge_idxs = np.where(edge_ascii != " ")
        ascii_frame[edge_idxs] = edge_ascii[edge_idxs]


        # cv2.imshow('Input', frame)
        print(''.join(list(ascii_frame.flatten()))+'\r', end = '' )
        # with open("imgs/live_conversion.txt", "w") as live_file:
        #     live_file.write(np.array2string(ascii_frame.flatten()))

        # with open("imgs/live_conversion.txt", "w") as live_file:
        #     for row in ascii_frame:
        #         live_file.write(''.join(row)+ '\n')
        #         live_file.flush()
        #     # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     #     break
        #     print("wrote")

    cap.release()
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
    main()
