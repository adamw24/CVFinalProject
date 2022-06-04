import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dark_to_light1 = "()1\{\}[]?$@B\%8&WMZO0QLCJUYX#*oahkbdpqwmzcvunxrjft+~<>i!lI;:,\"^`\'. "
dark_to_light2 = " .\':;o*O#@"[::-1]
scale_factor = 11

def int_to_ascii(i):
  return dark_to_light2[i]

def resize_image_to_scale(img, scale):
  width = int(img.shape[1] * scale / 100)
  height = int(img.shape[0] * scale / 25)
  dim = (height, width)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    
def main():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    range_mapping = interp1d([0, 255], [0, len(dark_to_light2) - 1])
    ascii_mapping = np.vectorize(int_to_ascii)

    _, frame = cap.read()
    frame = resize_image_to_scale(frame, scale_factor)
    x, y, z = frame.shape
    os.system("mode {},{}".format(y, x))

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = resize_image_to_scale(frame, scale_factor)

        mapped_frame = range_mapping(frame)
        mapped_frame = np.array(mapped_frame, dtype=int)
        ascii_frame = ascii_mapping(mapped_frame)

        # cv2.imshow('Input', frame)

        print(''.join(list(ascii_frame.flatten()))+'\r')
        # with open("imgs/live_conversion.txt", "w") as live_file:
        #     live_file.write(np.array2string(ascii_frame.flatten()))

        # with open("imgs/live_conversion.txt", "w") as live_file:
        #     for row in ascii_frame:
        #         live_file.write(''.join(row)+ '\n')
        #         live_file.flush()
        #     # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     #     break
        #     print("wrote")
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
    main()
