import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import img_to_ascii as asc 


dark_to_light1 = "?$@B\%8&#*oahkbdpqwmzcvunxrjft+~<>i!lI:. "[::-1]
dark_to_light2 = " .\':;o*O#"
dark_to_light3 = "@#B&$\%?*o+~;:\"\'`. "[::-1]

light_edges = "|/-\\"

range_mapping = asc.ascii_range_mapping(dark_to_light3)                    
shading_mapping = asc.ascii_shading_mapping(dark_to_light3)
edge_mapping = asc.ascii_edge_mapping(light_edges)


def main():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    _, frame = cap.read()
    frame = asc.resize_image_to_scale(frame, 0.15*2.1, 0.15, False)
    x,y,z = frame.shape
    os.system("mode {},{}".format(y,x))

    while True:
        _, frame = cap.read()
        temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("grayscale feed", temp)
        ascii_frame = asc.img_to_ascii(frame, 0.15, 2.1,range_mapping,shading_mapping,edge_mapping)
        print(''.join(list(ascii_frame.flatten()))+'\r', end = '' )

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
    main()
