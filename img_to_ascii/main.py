import cv2
import numpy as np
import matplotlib.pyplot as plt
from uwimg import *

image = cv2.imread('dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

