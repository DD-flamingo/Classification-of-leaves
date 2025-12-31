# Classification-of-leaves
#“Creating a webpage to identify a leaf from an image scanned or uploaded by the user.”
# 1. Import libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
