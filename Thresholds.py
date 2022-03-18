from pythreshold.utils import test_thresholds
from scipy.misc import ascent

# Testing all the included thresholding algorithms
test_thresholds()

# Testing all the included thresholding algorithms using a custom image
img = ascent()
test_thresholds(img)
