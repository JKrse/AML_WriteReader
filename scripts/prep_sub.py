# ============================================================================================================
# import packages
import numpy as np

# ============================================================================================================


def image_name(url):
    name = str(url.split('/')[-1])
    return name

g = 5
# ============================================================================================================

"""
Combine Image features and word embedding datasets 
Copy structure from cpvr
Split data: train, test, val ({"human": {..}, "mc": {..}})
"""

dd = np.load("../local_files/data/image_features.npz")
list(dd)
dd["Images"]
dd["Image_features"][0]