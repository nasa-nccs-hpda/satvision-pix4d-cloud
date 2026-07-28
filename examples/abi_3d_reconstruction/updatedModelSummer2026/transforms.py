import numpy as np

class SimpleMinMaxScale(object):
    """
    A simple scaling transform that you can update once you find the min/max 
    ranges for each of the 16 bands in the new chips.
    """
    def __init__(self):
        # TODO: Replace these with the actual min/max values you find during your exploration.
        # Currently, they are set to 0 and 1, which means no scaling will happen yet.
        self.min_vals = np.array([-25.936647415161133, -20.2899112701416, -12.037643432617188, -4.522368431091309, -3.0596137046813965, -0.9609506726264954, -0.03759999945759773, 0.1447715163230896, -0.8235999941825867, -0.9560999870300293, -1.3021999597549438, -1.5393999814987183, -1.6442999839782715, 5.90310001373291, -1.7558000087738037, -5.239200115203857], dtype=np.float32)
        self.max_vals = np.array([804.0360717773438, 628.9872436523438, 373.1669921875, 140.19342041015625, 94.84803009033203, 29.789470672607422, 25.589599609375, 9.452010154724121, 45.29140090942383, 81.0928955078125, 135.2646026611328, 109.84480285644531, 185.56988525390625, 200.9023895263672, 214.30140686035156, 174.69261169433594], dtype=np.float32)


    def __call__(self, img):
        # img shape is expected to be (7, 512, 512, 16)
        
        # Avoid division by zero if max == min
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0 

        # Apply min-max scaling
        img = (img - self.min_vals) / range_vals

        # Clip values to [0, 1] just to be safe
        img = np.clip(img, 0.0, 1.0)
        
        # Replace any missing NaN data with 0.0 so the model loss doesn't become NaN
        img = np.nan_to_num(img, nan=0.0)

        return img
