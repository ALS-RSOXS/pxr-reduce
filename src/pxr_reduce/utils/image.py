
import numpy as np
from scipy.ndimage import median_filter



# Replace pixels above a threshold with the average defined by a box of SIZE x SIZE around the pixel
# -- From Jan Ilavsky's IGOR implementation.
def dezinger_image(image, med_result=None, threshold=10, size=3):
    """
    Function to remove potential hot pixels during data collection. 
    Replaces pixels that have intensity greater than threshold above its nearest neighbors

    Parameters
    -----------
    image : np.array
        Image to be zinged.
    med_result: np.array
        image after median filter if it has already been run. Will speed up process
    threshold : float
        Threshold on whether to replace pixel with average
    size: int
        Number of pixels to define the nearest neighbor placement around hot pixel
        
    Returns
    --------
    zinged_image : np.array
        New image after being zinged
    """
    if med_result is None:
        med_result = ndimage.median_filter(image, size=size)  # Apply Median Filter to image if needed
    # Calculate Ratio of each pixel to compared to a threshold
    diff_image = image / np.abs(med_result)  
    # Repopulate image by removing pixels that exceed the threshold 
    zinged_image = image*np.greater(threshold, diff_image).astype(int) + med_result*np.greater(diff_image, threshold) 
    return zinged_image # Return dezingered image