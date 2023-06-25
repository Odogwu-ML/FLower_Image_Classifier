from PIL import Image
import numpy as np

def process_img(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image) # opens the image
    
    if img.size[0] > img.size[1]: # if width is greater than height
        img.thumbnail((10000, 256)) # makes th height which is the shortest side 256pixels
    else: 
        img.thumbnail((256, 10000))
       
    # Crop the image at the middle
    width, height = img.size
        
    left = (width - 224)/2
    bottom = (height - 224)/2
    right = (left + 224)
    top = (bottom + 224)
        
    img = img.crop(box= (left, bottom, right, top))
        
    # Normalize the image
    img = np.array(img)/255
        
    means = [0.485, 0.456, 0.406] # mean values
    std = [0.229, 0.224, 0.225] # std values
        
    img = (img - means) / std
    
    # Move the color channel to the first dimension
    img = img.transpose(2, 0, 1)
    return img