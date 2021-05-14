# transformations
import random
import numpy as np
from numpy import asarray
import skimage
import torch
import math
import scipy.ndimage


class Resize(object):

  def __init__(self, width, height):
      self.width = width
      self.height = height
                          
  def __call__(self, image):
      image = asarray(image)
      width = self.width
      height = self.height

      image = skimage.transform.resize(image, (width, height), preserve_range=True, anti_aliasing=True )

      return image


class HorizontalFlip(object):
  """ Flip images """
  def __call__(self, sample):
    image, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
    if random.random() <= 0.5:
      image = np.flip(image, axis = 1).copy()
    return {'image':image, 'idx': idx, 'patient':patient, 'label':label}

class Rotation(object):  
  """ Random rotate images """
  def __call__(self,sample):
    image, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
    angle = random.randint(-10, 10)
    image = scipy.ndimage.rotate(image, angle, reshape = False, order = 3)# has to be order 0
    return {'image':image, 'idx': idx, 'patient':patient, 'label':label}


class Shifting(object): 
  """ Random shift images """
  def __call__(self,sample):
    image, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
    max_val = 10
    x_shift, y_shift = random.randint(-max_val,max_val), random.randint(-max_val,max_val)          
    image = scipy.ndimage.shift(image, (y_shift, x_shift, 0), order = 3)
    return {'image':image, 'idx': idx, 'patient':patient, 'label':label}

class Noise(object):  # helps prevent overfitting
  """ Random noise images """
  def __call__(self,sample):
    image, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
    random_number = random.random()
    if random_number <= 0.5:
        image = skimage.util.random_noise(image)
      # would this work given that mask is unchanged??
    return {'image':image, 'idx': idx, 'patient':patient, 'label':label}

class Normalise(object):  # helps prevent overfitting
  """ Random noise images """
  def __call__(self,sample):
    image, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
    mini = image.min()
    image = image + mini
    maxi = image.max()
    image /= maxi
    return {'image':image, 'idx': idx, 'patient':patient, 'label':label}


"""
class GaussBlur(object):  # helps prevent overfitting
  # Gaussian blur images 
  def __call__(self,sample):
    image, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
    if random.random() <= 0.5:
      image = cv2.GaussianBlur(image,(3,3),0)
      # do i want to blur the mask??
    return {'image':image, 'idx': idx, 'patient':patient, 'label':label}
"""
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0, 0, 0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        
        img, idx, patient, label = sample['image'], sample['idx'], sample['patient'], sample['label']
        
        
        
        if random.uniform(0, 1) > self.probability:
            return {'image':img, 'idx': idx, 'patient':patient, 'label':label}

        for attempt in range(100):
            #print('height', img.shape[0])
            #print('width', img.shape[1])
            area = img.shape[0] * img.shape[1]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)
                if img.shape[2] == 2:
                    img[y1:y1+w, x1:x1+h,  0] = self.mean[0]
                    img[y1:y1+w, x1:x1+h,  1] = self.mean[1]
                    #img[y1:y1+w, x1:x1+h,  2] = self.mean[2]
                else:
                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                return {'image':img, 'idx': idx, 'patient':patient, 'label':label}
            
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, idx, patient, label = sample['image'],  sample['idx'], sample['patient'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image).float() # dont know why images/mask casted to float here but need to do it again later
        return {'image':image, 'idx': idx, 'patient':patient, 'label':label}

