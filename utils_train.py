"""
Created on Sun Dec 29 23:17:26 2019

@author: alin
"""

# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image
import random
import numpy as np
import torch

#Depth Datasetclass

def _is_pil_image(img):
    """check the type of the input image. 

    Args:
        img (Any): Any input. 

    Returns:
        bool: if input img is type of PIL image the returns true other wise false. 
    """
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    """check the type of the input image. 

    Args:
        img (Any): Any input. 

    Returns:
        bool: if input img is type of numpy image the returns true other wise false. 
    """
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class DepthDataset(Dataset):
    """Dataset 

    Args:
        Dataset (Class): Dataset class of the pytorch. 
    """
    os = __import__('os')  
    
    def __init__(self, traincsv, root_dir, transform=None):
        """intitialize the dataset. 

        Args:
            root_dir (str): takes the root diroctory where the testing images avilable. 
            transform (Class, optional): pass the transform class which is used to transform the input images. Defaults to None.
        """
        
        self.traincsv = traincsv
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """return the length of dataset. 

        Returns:
            int: returns the length of the dataset i.e the number of images in the dataset. 
        """
        return len(self.traincsv)

    def __getitem__(self, idx):
        """function returns the image at the index idx

        Args:
            idx (int): index 

        Returns:
            _type_: image at the idx.d
        """
        
        sample = self.traincsv[idx] 
        img_name = os.path.join(self.root_dir,sample[0]) 
        image = (Image.open(img_name)) 
        depth_name = os.path.join(self.root_dir,sample[1]) 
        depth =(Image.open(depth_name)) 
        ## depth = depth[..., np.newaxis]        
        sample1={'image': image, 'depth': depth} 

        if self.transform:  sample1 = self.transform({'image': image, 'depth': depth}) 
        
        return sample1

    
class Augmentation(object):
    """data augmaentation is important while training the model 
    """
    def __init__(self, probability):
        """initialize the probability of augmentation. 

        Args:
            probability (int): probabiliy.
        """
        from itertools import permutations
        self.probability = probability
        #generate some output like this [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        self.indices = list(permutations(range(3), 3))
        #followed by randomly picking one channel in the list above
    
    def __call__(self, sample):
        """overwrithe the call funtion

        Args:
            sample (dictornary): contents the training images which need to augment. 

        Raises:
            TypeError: if image is not of type PIL.
            TypeError: if image ie not of type Numpy. 

        Returns:
            dictornary : return the augmented images. 
        """
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))
        
        # flipping the image
        if random.random() < 0.5:
            #random number generated is less than 0.5 then flip image and depth
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        
        # rearranging the channels    
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])    

        return {'image': image, 'depth': depth}
    
    
class ToTensor(object):
    """Class used to convert the image to tensor. 

    """
    def __init__(self,is_test=False):
        
        """Initialize the ToTensor class. 

        Args:
            is_test (bool, optional): take is_test bool as the input which tells wheather we are using this at time of testing or training
            .Defaults to False. 
        """
        self.is_test = is_test

    def __call__(self, sample):
        
        """Overwrithing the call function which takes the sample as the input which is the dictonary contains 
            images and denseDepth of the image. 

        Args:
            sample (dictonary): take dictonary as the input which contents the image and densdepth. 

        Returns:
            dictornary: returns the image.
        """
        
        image, depth = sample['image'], sample['depth']
        

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        """
        convert the image to the tensor. 

        Args:
            pic (np.ndarray or PIL): take the image as the input. 

        Raises:
            TypeError: if image is not type of the numpy or PIL then this method raise the type error. 

        Returns:
            tensor: convert the numpy or PIL image to the tensor. 
        """
        
        pic = np.array(pic) 
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
                raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic))) 
             
        if isinstance(pic, np.ndarray):
            if pic.ndim==2:
                pic=pic[..., np.newaxis]
                
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255) 



