import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


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
    def __init__(self, root_dir, transform=None):
        """intitialize the dataset. 

        Args:
            root_dir (str): takes the root diroctory where the testing images avilable. 
            transform (Class, optional): pass the transform class which is used to transform the input images. Defaults to None.
        """
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """return the length of dataset. 

        Returns:
            int: returns the length of the dataset i.e the number of images in the dataset. 
        """
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        """function returns the image at the index idx

        Args:
            idx (int): index 

        Returns:
            _type_: image at the idx.d
        """
        
        img_name = os.path.join(self.root_dir,os.listdir(self.root_dir)[idx]) 
        
        # print("image_name", type(img_name))
        image = (Image.open(img_name))

        sample1={'image': image}

        if self.transform:  sample1 = self.transform({'image': image})
        return sample1 
    
    

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
        image= sample['image']
        
        image = image.resize((640, 480))
        image = self.to_tensor(image)

        return {'image': image}

    def to_tensor(self, pic):
        """convert the image to the tensor. 

        Args:
            pic (np.ndarray or PIL): take the image as the input. 

        Raises:
            TypeError: if image is not type of the numpy or PIL then this method raise the type error. 

        Returns:
            tensor: convert the numpy or PIL image to the tensor. 
        """
        
        pic = np.array(pic)
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
                raise TypeError(  'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
                             
        if isinstance(pic, np.ndarray):
            if pic.ndim==2:
                pic=pic[..., np.newaxis]
                
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255) 