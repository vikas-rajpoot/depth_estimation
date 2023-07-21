import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
from utils_test import DepthDataset
from utils_test import ToTensor
from torchvision import transforms


loc_img="./test_img_1" 


def visualze_test_dataset():
    """
        Visualize the test dataset. 
    """
    # Make the object of the Class DepthDataset which is used to load the data at the folder loc_img. 
    depth_dataset = DepthDataset(root_dir=loc_img)

    # Plot some of the images from the dataset. 
    fig = plt.figure()

    print(len(depth_dataset)) 

    for i in range(len(depth_dataset)):
        # the image of the index i. 
        sample = depth_dataset[i]
        # print the size of the image. 
        print(i, sample['image'].size)
        # show the image. 
        plt.imshow(sample['image'])
        plt.figure()

        if i == 1:
            plt.show() 
            break 


def test_model(file):
    """
    This function load the model save at the file and generate the denseDepth map of the test images. 

    Args:
        file (str): path to the model.
    """
    
    depth_dataset = DepthDataset(root_dir=loc_img,transform=transforms.Compose([ToTensor()]))
    batch_size=1 
    # Make the data loader of the testing images. 
    train_loader=torch.utils.data.DataLoader(depth_dataset, batch_size) 
    dataiter = iter(train_loader) 
    # images = dataiter.next() 
    images = next(dataiter) 


    # importing the mobilenetv2 model 
    from mobile_model import Model 

    # make model on cuda.
    model = Model().cuda() 
    model = nn.DataParallel(model) 
    # load the trained model 
    model.load_state_dict(torch.load(file), strict=False) 
    # model.eval()   



    #Upscaling image and saving the image
    # os.mkdir('./results') 

    for i,sample_batched1  in enumerate (train_loader): 
        
        image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
        
        print(image1.shape) 
        # Predicting the DenseDepth of the image. 
        outtt=model(image1 ) 
        
        x=outtt.detach().cpu().numpy() 
        img=x.reshape(240,320) 
        scale_percent = 200 
        # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        
        dim = (width, height) 
        
        # resize image 
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        # saving the DenseDepth image of the original image
        plt.imsave('./results/%d_depth.jpg' %i, resized, cmap='inferno') 
        
        # saving th image. 
        s_img=sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0)
        plt.imsave('./results/%d_image.jpg' %i, s_img)   


if __name__=='__main__':
    # un-comment the felloing line to visualize the dataset. 
    # visualze_test_dataset() 
    
    # un-comment the to test the trained model.
    file = './models/19.pth'
    test_model(file)  
    
    
    

