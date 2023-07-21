import pandas as pd
import matplotlib.pyplot as plt 
from torch.utils.data import  DataLoader
from torchvision import transforms
from sklearn.utils import shuffle
import cv2
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from utils_train import DepthDataset 
from utils_train import Augmentation 
from utils_train import ToTensor  

#loading the mobilNetDepth model
from mobile_model import Model 

import kornia
import matplotlib
import matplotlib.cm

import torch
import time
import datetime
import torch.nn as nn
torch.cuda.empty_cache()  

traincsv=pd.read_csv('./data/nyu2_train.csv')
traincsv = traincsv.values.tolist()  
traincsv = shuffle(traincsv, random_state=2) 


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    """this function is used to compute the Structural Similarity Index(SSIM). 

    Args:
        img1 (numpy array): image. 
        img2 (numpy array): image. 
        val_range (int): value range. 
        window_size (int, optional): window_size. Defaults to 11.
        window (_type_, optional): size average. Defaults to None.
        size_average (bool, optional): Defaults to True.
        full (bool, optional): Defaults to False.

    Returns:
        int: ssim loss/index. 
    """
    ssim_loss = kornia.losses.SSIMLoss(window_size=11,max_val=val_range,reduction='none')
    return ssim_loss(img1, img2)  


def visualization_of_dataset():
    """
        Visualize the dataset. 
    """ 
    # display a sample set of image and depth image  
    depth_dataset = DepthDataset(traincsv=traincsv,root_dir='./')
    fig = plt.figure() 
    len(depth_dataset) 
    for i in range(len(depth_dataset)):
        sample = depth_dataset[i] 
        print(i, sample['image'].size, sample['depth'].size)
        plt.imshow(sample['image']) 
        plt.figure() 
        plt.imshow(sample['depth'])
        plt.figure() 
        if i == 1:
            plt.show() 
            break 
        
        
def DepthNorm(depth, maxDepth=1000.0):
    """calculate the normalized depth. 

    Args:
        depth (Any): Depth. 
        maxDepth (float, optional): maximum depth.. Defaults to 1000.0.

    Returns:
        int: normailze depth.
    """
    return maxDepth / depth


class AverageMeter(object):
    """AverageMeter
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:] 

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1)) 


def train_model():
    """
        This Method is used to train the model. 
    """
    # Print the summary of the CUDA GPU. 
    print(torch.cuda.memory_summary(device=None, abbreviated=False))  
    
    # create the MobileNet Model.
    model = Model().cuda()

    # If we have more the one GPU the train the model on multiple GPUs using DataParallerl.
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)


    print('Model created....') 
    
    ## number of epochs of training.
    epochs=16
    # learning rate. 
    lr=0.0001
    # batch size.
    batch_size=32    # (64 initially)

    # load the training dataset. 
    depth_dataset = DepthDataset(traincsv=traincsv, root_dir='./',
                    transform=transforms.Compose([Augmentation(0.5),ToTensor()])) 


    print(type(depth_dataset))

    # load the dataset set. 
    train_loader=DataLoader(depth_dataset, batch_size, shuffle=True) 
    l1_criterion = nn.L1Loss()

    optimizer = torch.optim.Adam( model.parameters(), lr )

    # Start training...
    for epoch in range(epochs):
        # save the model at each epochs 
        path='./models/'+str(epoch)+'.pth'        
        torch.save(model.state_dict(), path) 
        batch_time = AverageMeter() 
        losses = AverageMeter() 
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time() 

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            #Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output = model(image)

            # Compute the loss 
            l_depth = l1_criterion(output, depth_n) 
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            # backward propogation. 
            loss.backward()
            optimizer.step()

            # Measure elapsed time 
            batch_time.update(time.time() - end) 
            end = time.time() 
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                
        path='./models/'+str(epoch)+'.pth'        
        torch.save(model.state_dict(), path)   

# Evaluations

def evaluate_mode(itr): 
    """Evaluate the trained model.

    Args:
        itr (int): integer to save the model.
    """

    model = Model().cuda() 
    model = nn.DataParallel(model)  

    #load the model if needed
    model.load_state_dict(torch.load('./models/16.pth'), strict=False)  

    model.eval() 
    batch_size=1 

    depth_dataset = DepthDataset(traincsv=traincsv, root_dir='./',
                    transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
    train_loader=DataLoader(depth_dataset, batch_size, shuffle=True)

    outtt = None 
    
    for sample_batched1  in (train_loader):
        image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
        
        outtt=model(image1 ) 
        break 
    
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
    plt.imsave('./results/%d_depth.jpg' %itr, resized, cmap='inferno') 
    # plt.show(resized) 
    
    # saving th image. 
    s_img=sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0)
    plt.imsave('./results/%d_image.jpg' %itr, s_img)   
    # plt.show(s_img) 





if __name__=='__main__':
    # un-comment to visualize the dataset. 
    # visualization_of_dataset() 
    
    # un-comment to train the mobilenetv2 model.
    # train_model() 
    
    # un-comment to evaluate the model. 
    # evaluate_mode(0)
    pass 






