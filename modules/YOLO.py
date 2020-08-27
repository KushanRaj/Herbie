import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
from modules.CSPDarknet53 import Darknet




class SPP(nn.Module):

    def __init__(self):
        super(SPP,self).__init__()
        self.maxpool = nn.MaxPool2d(5,stride=1,padding=4)   
        self.maxpool = nn.MaxPool2d(9,stride=1,padding=8)
        self.maxpool = nn.MaxPool2d(13,stride=1,padding=12)
        
    def forward(self,x):
        outputs = []
        m1 = self.maxpool1(x)
        outputs.append(m1)
        outputs.append(x)
        m2 = self.maxpool2(x)
        outputs.append(m2)
        outputs.append(x)
        m3 = self.maxpool3(x)
        outputs.append(m3)
        outputs.append(tf.cat((m3,m2,m1,x),1))
        return outputs

class upsample(nn.Module):
    def __init__(self,stride):
        super(upsample,self).__init__()
        self.stride = stride
    def forward(self,x):
        
        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1
        ).expand(
                x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride
                ).contiguous().view(
                                    x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride
                                    )
        return x


        
class Yolo(nn.Module):

    def __init__(self):
        super(Yolo,self).__init__()

        self.darknet = Darknet()
        self.conv = nn.Conv2d(512,30, 1,stride=1)
        self.anchor = anchor
                                
        
        self.SPP = SPP()

    def loop_conv(self,l,filters,k,p,x):
        output = []
        for i in range(l):
            x = nn.Sequential(
                                    nn.Conv2d(filters[i],filters[i+1], kernel_size=k[i],padding=p[i],stride=1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(filters[i+1])
                                    )(x)
            output.append(x)
        return output

    def upsample(self,id,stride,infilter,filter,x):
        outputs = []
        
        outputs.append(upsample(stride)(x))
        outputs.append(self.loop_conv(1,[infilter,filter],[1],[0],self.darknet[id]))
        outputs.append(torch.cat((outputs[0],outputs[1]),1))
        return outputs





    def forward(self,x,boxes):
        x = self.darknet(x)
        outputs = darknet[:]
        outputs += self.loop_conv(3,[1024,512,1024,512],[1,3,1],[0,1,0],x)
        
        outputs += self.SPP(outputs[-1])
        
        outputs += self.loop_conv(4,[512*4,512,1024,512,256],[1,3,1,1],[0,1,0,0],outputs[-1])
        outputs += self.upsample(85,2,512,256,outputs[-1])
        outputs += self.loop_conv(6,[512,256,512,256,512,256,128],[1,3,1,3,1,1],[0,1,0,1,0,0],outputs[-1])
        outputs += self.upsample(54,2,256,128,outputs[-1])
        outputs += self.loop_conv(5,[256,128,256,128,256,128],[1,3,1,3,1],[0,1,0,1,0],outputs[-1])
        outputs += self.loop_conv(1,[128,512],[3],[1],outputs[-1])
        outputs.append(self.conv(outputs[-1]))
        loss = anchor()
        return outputs[-1]

#to be completed                                     
class anchor(nn.Module):

    def __init__(self,anchors,boxes,img_size):

        super(anchor,self).__init__()

        self.num = len(anchors)
        self.anchors = anchors
        self.imsize = img_size
        
    def forward(self,x,target):

        B,C,G,G = x.size()
        tB,tN,tC = target.size()
        C /= self.num
        y = x.view(B,self.num,C,G,G).permute(0,1,3,4,2).contiguous()

        off_x = torch.sigmoid(y[..., 0])
        off_y = torch.sigmoid(y[..., 1])
        scale_w = torch.exp(y[..., 2])  
        scale_h = torch.exp(y[..., 3])  
        anchor_yaw = y[..., 4]  
        anchor_conf = torch.sigmoid(y[..., 5])  
        anchor_cls = torch.argmax(y[..., 6:],-1)
       

        grid_x = torch.tensor(range(0,G)).repeat((G,1))
        grid_y = grid_x.t()
        stride = self.imsize/G
        anchor_x = torch.floor((off_x + gird_x)*stride)
        anchor_y = torch.floor((off_y + grid_y)*stride)
        anchor_w = scale_w*self.anchors[:,0]
        anchor_h = scale_w*self.anchors[:,1]

        pred = torch.cat((anchor_x,anchor_y,anchor_w,anchor_h,anchor_yaw,anchor_conf,anchor_cls),1
                        ).view(B,3,self.num,G,G).permute(0,2,3,4,1).view(B,self.num*G*G,3).contiguous().to('cuda')

        if targets:
            assert targets.size(0) == pred.size(0)
            for i,val in enumerate(targets):
                
                mask = torch.where(pred[i,:,-1]>self.thresh)
                output = pred[[i for j in mask],mask[0]]

                #temp = torch.zeros()





        

        
        





        
