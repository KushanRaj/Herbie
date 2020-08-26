import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.optim as optim



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, in_filters,mid_filters,filters):
        super(ResBlock, self).__init__()
        
        

        self.conv_1 = nn.Sequential(
                                    nn.Conv2d(in_filters,mid_filters, kernel_size=1, stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(mid_filters)
                                    )
        self.conv_2 = nn.Sequential(
                                    nn.Conv2d(mid_filters,filters, kernel_size=3, padding=1,stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(filters)
                                    )
        
        
    
        
    
    def forward(self, x):
        output = []
        conv1 = self.conv_1(x)
        output.append(conv1)
        
        conv2 = self.conv_2(conv1)
        output.append(conv2)
        

        return x + conv2, output
        
class Split(nn.Module):

    def __init__(self,pre,post,pre_arg,post_arg):

        super(Split,self).__init__()
        
        self.start = []
        self.post = post
        for i in range(pre):
            if pre<3:
                if i == 0:
                    self.start += [nn.Sequential(
                                    nn.Conv2d(pre_arg[i],pre_arg[i+1], kernel_size=3, padding=1,stride=2),
                                    Mish(),
                                    nn.BatchNorm2d(pre_arg[i+1])
                                    )
                                    ]
                if i==1:
                    self.start += [nn.Sequential(
                                            nn.Conv2d(pre_arg[i],pre_arg[i+1], kernel_size=1,stride=1),
                                            Mish(),
                                            nn.BatchNorm2d(pre_arg[i+1])
                                            )
                                            ]
            else:
                if i == 0:
                    
                    self.start += [nn.Sequential(
                                    nn.Conv2d(pre_arg[i],pre_arg[i+1], kernel_size=1,stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(pre_arg[i+1])
                                    )
                                    ]
                if i == 1:
                    self.start += [nn.Sequential(
                                    nn.Conv2d(pre_arg[i],pre_arg[i+1], kernel_size=3, padding=1,stride=2),
                                    Mish(),
                                    nn.BatchNorm2d(pre_arg[i+1])
                                    )
                                    ]
                if i==2:
                    self.start += [nn.Sequential(
                                            nn.Conv2d(pre_arg[i],pre_arg[i+1], kernel_size=1,stride=1),
                                            Mish(),
                                            nn.BatchNorm2d(pre_arg[i+1])
                                            )
                                            ]

            
        
        self.fork = [nn.Sequential(
                nn.Conv2d(post_arg[0],post_arg[1], kernel_size=1,stride=1),
                Mish(),
                nn.BatchNorm2d(post_arg[1])
                )]

        for i in range(0,post*3,3):
            self.fork += [ResBlock(post_arg[i+1],post_arg[i+2],post_arg[i+3])]
        
        
        self.fork.append(nn.Sequential(
                                    nn.Conv2d(post_arg[-1],post_arg[-1], kernel_size=1,stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(post_arg[-1])
                                    ))

    def forward(self,x):
        output = []
             
        for i in self.start:
            
            x = i(x)
            
            
            output.append(x)
        
        x = output[-2]
        output.append(x)  
        x = self.fork[0](x)
        output.append(x) 
        for i in self.fork[1:self.post+1]:
            
            x,y = i(x)
            output += y
            output.append(x)
        x = self.fork[-1](x)
        output.append(x) 
            
            
        
        
        return(torch.cat((x,output[-1]),1)),output

class Darknet(nn.Module):

    def __init__(self):

        super(Darknet,self).__init__()
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(3,32, kernel_size=3,stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(32)
                                    )
        self.split1 = Split(2,1,[32,64,64],[64,64,32,64,64,64])
        self.split2 = Split(3,2,[128,64,128,64],[128]+[64 for i in range(6)])
        self.split3 = Split(3,8,[128,128,256,128],[256]+[128 for i in range(26)])
        self.split4 = Split(3,8,[256,256,512,256],[512]+[256 for i in range(26)])
        self.split5 = Split(3,4,[512,512,1024,512],[1024]+[512 for i in range(14)])
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(1024,512, kernel_size=3,stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(512)
                                    )
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(512,1024, kernel_size=3,stride=1),
                                    Mish(),
                                    nn.BatchNorm2d(1024)
                                    )
        self.split= [self.split1,self.split2,self.split3,self.split4,self.split5]
        self.output = []
        
    def forward(self,x):
        
        x = self.conv1(x)
        for i in self.split:

            x,y = i(x)
            self.output += y
            self.output.append(x)
        x = self.conv2(x)
        self.output.append(x)

        x=self.conv3(x)
        self.output.append(x)

        

        
        return x

    def __getitem__(self,id):
        return self.output[id]
    def __len__(self):
        return len(self.output)

class ImageNet(nn.Module):

    def __init__(self):
        super(ImageNet,self).__init__()
        self.darknet = Darknet()
        self.pool = nn.AvgPool2d()
        self.dense = nn.Linear(1024,1000)

    def forward(self,x):
        x = self.darknet(x)
        x = nn.AvgPool2d(x.size()[-2:])(x)
        
        x = x.view(-1,1024)
        x = self.dense(x)

        return F.softmax(x)
                                     
