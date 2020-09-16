import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import common,segmentation_metrics
import torch.optim as optim
import numpy as np


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

        self.start = nn.ModuleList(self.start)

            
        
        self.fork = [nn.Sequential(
                nn.Conv2d(post_arg[0],post_arg[1], kernel_size=1,stride=1),
                Mish(),
                nn.BatchNorm2d(post_arg[1])
                )]

        for i in range(0,post*3,3):
            self.fork += [ResBlock(post_arg[i+1],post_arg[i+2],post_arg[i+3])]

        self.fork = nn.ModuleList(self.fork)
        
        
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
        self.dense = nn.Linear(1024,1000)

    def forward(self,x):
        x = self.darknet(x)
        x = F.avg_pool2d(x, x.size()[-2:])
        x = x.view(-1,1024)
        x = self.dense(x)

        return F.softmax(x)
                                     
class Detector():
    
    def __init__(self, config, dataset_helper, device):
        self.config = config
        self.device = device
        self.dataset_helper = dataset_helper
        
        self.n_classes = self.config['num_classes']
        self.model = ImageNet().to(self.device)
        
        self.epsilon = self.config["epsilon"]
        
        self.epochs = self.config["epochs"]
        self.lr = self.config["lr"]
        self.lr_decay = self.config["lr_decay"]
        self.iter = 0
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.config["momentum"], weight_decay=self.config["w_decay"])
        self.evaluator = segmentation_metrics.SegmentationMetrics(self.n_classes, self.device)
    
    def train(self, dataloader, writer):
        torch.cuda.empty_cache()
        self.model.train()
        epoch_logs = {"loss": [], "acc": []}
        
        for _, (img,label) in enumerate(dataloader):
            img = img.to(self.device)
            pred = self.model(img).to(self.device)
            #label = ((1 - self.epsilon)*label + (self.epsilon / self.n_classes)).to(self.device)
            label = label.to(self.device)
            loss = self.loss(pred, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.evaluator.reset()
            pred_argmax = loss.argmax(dim=-1)
            self.evaluator.addbatch(pred_argmax, label)
            accuracy = self.evaluator.getacc()
            
            epoch_logs["loss"].append(loss.mean().item())
            epoch_logs["acc"].append(accuracy.item())
            

            

        self.iter+=1
        decay = (1-(self.iter/self.epochs))**self.lr_decay
        self.lr, self.optimizer = common.adjust_lr(self.lr, self.optimizer,decay, mode="prod")
        writer.add_scalar("Loss/Train",np.mean(epoch_logs["loss"]),self.iter -1)
        writer.add_scalar("Accuracy/Train",np.mean(epoch_logs["acc"]),self.iter -1)
        return epoch_logs
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    def valid(self, dataloader, writer):
        torch.cuda.empty_cache()
        self.model.eval()
        epoch_logs = {"loss": [], "acc": [], "iou": [], "c_iou": []}
        
        for indx, (img,label) in enumerate(dataloader):
            print(indx)
            img = img.to(self.device)
            
            with torch.no_grad():
                pred = self.model(img)
                loss = self.loss(pred, label)
            
            self.evaluator.reset()
            pred_argmax = pred.argmax(dim=1)
            self.evaluator.addbatch(pred_argmax, label)
            accuracy = self.evaluator.getacc()
            
            epoch_logs["loss"].append(loss.mean().item())
            epoch_logs["acc"].append(accuracy.item())
            

        writer.add_scalar("Loss/Val",np.mean(epoch_logs["loss"]),self.iter -1)
        writer.add_scalar("Accuracy/Train",np.mean(epoch_logs["acc"]),self.iter -1)

        return epoch_logs