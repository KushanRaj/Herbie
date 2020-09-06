import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from modules.CSPDarknet53 import Darknet
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from utils import *




class SPP(nn.Module):

    def __init__(self):
        super(SPP,self).__init__()
        self.maxpool = nn.MaxPool2d(5,stride=1,padding=4)   
        self.maxpool = nn.MaxPool2d(9,stride=1,padding=8)
        self.maxpool = nn.MaxPool2d(13,stride=1,padding=12)
        
    def forward(self,x):
        output = []
        m1 = self.maxpool1(x)
        output.append(m1)
        output.append(x)
        m2 = self.maxpool2(x)
        output.append(m2)
        output.append(x)
        m3 = self.maxpool3(x)
        output.append(m3)
        output.append(torch.cat((m3,m2,m1,x),1))
        return output



class anchor(nn.Module):

    def __init__(self,anchors,img_size,device,xy_scale):

        super(anchor,self).__init__()

        self.num = len(anchors)
        self.anchors = anchors
        self.imsize = img_size
        self.device = device
        self.xy_scale = xy_scale
        self.iou_scale = 3.54
        self.cls_scale = 37.4
        self.obj_scale = 64.3
        self.euler_scale = 3.54
        self.obj_thresh = 0.7
        self.thresh = 0.213
        self.num_class = 7

    def get_iou(self,target,anchor,sizes):

        

        iou_val = torch.zeros((len(target),len(anchor[0])),device=self.device) 

        
        for ind,val in enumerate(target):
                for a_ind,a_val in enumerate(anchor[sizes[ind]]):
                                   
                    iou_val[ind,a_ind] = val.intersection(a_val).area/(val.area + a_val.area - val.intersection(a_val).area)
                
        return iou_val
                
    def get_best(self,iou,lengths):

        start = 0
        obj_mask = torch.zeros((iou.size(0))).cpu()
        for val in lengths:

            temp = iou[start:start + val]
                
            temp = torch.cat((
                       torch.cat((
                                  torch.tensor([-i-1 for i in range(val)],device=self.device).view(val,1),temp),1),
                                  torch.tensor([[-i-1 for i in range(anchors.size(1)+1)]],device=self.device)
                                  ))

            
            
            for j in range(val):
                        

                        
                        ind,pos = torch.argmax(temp)//(temp.size(1)),torch.argmax(temp)%(temp.size(1))
                        
                        
                        obj_mask[-(torch.min(temp[ind])+1).long()+start] = -(torch.min(temp[...,pos]) + 2) 
                        

                        temp = torch.roll(temp,(-(ind+1),-(pos+1)),(0,1))[:-1,:-1]
                        
                        
                
            start += val

        return obj_mask.long()

 
    
    def get_corner(self,pred,target=False):
        if not target:
            corners = torch.zeros((pred.size(0),pred.size(1),4,2),device=self.device)
            corners[...,0] = (pred[...,-4].repeat(4,1,1) + 
                              torch.stack((pred[...,-2]*torch.cos(pred[...,1])-pred[...,-1]*torch.sin(pred[...,1])
                                         
                                         ,pred[...,-2]*torch.cos(pred[...,1])+pred[...,-1]*torch.sin(pred[...,1])
                                         
                                         ,-pred[...,-2]*torch.cos(pred[...,1])+pred[...,-1]*torch.sin(pred[...,1])
                                         
                                         ,-pred[...,-2]*torch.cos(pred[...,1])-pred[...,-1]*torch.sin(pred[...,1])
                                         
                                         ))).T.contiguous().view(pred.size(0),pred.size(1),-1)/2


            corners[...,1] = (pred[...,-3].repeat(4,1,1) + 
                              torch.stack((pred[...,-2]*torch.sin(pred[...,1])+pred[...,-1]*torch.cos(pred[...,1]),

                                          pred[...,-2]*torch.sin(pred[...,1])-pred[...,-1]*torch.cos(pred[...,1]),

                                          -pred[...,-2]*torch.sin(pred[...,1])-pred[...,-1]*torch.cos(pred[...,1]),

                                          -pred[...,-2]*torch.sin(pred[...,1])+pred[...,-1]*torch.cos(pred[...,1])

                                          ))).T.contiguous().view(pred.size(0),pred.size(1),-1)/2
            


        else:
            corners = torch.zeros((pred.size(0),4,2),device=self.device)
            corners[...,0] = (pred[...,-4].repeat(4,1) + 
                            torch.stack((pred[...,-2]*torch.cos(pred[...,1])-pred[...,-1]*torch.sin(pred[...,1])
                                            
                                        ,pred[...,-2]*torch.cos(pred[...,1])+pred[...,-1]*torch.sin(pred[...,1])
                                            
                                        ,-pred[...,-2]*torch.cos(pred[...,1])+pred[...,-1]*torch.sin(pred[...,1])
                                        
                                        ,-pred[...,-2]*torch.cos(pred[...,1])-pred[...,-1]*torch.sin(pred[...,1])
                                            
                                        ))/2).T.contiguous().view(pred.size(0),-1)


            corners[...,1] = (pred[...,-3].repeat(4,1) + 
                            torch.stack((pred[...,-2]*torch.sin(pred[...,1])+pred[...,-1]*torch.cos(pred[...,1]),

                            pred[...,-2]*torch.sin(pred[...,1])-pred[...,-1]*torch.cos(pred[...,1]),

                            -pred[...,-2]*torch.sin(pred[...,1])-pred[...,-1]*torch.cos(pred[...,1]),

                            -pred[...,-2]*torch.sin(pred[...,1])+pred[...,-1]*torch.cos(pred[...,1])

                            ))/2).T.contiguous().view(pred.size(0),-1)
        return corners
        
        
    def get_giou(self,pred_corners,gt_corners,pred_poly,gt_poly):
        giou_loss = 0.
        for a,gt,a_poly,t_poly in zip(pred_corners,gt_corners,pred_poly,gt_poly):
            convex_conners = torch.cat((a,gt), dim=0)
            hull = ConvexHull(convex_conners.clone().detach().cpu().numpy()) 
            pts = convex_conners[hull.vertices]
            roll_pts = torch.roll(pts, -1, dims=0)
            convex_area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
            union = t_poly.area + a_poly.area - t_poly.intersection(a_poly).area
            iou = t_poly.intersection(a_poly).area/union
            giou_loss += 1. - (iou - (convex_area - union) /convex_area)
        return giou_loss

    def forward(self,x,targets,lengths):

        B,C,G,G = x.size()
        
        
        C /= self.num
        y = x.view(B,self.num,C,G,G).permute(0,1,3,4,2).contiguous()

        off_x = torch.sigmoid(y[..., 0]) 
        off_y = torch.sigmoid(y[..., 1])
        scale_w = torch.exp(y[..., 2])  
        scale_h = torch.exp(y[..., 3])  
        im = y[..., 4]  
        re = y[..., 5]  
        anchor_conf = torch.sigmoid(y[..., 6])  
        anchor_cls = y[..., 7:]
       

        grid_x = torch.tensor(range(0,G)).repeat((G,1))
        grid_y = grid_x.t()
        stride = self.imsize/G
        anchor_x = torch.floor((off_x + grid_x)*stride)
        anchor_y = torch.floor((off_y + grid_y)*stride)
        anchor_yaw = torch.atan(torch.true_divide(im,re))
        anchor_w = scale_w*anchor[...,0:1].repeat(1,G*G).view(1,self.num,G,G).repeat(B,1,1,1)
        anchor_h = scale_h*anchor[...,1:].repeat(1,G*G).view(1,self.num,G,G).repeat(B,1,1,1)

        pred = torch.cat((im,re,anchor_yaw,anchor_conf,anchor_x,anchor_y,anchor_w,anchor_h,anchor_cls.permute(0,1,4,2,3).contiguous().view(B,self.num_class*self.num,G,G)) , 1
                        ).view(B,15,self.num,G,G).permute(0,2,3,4,1).view(B,self.num*G*G,15).contiguous().to(self.device)

        if targets:
            assert targets.size(0) == pred.size(0)

            sizes = [ind for ind,i in enumerate(lengths) for j in range(i) ]

            gt = torch.zeros((lengths.sum(),targets.size(-1)),device=self.device).float() 


            pred_corners = self.get_corner(pred)
            pred_poly = np.array([list(map(lambda x : Polygon(x),i)) for i in pred_corners])

            for i,val in enumerate(targets):
                
                gt[torch.sum(lengths[:i]):torch.sum(lengths[:i+1])] =  val[:lengths[i]]

            gt_corners = self.get_corner(gt,True)

            gt_poly = np.array(list(map(lambda x : Polygon(x),gt_corners)))

            iou = self.get_iou(gt_poly,pred_poly,sizes)
            obj_mask = (sizes,self.get_best(iou,lengths))
            cond = torch.where(iou>self.thresh)
            nobj = torch.ones((B,self.num*G*G),device=self.device)
            nobj[[sizes[i] for i in cond[0]],cond[1]] = 0
            nobj[obj_mask] = 0
            nobj_mask = torch.where(nobj==1)

            loss_regression = F.mse_loss(pred[obj_mask][...,2], gt[...,2]) + \
                              F.mse_loss(pred[obj_mask][...,2], gt[...,3]) + \
                              F.mse_loss(pred[obj_mask][...,2], gt[...,4]) + \
                              F.mse_loss(pred[obj_mask][...,2], gt[...,5])
            
            loss_im = F.mse_loss(pred[obj_mask][...,0], torch.sin(gt[...,1]))
            loss_re = F.mse_loss(pred[obj_mask][...,1], torch.cos(gt[...,1]))
            loss_circle = ((1. - torch.sqrt(pred[obj_mask][...,0] ** 2 + pred[obj_mask][...,1] ** 2)) ** 2).mean()  
            
            loss_eular = loss_im + loss_re + loss_circle

            loss_obj = F.binary_cross_entropy_with_logits(pred[obj_mask][...,3], torch.ones_like(pred[obj_mask][...,3])) + \
                       F.binary_cross_entropy_with_logits(pred[nobj_mask][...,3], torch.zeros_like(pred[nobj_mask][...,3]))
            loss_cls = F.binary_cross_entropy_with_logits(pred[obj_mask][...,8:],gt[...,:7])

            loss_giou = self.get_giou(pred_corners[obj_mask],gt_corners,pred_poly[obj_mask],gt_poly)

            total_loss = loss_giou * self.iou_scale + loss_cls * self.cls_scale + loss_eular * self.eular_scale + loss_obj * self.obj_scale + loss_regression * self.xy_scale

            

            cls_acc = 100 * torch.argmax(pred_corners[obj_mask][...,8:]).float().mean()
            avg_iou = iou[obj_mask].sum()
            
            precision = (iou*(pred[...,3]>self.obj_thresh).float()[sizes]).sum()/(pred[...,3]>self.obj_thresh).float()[sizes].sum()
            
            metrics = {
                "loss": total_loss,
                "iou_score": avg_iou,
                'giou_loss': loss_giou,
                'loss_regression': loss_regression,
                'loss_eular': loss_eular,
                'loss_im': loss_im,
                'loss_re': loss_re,
                "loss_obj": loss_obj,
                "loss_cls": loss_cls,
                "cls_acc": cls_acc,
                "precision": precision,
                
            }

            
            return pred,total_loss, metrics
        else:
            return pred,None,None


            '''
            loss_x = F.mse_loss(pred[obj_mask][...,2], gt[...,2])
            loss_y = F.mse_loss(pred[obj_mask][...,2], gt[...,3])
            loss_w = F.mse_loss(pred[obj_mask][...,2], gt[...,4])
            loss_h = F.mse_loss(pred[obj_mask][...,2], gt[...,5])
            batch=[]
            anchor=[]
            for i in zip(nobj[0],nobj[1]):
                if i not in zip(obj_mask[0],obj_mask[1]):
                    batch.append(i[0])
                    anchor.append(i[1])
            nobj_mask = (batch,anchor)

            
            gt = torch.zeros((torch.sum(sizes),6),device=self.device)

            g_cls = torch.zeros((B,self.num*G*G,1),device=self.device)
            g_yaw = torch.zeros((B,self.num*G*G,1),device=self.device)
            g_conf = torch.zeros((B,self.num*G*G,1),device=self.device)
            gx = torch.zeros((B,self.num*G*G,1),device=self.device)
            gy = torch.zeros((B,self.num*G*G,1),device=self.device)
            gh = torch.zeros((B,self.num*G*G,1),device=self.device)
            gw = torch.zeros((B,self.num*G*G,1),device=self.device)
            

            g_cls[obj_mask] = gt[...,0].view(-1,1)

            g_conf[obj_mask]  = 1
            g_yaw[obj_mask]  = gt[...,1].view(-1,1)
            gx[obj_mask] = gt[...,2].view(-1,1)
            gy[obj_mask] = gt[...,3].view(-1,1)
            gw[obj_mask] = gt[...,4].view(-1,1)
            gh[obj_mask] = gt[...,5].view(-1,1)
            '''
       
class loop_conv(nn.Module):

    def __init__(self,l,filters,k,p):
        super(loop_conv,self).__init__()
        model = []
        for i in range(l):
            model += nn.Sequential(
                                    nn.Conv2d(filters[i],filters[i+1], kernel_size=k[i],padding=p[i],stride=1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(filters[i+1])
                                    )
        self.model = nn.ModuleList(model)
        
    def forward(self,x):
        output = []
        for i in self.model:
            x = i(x)
            output.append(x)
        return output


class upsample(nn.Module):

    def __init__(self,stride,infilter,filter):
        super(upsample,self).__init__()

        self.loop = loop_conv(1,[infilter,filter],[1],[0])

        self.stride = stride


    def scale(self,x):
        
        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1
        ).expand(
                x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride
                ).contiguous().view(
                                    x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride
                                    )
        return x

    def forward(self,x,y):
        output = []
        
        output.append(scale(x))
        output.append(self.loop(y))
        output.append(torch.cat((output[0],output[1]),1))
        return output

class Yolo(nn.Module):

    def __init__(self,config,device):
        super(Yolo,self).__init__()

        self.darknet = Darknet()
        self.loop1 = loop_conv(3,[1024,512,1024,512],[1,3,1],[0,1,0])
        self.loop2 = loop_conv(4,[512*4,512,1024,512,256],[1,3,1,1],[0,1,0,0])
        self.loop3 = loop_conv(6,[512,256,512,256,512,256,128],[1,3,1,3,1,1],[0,1,0,1,0,0])
        self.loop4 = loop_conv(5,[256,128,256,128,256,128],[1,3,1,3,1],[0,1,0,1,0])
        self.loop5 = loop_conv(1,[128,512],[3],[1])
        self.loop6 = loop_conv(6,[512,256,512,256,512,256,512],[1,3,1,3,1,3],[0,1,0,1,0,1])
        self.loop7 = loop_conv(6,[1024,512,1024,512,1024,512,1024],[1,3,1,3,1,3],[0,1,0,1,0,1])
        self.upsample1 = upsample(2,512,256)
        self.upsample2 = upsample(2,256,128)
        self.anchor1 = anchor([],512,device,1.2)
        self.anchor2 = anchor([],512,device,1.1)
        self.anchor3 = anchor([],512,device,1.05)
        self.conv = nn.Conv2d(512,3*14, 1, padding = 0,stride=1)
        self.conv2 =  nn.Sequential(
                                    nn.Conv2d(128,256, 3,padding=1,stride=2),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(256)
                                    )
        self.conv3 = nn.Conv2d(512,3*14, 1, padding = 0,stride=1)
        self.conv4 =  nn.Sequential(
                                    nn.Conv2d(256,512, 3,padding=1,stride=2),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(512)
                                    )
        self.conv5 = nn.Conv2d(1024,3*14, 1, padding = 0,stride=1)                            
        self.SPP = SPP()
        self.outputs = []
                                
        
        


    def forward(self,x,boxes,lenghts):
        x = self.darknet(x)
        self.outputs += self.darknet[:]
        self.outputs += self.loop1(x)
        
        self.outputs += self.SPP(self.outputs[-1])
        
        self.outputs += self.loop2(self.outputs[-1])
        self.outputs += self.upsample1(self.outputs[-1],darknet[85])
        self.outputs += self.loop3(self.outputs[-1])
        self.outputs += self.upsample2(self.outputs[-1],darknet[54])
        self.outputs += self.loop4(self.outputs[-1])
        self.outputs += self.loop5(self.outputs[-1])
        self.outputs.append(self.conv(self.outputs[-1]))

        pred,loss,_ = self.anchor1(x,boxes,lenghts)

        self.outputs.append(pred)
        self.outputs.append(self.outputs[-4])
        self.outputs.append(self.conv2(self.outputs[-1]))
        self.outputs.append(torch.cat((self.outputs[-1],self.outputs[-16]),1))
        self.outputs += self.loop6(self.outputs[-1])
        self.outputs.append(self.conv3(self.outputs[-1]))

        pred,loss2,_ = self.anchor2(x,boxes,lenghts)
        
        self.outputs.append(pred)
        self.outputs.append(self.outputs[-4])
        self.outputs.append(self.conv4(self.outputs[-1]))
        self.outputs.append(torch.cat((self.outputs[-1],self.outputs[-37]),1))
        self.outputs += self.loop7(self.outputs[-1])
        self.outputs.append(self.conv5(self.outputs[-1]))

        pred,loss3,metrics = self.anchor3(x,boxes,lenghts)
        self.outputs.append(pred)



        return pred,loss+loss2+loss3,metrics

                                   
 
class Complex_Yolo():
    def __init__(self, config, dataset_helper, device):

        self.config = config
        self.device = device
        self.dataset_helper = dataset_helper
        #self.ignore_class = 0
        '''
        learning_map_inv = dataset_helper.LEARNING_MAP_INV
        learning_map = dataset_helper.LEARNING_MAP
        content = dataset_helper.CONTENT
        learning_ignore = dataset_helper.LEARNING_IGNORE
        

        if self.config["uncertainity"] == False:
            self.model = Model(self.n_classes).to(self.device)
        '''
        n_classes = self.config['num_classes']
        epsilon = self.config["epsilon"]
        content = torch.zeros(self.n_classes, dtype=torch.float)
        
        for cl, freq in content.items():
            content[learning_map[cl]] += freq
        
        self.loss_weights = 1/(content+epsilon)
        self.loss_weights[self.ignore_class] = 0

        self.lr = self.config["lr"]
        self.lr_decay = self.config["lr_decay"]
        #self.criterion = nn.NLLLoss(weight=self.loss_weights).to(self.device)
        #self.loss = lovasz_softmax.Lovasz_softmax(ignore=self.ignore_class).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.config["momentum"], weight_decay=self.config["w_decay"])
        self.evaluator = segmentation_metrics.SegmentationMetrics(n_classes, self.device, [self.ignore_class])
    
    def train(self, dataloader, writer):
        torch.cuda.empty_cache()
        self.model.train()
        epoch_logs = {"loss": [], "acc": [], "iou": [], "c_iou": []}
        
        for indx, data in enumerate(dataloader):
            proj, proj_labels = data["proj"].to(self.device), data["proj_labels"].to(self.device).long()
            pred_proj_labels = self.model(proj_labels)
            loss = self.criterion(torch.log(pred_proj_labels.clamp(min=1e-8)), proj_labels) + self.loss(pred_proj_labels, proj_labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.evaluator.reset()
            pred_argmax = pred_proj_labels.argmax(dim=1)
            self.evaluator.addbatch(pred_argmax, proj_labels)
            accuracy = self.evaluator.getacc()
            iou, class_iou = self.evaluator.getiou()
            epoch_logs["loss"].append(loss.mean().item())
            epoch_logs["acc"].append(accuracy.item())
            epoch_logs["iou"].append(iou.item())
            epoch_logs["c_iou"].append(class_iou.item())

            writer.log(epoch_logs, mode="train")

        self.lr, self.optimizer = common.adjust_lr(self.lr, self.optimizer, self.lr_decay, mode="prod")
        return epoch_logs
    
    def save_model(self, path):
        torch.save({self.model.state_dict(), path})

    def valid(self, dataloader, writer):
        torch.cuda.empty_cache()
        self.model.eval()
        epoch_logs = {"loss": [], "acc": [], "iou": [], "c_iou": []}
        
        for indx, data in enumerate(dataloader):
            proj, proj_labels = data["proj"].to(self.device), data["proj_labels"].to(self.device).long()
            with torch.no_grad():
                pred_proj_labels = self.model(proj_labels)
                loss = self.criterion(torch.log(pred_proj_labels.clamp(min=1e-8)), proj_labels) + self.loss(pred_proj_labels, proj_labels)
            
            self.evaluator.reset()
            pred_argmax = pred_proj_labels.argmax(dim=1)
            self.evaluator.addbatch(pred_argmax, proj_labels)
            accuracy = self.evaluator.getacc()
            iou, class_iou = self.evaluator.getiou()
            epoch_logs["loss"].append(loss.mean().item())
            epoch_logs["acc"].append(accuracy.item())
            epoch_logs["iou"].append(iou.item())
            epoch_logs["c_iou"].append(class_iou.item())

            writer.log(epoch_logs, mode="valid")

        return epoch_logs        


            







            



