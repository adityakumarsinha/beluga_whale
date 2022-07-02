from pathlib import Path
import numpy as np
from loguru import logger
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors 
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.metrics import average_precision_score
import typer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F
import timm
import cv2
from types import SimpleNamespace
import albumentations as A
import torch
import torch.nn as nn
import timm
import torchvision
import pickle
#from sklearn.decomposition import PCA


ROOT_DIRECTORY = Path("/code_execution")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

import torch.nn as nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            
import torch.nn as nn
class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)



class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim =n_classes
            
    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean",label_smoothing=None,class_weights_norm=None ):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        if label_smoothing is None:
            self.crit = nn.CrossEntropyLoss(reduction="none")   
        else:
            self.crit = nn.CrossEntropyLoss(reduction="none",label_smoothing=label_smoothing)   
        
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss    

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=False, 
                                          num_classes=0, 
                                          global_pool="", 
                                          in_chans=self.cfg.in_channels,features_only = True)

        
        #if ("efficientnet" in cfg.backbone) & (self.cfg.stride is not None):
        #    self.backbone.conv_stem.stride = self.cfg.stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        #backbone_out_1 = self.backbone.feature_info[-2]['num_chs']

        if cfg.pool == "gem":
            self.global_pool1 = GeM(p_trainable=cfg.gem_p_trainable)
            self.global_pool2 = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool1 = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool1 = nn.AdaptiveAvgPool2d(1)
            self.global_pool2 = nn.AdaptiveAvgPool2d(1)

        self.embedding_size = backbone_out

        feature_dim_l_g = 688

        self.neck =nn.Sequential(nn.Linear(feature_dim_l_g, self.embedding_size, bias=True),nn.PReLU())
        self.bottleneck = nn.BatchNorm1d(self.embedding_size)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        
    def forward(self, x):

        #x = batch['input']
        
        dev = x.device

        x = self.backbone(x)
        
        x_l = self.global_pool1(x[-2])[:,:,0,0]
        x_g = self.global_pool2(x[-1])[:,:,0,0]

        x_g = torch.cat([x_g,x_l],axis=1) 
        x_g = self.neck(x_g)
        
        x_emb = self.bottleneck(x_g)

        return {'embeddings': x_emb}

    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False


    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True        

def load_model(model,ckpt_path):
    state = torch.load(ckpt_path)
    new_state = {}
    o = model.load_state_dict(state,strict=False)
    print('o',o)
    return model                    
                    
                    
class BelugaDataset(torch.utils.data.Dataset):

    def __init__(self, df,transforms=None):
        self.data = df
        self.transforms = transforms
        self.normalization = 'imagenet'
        
        
    def normalize_img(self,img):
        
        if self.normalization == 'channel':
            #print(img.shape)
            pixel_mean = img.mean((0,1))
            pixel_std = img.std((0,1)) + 1e-4
            img = (img - pixel_mean[None,None,:]) / pixel_std[None,None,:]
            img = img.clip(-20,20)
           
        elif self.normalization == 'image':
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20,20)
            
        elif self.normalization == 'simple':
            img = img/255
            
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'min_max':
            img = img - np.min(img)
            img = img / np.max(img)
            return img
        
        else:
            pass
        
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row.image_id
        fname = str(DATA_DIRECTORY / row.path)
        img = cv2.imread(fname,cv2.COLOR_BGR2RGB)
        
        if row.viewpoint != 'top':
            img = np.transpose(img, [1, 0, 2])
    
        if self.transforms:
            sample_d = self.transforms(image=img)
            img=sample_d["image"]  
            
            img = self.normalize_img(img)
            img = np.transpose(img, [2, 0, 1])

            sample = {"image_id": image_id, "image": img}
            del img

        return sample

cfg = SimpleNamespace(**{})

cfg.backbone = "tf_efficientnet_b5_ns"
cfg.pretrained = False
cfg.embedding_size = 512
cfg.pool = "gem"
cfg.gem_p_trainable = True
cfg.headless = False
cfg.in_channels = 3


cfg.batch_size=16
cfg.num_workers=0
cfg.img_size = (512,256)
# AUGS

image_size0 = cfg.img_size[0]
image_size1 = cfg.img_size[1]

cfg.val_aug = A.Compose([
        A.Resize(image_size0, image_size1),
    ])
 
def get_embeddings(model,df,cfg):
    
    train_dataset = BelugaDataset(
            df=df, transforms=cfg.val_aug
        )
    
    
    trainDataLoader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=cfg.batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=False,
                    )
    
    embeddings_dicts = {}
    total_step=0
    model.eval()
    #pbar= tqdm(enumerate(trainDataLoader),total=len(trainDataLoader))
    for bi,batch in enumerate(trainDataLoader) :
        
        x = batch["image"].float().cuda()
        # Forward pass & softmax
        with torch.no_grad():
            embedding = model(x)['embeddings']
            embedding = embedding.cpu().numpy()
        
        for ri,iid in enumerate(batch['image_id']):
            embeddings_dicts[iid] = embedding[ri]
         
    return embeddings_dicts        
    
    
def effb5_main():

    logger.info("Starting main script")
    # load test set data and pretrained model
    query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
    metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
    
    models = []
    for i in range(5):
        MDL_PATH = f'./effnet-b5/tf_efficientnet_b5_ns-fold{i}.pth'
        logger.info("Loading pre-trained model",MDL_PATH)
        model_state = torch.load(MDL_PATH)
        model = Net(cfg)
        #load model
        errout = model.load_state_dict(model_state,strict=False)
        model = model.cuda()
        model = model.eval()
        models.append(model)
        print('model loaded',errout)
    
   
    
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]
    
    metadata_1 = metadata.reset_index(drop=False)
    
    test_embeddings_list  = []
    for model in models:
        test_embeddings_list.append( get_embeddings(model,metadata_1,cfg))
      
    test_embeddings_arr = []
    for k in test_embeddings_list[0].keys():
        emb = np.concatenate([te[k] for te in test_embeddings_list])
        test_embeddings_arr.append(emb)
    test_embeddings_arr = np.stack(test_embeddings_arr)
    
    print('test_embeddings_arr',test_embeddings_arr.shape)

    
    test_embeddings = {}
    for i,k in enumerate(test_embeddings_list[0].keys()):
        emb = test_embeddings_arr[i]
        test_embeddings[k] = emb
    
    with open('./effnet_b5_embeddings.pkl', 'wb') as handle:
        pickle.dump(test_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    effb5_main()