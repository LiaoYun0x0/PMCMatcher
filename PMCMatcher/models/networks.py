import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet.resnet import resnet18,resnet50
from models.transformer import PositionEncodingSine
from models.mtransformer import MLocalFeatureTransformer
from models.channelscore import LoFTRWithChannelenhance
import sys
import math

sys.path.append("../")
cfg = {}
cfg["arc_m"] = 0.2
cfg["arc_s"] = 64
cfg["local_feature_dim"] = 512
cfg["global_feature_dim"] = 256
cfg['temp_bug_fix'] = False
cfg["model_weights"] = "./weights/weights_lo_1019/GLNet_55000_481.488.tar"


cfg["lo_cfg"] = {}
lo_cfg = cfg["lo_cfg"]
lo_cfg["d_model"] = 256
lo_cfg["layer_names"] = ["self","cross"] * 4
lo_cfg["nhead"] = 8
lo_cfg["attention"] = "linear"


def normalize(x):
    '''
    scale the vector length to 1.
    params:
        x: torch tensor, shape "[...,vector_dim]"
    '''
    norm = torch.sqrt(torch.sum(torch.pow(x,2),-1,keepdims=True))
    return x / (norm)


class GLNet(torch.nn.Module):
    def __init__(self,config=cfg,backbone="resnet18"):
        super(GLNet,self).__init__()
        self.backbone = eval(backbone)(include_top=False)
        
        if backbone == "resnet50":
            self.feature_conv16 = nn.Conv2d(1024,config["local_feature_dim"],1,1,0,bias=False)
            self.feature_conv32 = nn.Conv2d(2048,config["local_feature_dim"],1,1,0,bias=False)
            self.feature_conv8 = nn.Conv2d(512,config["global_feature_dim"],1,1,0,bias=False)
        else: 
            self.feature_conv16 = nn.Conv2d(256,config["local_feature_dim"],1,1,0,bias=False)
            self.feature_conv32 = nn.Conv2d(512,config["global_feature_dim"],1,1,0,bias=False)
        self.pos_encoding = PositionEncodingSine( 
            config['global_feature_dim'], 
            temp_bug_fix=config['temp_bug_fix'])
        self.local_transformer = MLocalFeatureTransformer(config["lo_cfg"])
        self.channelenhance = LoFTRWithChannelenhance(reduction_ratio=0.5)
        self.dropout = nn.Dropout(p=0.5)

    
    def attention_local_feature(self,x0,x1):
        b,c,h0,w0 = x0.shape
        _,_,h1,w1 = x1.shape
        size = (h0,w0)
        x0 = self.pos_encoding(x0)
        x1 = self.pos_encoding(x1)
        x0 = x0.view(b,c,h0*w0).transpose(1,2)
        x1 = x1.view(b,c,h1*w1).transpose(1,2)
        x0,x1 = self.local_transformer(x0,x1)
        return x0,x1

    def forward_pair_lo(self,pbatch,abatch):
     
        lf0_4x,lf0_8x,lf0_16x,lf0_32x, lf0_2x = self.backbone.extract_endpoints(pbatch)
        lf1_4x,lf1_8x,lf1_16x,lf1_32x, lf1_2x = self.backbone.extract_endpoints(abatch)
        _, _, H032, W032 = lf0_32x.size() 
        _, _, H132, W132 = lf1_32x.size()
        _, _, H016, W016 = lf0_16x.size()  
        _, _, H116, W116 = lf1_16x.size()
        _, _, H08, W08 = lf0_8x.size()  
        _, _, H18, W18 = lf1_8x.size()

        lf0_32x = self.feature_conv32(lf0_32x)
        lf1_32x = self.feature_conv32(lf1_32x)
        #trans[n, 256, 10, 10]
        feat_c32_0_selected, feat_c32_0_remaining,feat_c32_1_selected, feat_c32_1_remaining = self.channelenhance(lf0_32x, lf1_32x)
        #pos[n, 100, 256]
        feat_c32_0_selected,feat_c32_1_selected = self.attention_local_feature(feat_c32_0_selected,feat_c32_1_selected)
        
        N, hw, rc = feat_c32_0_selected.shape  #  [N, H*W, rc]
        feat_c32_0_selected = feat_c32_0_selected.permute(0, 2, 1)  # [N, rc, H*W]-(, 256, 100)
        feat_c32_0_selected = feat_c32_0_selected.view(N, rc, H032, W032) #  [N, rc, H, W]-[, 256, 10, 10]
      
        feat_c32_1_selected = feat_c32_1_selected.permute(0, 2, 1)  
        feat_c32_1_selected = feat_c32_1_selected.view(N, rc, H132, W132)
    
        feat_c32_0 = torch.cat([feat_c32_0_selected, feat_c32_0_remaining], dim=1)  # [512, 10, 10])
        feat_c32_1 = torch.cat([feat_c32_1_selected, feat_c32_1_remaining], dim=1) 
        
        feat_c32_0_upsampled = F.interpolate(feat_c32_0, size=(H016, W016), mode='nearest') # [512, 20, 20]
        feat_c32_1_upsampled = F.interpolate(feat_c32_1, size=(H116, W116), mode='nearest')
    
        lf0_16x = self.feature_conv16(lf0_16x)
        lf1_16x = self.feature_conv16(lf1_16x)
        lf0_16x = feat_c32_0_upsampled + lf0_16x # [, 512, 20, 20]
        lf1_16x = feat_c32_1_upsampled + lf1_16x
        #[, 256, 20, 20]
        feat_c16_0_selected, feat_c16_0_remaining,feat_c16_1_selected, feat_c16_1_remaining = self.channelenhance(lf0_16x, lf1_16x)
        #pos[, 400, 256]
        feat_c16_0_selected,feat_c16_1_selected = self.attention_local_feature(feat_c16_0_selected,feat_c16_1_selected)
        
        N, hw, rc = feat_c16_0_selected.shape  #  [N, H*W, rc]
        feat_c16_0_selected = feat_c16_0_selected.permute(0, 2, 1)  #  [N, rc, H*W]-(, 256, 400)
        feat_c16_0_selected = feat_c16_0_selected.view(N, rc, H016, W016) #[N, rc, H, W]-[, 256, 20, 20]
      
        feat_c16_1_selected = feat_c16_1_selected.permute(0, 2, 1)  
        feat_c16_1_selected = feat_c16_1_selected.view(N, rc, H116, W116)
    
        feat_c16_0 = torch.cat([feat_c16_0_selected, feat_c16_0_remaining], dim=1)  # [, 512, 20, 20])
        feat_c16_1 = torch.cat([feat_c16_1_selected, feat_c16_1_remaining], dim=1) 
        
        feat_c16_0_upsampled = F.interpolate(feat_c16_0, size=(H08, W08), mode='nearest') #[, 512, 40, 40]
        feat_c16_1_upsampled = F.interpolate(feat_c16_1, size=(H18, W18), mode='nearest')
      
        lf0_8x = feat_c16_0_upsampled + lf0_8x 
        lf1_8x = feat_c16_1_upsampled + lf1_8x
        lf0 = self.feature_conv8(lf0_8x) 
        lf1 = self.feature_conv8(lf1_8x)
        lf0,lf1 = self.attention_local_feature(lf0,lf1)

        return lf0,lf1, lf0_2x, lf1_2x,lf0_4x,lf1_4x



if __name__ == "__main__":
    m1 = GLNet()
    
    for p in m1.parameters():
        p.data = p.data * 2 + 1

    for p in m1.parameters():
        print(p)
    

        
