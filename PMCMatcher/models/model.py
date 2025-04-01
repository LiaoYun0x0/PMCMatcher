import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import spatial_soft_argmax2d, spatial_expectation2d
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange, repeat

from common.functions import * 
from common.nest import NestedTensor
from models.loftr import LoFTRModule
from models.position import PositionEmbedding2D, PositionEmbedding1D
from models.transformer import LocalFineFeatureTransformer
from models.networks import GLNet

from models.featenhance import LocalFeatureEnhancementModule

# transformer parameters
cfg={}
cfg["lo_cfg"] = {}
lo_cfg = cfg["lo_cfg"]
lo_cfg["d_model"] = 128
lo_cfg["layer_names"] = ["self","cross"] * 1
lo_cfg["nhead"] = 8
lo_cfg["attention"] = "linear"


def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        #self.elu = nn.ELU(inplace=True)
        self.mish = nn.Mish(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        #x = self.elu(x)
        x = self.mish(x)
        return x

class MatchingNet(nn.Module):
    def __init__(
        self,
        d_coarse_model: int=256,
        d_mid_model: int=196,
        d_fine_model: int=128,  
        n_coarse_layers: int=6,
        n_fine_layers: int=4,
        n_heads: int=8,
        backbone_name: str='resnet18',
        matching_name: str='sinkhorn',
        match_threshold: float=0.2,
        window: int=5,
        border: int=1,
        sinkhorn_iterations: int=50,
    ):
        super().__init__()
        
        
        self.localfeatenhance = LocalFeatureEnhancementModule(in_channels=256)
        self.projf4 = nn.Linear(d_coarse_model, d_mid_model, bias=True)
        self.mergef4 = nn.Linear(392, d_mid_model, bias=True)
        self.position1df4 = PositionEmbedding1D(d_mid_model, max_len=window**2)
        self.conv4d = ConvBN(d_coarse_model, 196, 1, 1)
        self.changechannnel = nn.Linear(d_mid_model, d_fine_model, bias=True)

        
        
        self.backbone = GLNet(backbone="resnet50")
        self.position2d = PositionEmbedding2D(d_coarse_model)
        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window**2)

        #self.local_transformer = LocalFeatureTransformer(cfg["lo_cfg"])
        self.local_transformer = LocalFineFeatureTransformer(cfg["lo_cfg"])
        
        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(d_coarse_model, d_fine_model, bias=True)

        self.conv2d = ConvBN(d_coarse_model, d_fine_model, 1, 1)

        self.regression1 = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.regression2 = nn.Linear(3200, d_fine_model, bias=True)
        self.regression = nn.Linear(d_fine_model, 2, bias=True)
        self.dropout = nn.Dropout(p=0.5)


        self.border = border
        self.window = window
        self.num_iter = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.matching_name = matching_name
        self.step_coarse = 8
        self.step_fine = 2
        self.step_mid = 4

        if matching_name == 'sinkhorn':
            bin_score = nn.Parameter(torch.tensor(1.))
            self.register_parameter("bin_score", bin_score)
        self.th = 0.1

    def fine_matching(self,x0,x1):
        x0,x1 = self.local_transformer(x0,x1)
        #x0, x1 = self.L2Normalize(x0, dim=0), self.L2Normalize(x1, dim=0)
        return x0,x1
    
    
    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat
    

    def compute_confidence_matrix(self, query_lf,refer_lf, gt_matrix=None):
        _d =  query_lf.shape[-1]
        
        query_lf = query_lf / _d
        refer_lf = refer_lf / _d
        
        similarity_matrix = torch.matmul(query_lf,refer_lf.transpose(1,2)) / 0.1
        #sim_matrix = torch.einsum("nlc,nsc->nls", query_lf, refer_lf) / 0.1
        confidence_matrix = torch.softmax(similarity_matrix,1) * torch.softmax(similarity_matrix,2)
        return confidence_matrix

    def unfold_within_window(self, featmap):
        
        scale = self.step_coarse - self.step_fine
        #stride = int(math.pow(2, scale))
        stride = 4
        #self.window=5
        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window//2
        )

        featmap_unfold = rearrange(
            featmap_unfold,
            "B (C MM) L -> B L MM C",
            MM=self.window ** 2
        )
        return featmap_unfold
    
   
    def unfold_f4_window(self, featmap):
        scale = self.step_coarse - self.step_mid
        #stride = int(math.pow(2, scale))
        stride = 2
        #self.window=5
        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window//2
        )

        featmap_unfold = rearrange(
            featmap_unfold,
            "B (C MM) L -> B L MM C",
            MM=self.window ** 2
        )
        return featmap_unfold


    def forward(self, samples0, samples1, gt_matrix):
        
        device = samples0.device

        mdesc0, mdesc1, fine_featmap0, fine_featmap1, lf0_4x,lf1_4x = self.backbone.forward_pair_lo(samples0, samples1)

###enhance
        N, hw, c = mdesc0.shape
        h8 = w8 = int(math.sqrt(hw))
        feat_c8_0_enhance = mdesc0.permute(0, 2, 1) #[n,256,1600]
        feat_c8_1_enhance = mdesc1.permute(0, 2, 1) 
        feat_c8_0_enhance = feat_c8_0_enhance.view(N, c, h8, w8)
        feat_c8_1_enhance = feat_c8_1_enhance.view(N, c, h8, w8)
        feat_c8_0_enhance = self.localfeatenhance(feat_c8_0_enhance)#[n, 256, 40, 40]
        feat_c8_1_enhance = self.localfeatenhance(feat_c8_1_enhance)
        N, C, H, W = feat_c8_0_enhance.shape
        feat_c8_0_final_flattened = feat_c8_0_enhance.view(N, C, H * W)
        feat_c8_1_final_flattened = feat_c8_1_enhance.view(N, C, H * W)
        feat_c8_0 = feat_c8_0_final_flattened.permute(0, 2, 1) # ([n, 1600, 256])
        feat_c8_1 = feat_c8_1_final_flattened.permute(0, 2, 1)
        cm_matrix = self.compute_confidence_matrix(feat_c8_0, feat_c8_1)#([n, 1600, 1600])
        
        cf_matrix = cm_matrix * (cm_matrix == cm_matrix.max(dim=2, keepdim=True)[0]) * (cm_matrix == cm_matrix.max(dim=1, keepdim=True)[0])
        #mask_v：Store the maximum matching confidence value of the query point
        #all_j_ids：Store the indices of the reference points corresponding to these maximum values
        mask_v, all_j_ids = cf_matrix.max(dim=2)
        #b_ids：Indices of the batches (if multiple batches of input are used)
        #i_ids：Indices of the query points, indicating where valid matches were found
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        # [batch_index, query_point_index, reference_point_index] 
        matches = torch.stack([b_ids, i_ids, j_ids]).T
        
        
        
        if matches.shape[0] == 0:
            return {
                        "cm_matrix": cm_matrix,
                        "feat_c8_0": feat_c8_0,
                        "feat_c8_1": feat_c8_1,
                        "mkpts1": torch.Tensor(0, 2),
                        'mkpts0': torch.Tensor(0, 2),
                        'samples0': samples0,
                        'samples1': samples1
                    }
        #Calculate the coordinates of each matching point in the query image and the reference image
        mkpts0, mkpts1 = batch_get_mkpts( matches, samples0, samples1)
        
        
        lf0_4x = self.conv4d(lf0_4x)
        lf1_4x  = self.conv4d(lf1_4x)
        
        #crop
        lf0_4x_unfold = self.unfold_f4_window(lf0_4x) # nx1600x25x196
        lf1_4x_unfold = self.unfold_f4_window(lf1_4x)
        
        # local_desc
        # [2*N, ww,C]
        local_f4_desc = torch.cat([
            lf0_4x_unfold[matches[:, 0], matches[:, 1]],
            lf1_4x_unfold[matches[:, 0], matches[:, 2]]
        ], dim=0)
        
        #center_desc
        # [2*N, ww,C]
        center_f4_desc = repeat(torch.cat([
            feat_c8_0[matches[:, 0], matches[:, 1]],
            feat_c8_1[matches[:, 0], matches[:, 2]]
            ], dim=0), 
            'N C -> N WW C', 
            WW=self.window**2)
        
         #256->196 
        center_f4_desc = self.projf4(center_f4_desc)
        # [2*N, WW, C1 + C2]
        local_f4_desc = torch.cat([local_f4_desc, center_f4_desc], dim=-1)
        #392->196 
        local_f4_desc = self.mergef4(local_f4_desc)
        #pos
        local_f4_position = self.position1df4(local_f4_desc)#[n, 25, 196]
        local_f4_desc = local_f4_desc + local_f4_position
        
        #local_desc  [2*N, WW, C]， desc0 and desc1 [N, WW, C]
        descf40, descf41 = torch.chunk(local_f4_desc, 2, dim=0)  #[n,25,196]
        descf40 = self.changechannnel(descf40)#[n, 196, 80, 80]
        descf41  = self.changechannnel(descf41)
        f4desc0, f4desc1 = self.fine_matching(descf40, descf41)

        c = self.window ** 2 // 2
        
        center_f4_desc = repeat(f4desc0[:, c, :], 'N C->N WW C', WW=self.window**2)
        
        #[N, WW, 2*C]
        center_f4_desc = torch.cat([center_f4_desc, f4desc1], dim=-1)
       
        expected_f4_coords = self._regression(center_f4_desc)
        
        mkpts1f4 = mkpts1[:, 1:] + expected_f4_coords
        
        fine_featmap0 = self.conv2d(fine_featmap0)
        fine_featmap1  = self.conv2d(fine_featmap1)
        
        #crop
        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0) # nx1600x25x256
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1)
        
        
        local_desc = torch.cat([ 
            fine_featmap0_unfold[matches[:, 0], matches[:, 1]],
            fine_featmap1_unfold[matches[:, 0], matches[:, 2]]
        ], dim=0)
        
        
        # [2*N, ww,C]
        center_desc = repeat(torch.cat([
            feat_c8_0[matches[:, 0], matches[:, 1]],
            feat_c8_1[matches[:, 0], matches[:, 2]]
            ], dim=0), 
            'N C -> N WW C', 
            WW=self.window**2)
        #256->128 
        center_desc = self.proj(center_desc)
        # [2*N, WW, C1 + C2]
        local_desc = torch.cat([local_desc, center_desc], dim=-1)
        #256->128 
        local_desc = self.merge(local_desc)
       
        local_position = self.position1d(local_desc)
        local_desc = local_desc + local_position
       
        
        desc0, desc1 = torch.chunk(local_desc, 2, dim=0)  #[n,25,128]
        fdesc0, fdesc1 = self.fine_matching(desc0, desc1)

        c = self.window ** 2 // 2
       
        center_desc = repeat(fdesc0[:, c, :], 'N C->N WW C', WW=self.window**2)
        # center_desc
        # [N, WW, 2*C]
        center_desc = torch.cat([center_desc, fdesc1], dim=-1)
        expected_coords = self._regression(center_desc)
        mkpts1 = mkpts1f4 + expected_coords

        return {
            "cm_matrix": cm_matrix,
            'matches': matches,
            'samples0': samples0,
            'samples1': samples1,
            'mkpts1f4': mkpts1f4,
            'mkpts1': mkpts1,
            'mkpts0': mkpts0,
            "feat_c8_0": feat_c8_0,
            "feat_c8_1": feat_c8_1,
        }
        
       