import torch
import torch.nn as nn
import torch.nn.functional as F


class Channelenhance(nn.Module):
    def __init__(self, reduction_ratio=0.5):
        super(Channelenhance, self).__init__()
        self.reduction_ratio = reduction_ratio

        # avg_pool
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图，形状为 (N, C, H, W)
        Returns:
            selected_feats: 选中的前 rc 个通道特征，形状为 (N, rc, H, W)
            remaining_feats: 剩余的通道特征，形状为 (N, (1-r)c, H, W)
        """
        device = x.device
        N, C, H, W = x.size()
        # print(f"Input shape: {x.shape}")

        # 1. avg_pool z(N, C)  
        z = F.adaptive_avg_pool2d(x, 1).view(N, C)  # (N, C)

        # 2. FC
        fc1 = nn.Linear(C, C // 2).to(device)
        fc2 = nn.Linear(C // 2, C).to(device)

        # 3. FFN， scores
        scores = fc1(z)  # (N, C // 2)
        scores = self.relu(scores)
        scores = fc2(scores)  # (N, C)
        scores = self.sigmoid(scores)  # (N, C)

        # 4. SORT,SELECT rc 
        _, indices = torch.sort(scores, dim=1, descending=True)  # sort
        rc = int(self.reduction_ratio * C)  # compute
        selected_indices = indices[:, :rc]  
        remaining_indices = indices[:, rc:]  
        # print(f"Selected indices shape: {selected_indices.shape}")
        # print(f"Remaining indices shape: {remaining_indices.shape}")

        # gather 
        selected_feats = torch.gather(x, 1, selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))
        remaining_feats = torch.gather(x, 1, remaining_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))

        return selected_feats, remaining_feats #  (N, rc, H, W) (N, (1-r)c, H, W)


class LoFTRWithChannelenhance(nn.Module):
    def __init__(self, reduction_ratio):
        super(LoFTRWithChannelenhance, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channel_enhance = Channelenhance(reduction_ratio)

    def forward(self, feat_c0, feat_c1):
        """
        Args:
            feat_c0: 输入图像0的特征,形状为 (N, C, H, W)
            feat_c1: 输入图像1的特征,形状为 (N, C, H, W)
        Returns:
            拼接后的最终输出特征，形状与输入相同 (N, C, H, W)
        """
        # 对输入的 feat_c0 和 feat_c1 进行通道注意力处理
        # (N, rc, H, W) (N, (1-r)c, H, W)
        # print(f"Input feat_c0 shape: {feat_c0.shape}")
        # print(f"Input feat_c1 shape: {feat_c1.shape}")
        device = feat_c0.device  # 获取输入张量的设备
        feat_c0 = feat_c0.to(device)
        feat_c1 = feat_c1.to(device)
        feat_c0_selected, feat_c0_remaining = self.channel_enhance(feat_c0)
        feat_c1_selected, feat_c1_remaining = self.channel_enhance(feat_c1)
        # print(f"feat_c0_selected shape: {feat_c0_selected.shape}")
        # print(f"feat_c0_remaining shape: {feat_c0_remaining.shape}")
        # print(f"feat_c1_selected shape: {feat_c1_selected.shape}")
        # print(f"feat_c1_remaining shape: {feat_c1_remaining.shape}")

        return feat_c0_selected, feat_c0_remaining,feat_c1_selected, feat_c1_remaining