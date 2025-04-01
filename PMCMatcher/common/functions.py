import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.einops import rearrange, repeat
import cv2
import pydegensac
def mask_border(m, b, v):
    m[:, :b]=v
    m[:, :, :b]=v
    m[:, :, :, :b]=v
    m[:, :, :, :, :b]=v
    m[:, -b:0]=v
    m[:, :, -b:0]=v
    m[:, :, :, -b:0]=v
    m[:, :, :, :, -b:0]=v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd]= v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0-bd:] = v
        m[b_idx, :, w0-bd:] = v
        m[b_idx, :, :, h1-bd:] = v
        m[b_idx, :, :, h1-bd:] = v

@torch.no_grad()
def mutual_nearest_neighbor_match(
    scores,
    axes_lengths: dict,
    border: int,
    match_threshold: float,
    mask0=None,
    mask1=None,
    use_bins=False
):
    if use_bins:
        conf_matrix = scores[:, :-1, :-1].exp()
    else:
        conf_matrix = scores

    mask = conf_matrix > match_threshold
    mask = rearrange(
        mask,
        'B (H0 W0) (H1 W1) -> B H0 W0 H1 W1',
        **axes_lengths
    )
    if mask0 is not None:
        mask_border(mask, border, False)
    else:
        mask_border_with_padding(mask, border, False, mask0, mask1)

    mask = rearrange(
        mask,
        'B H0 W0 H1 W1 -> B (H0 W0) (H1 W1)',
        axes_lengths
    )

    max0 = conf_matrix.max(dim=2, keepdim=True)[0]
    max1 = conf_matrix.max(dim=1, keepdim=True)[0]

    mask = mask * (conf_matrix == max0) * (conf_matrix == max1)

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mconf = conf_matrix[b_ids, i_ids, j_ids]
    matches = torch.stack([b_ids, i_ids, j_ids]).T

    return matches[mconf != 0], mconf[mconf != 0]

def assignments_to_matches(assignments, use_bins=False):
    if use_bins:
        assignments = assignments[:, :-1, :-1]
    mask = assignments > 0
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mids = torch.stack([b_ids, i_ids, j_ids]).T

    return mids

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid

def grid_positions(h, w, device='cpu', matrix=False):
    rows = torch.arange(
        0, h, device=device
    ).view(-1, 1).float().repeat(1, w)

    cols = torch.arange(
        0, w, device=device
    ).view(1, -1).float().repeat(h, 1)

    if matrix:
        return torch.stack([cols, rows], dim=0)
    else:
        return torch.cat([cols.view(1, -1), rows.view(1, -1)], dim=0)

def upscale(coord, scaling_steps):
    for _ in range(scaling_steps):
        coord = coord * 2.0 + 0.5
    return coord

def downscale(coord, scaling_steps):
    for _ in range(scaling_steps):
        coord = (coord - .5) / 2.
    return coord

def normalize(coord, h, w):
    c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord.device).float()
    coord_norm = (coord - c) / c
    return coord_norm

def denormalize(coord_norm, h, w):
    c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord_norm.device)
    coord = coord_norm * c + c
    return coord

def ind2coord(ind, w):
    ind = ind.unsqueeze(-1)
    x = ind % w
    y = ind // w
    coord = torch.cat([x, y], -1).float()
    return coord

def test_ind2coord(assignments, w, use_bins=True):
    if use_bins:
        assignments = assignments[:, :-1, :-1]
    mask = assignments > 0
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mids = torch.stack([b_ids, i_ids, j_ids]).T

    mkpts0 = ind2coord(i_ids, w)
    mkpts1 = ind2coord(j_ids, w)

    return mkpts0, mkpts1

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid 

def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

def draw_match(match_mask,query,refer,x, y, homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts[::16]:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        xx_mean = np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0)
        yy_mean = np.mean((refer_pts - query_pts-y)[:, 1], axis=0)
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match2(match_mask,query,refer,x, y, homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        xx_mean = np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0)
        yy_mean = np.mean((refer_pts - query_pts-y)[:, 1], axis=0)
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match3(matches,query,refer,x, y, homo_filter=True,patch_size=8):
    matches = matches.detach().cpu().numpy()
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = []
    refer_pts = []
    for pt in matches[::16,:]:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def get_mkpts(matches, query, refer, patch_size=8):
    #grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    #_pts = grid[match_mask]
    #matches = match_mask.detach().cpu().numpy()
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + torch.div(pt[1], qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + torch.div(pt[2], rcols) * patch_size
        query_pts.append(torch.Tensor([x0, y0]))
        refer_pts.append(torch.Tensor([x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts

def draw_match_se(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=1e3):
    #query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    #refer = cv2.copyMakeBorder(refer, 128, 0, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-wq-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])
            x_mean.append(x1-wq-x0-x)
            y_mean.append(y1-y-y0)


    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if query_pts[i, 0] == refer_pts[i, 0]-512 and query_pts[i, 1] == refer_pts[i, 1]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    #cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    #cv2.putText(out_img, 'mean:' + str(x_means) + ' ' + str(y_means), (100,130), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def cal_reproj_dists(p1s, p2s, homography):
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist

def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index-1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs)

def eval_src_mma(mkpts0, mkpts1, query, refer, i_err, thres=[1,2,3,4,5,6,7,8,9,10],patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    if len(mkpts1) >3:
        H_pred, inliers = pydegensac.findHomography(mkpts0, mkpts1, ransac_thres)
        if len([mkpts0[x, :] for x in np.where(inliers>0)[0]]) > 0:
            mkpts0 = np.stack([mkpts0[x, :] for x in np.where(inliers>0)[0]])
            mkpts1= np.stack([mkpts1[x, :] for x in np.where(inliers>0)[0]])
    
    H_gt = np.array([[1,0,0],[0,1,0],[0,0,1]])

    if len(mkpts1) == 0 or H_gt is None:
        dist = np.array([float("inf")])
    else:
        dist = cal_reproj_dists(mkpts0, mkpts1, H_gt)
    for thr in thres:
        i_err[thr] += np.mean(dist <= thr)
    return i_err, 1

def eval_src_homography(mkpts0, mkpts1, query, refer,patch_size=8, ransac_thres=3):
    np.set_printoptions(suppress=True)
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    h,w = query.shape[:2]
    H_gt = np.array([[1,0,0],[0,1,0],[0,0,1]])
    if mkpts0.shape[0] < 4:
        return np.nan

    corners = np.array([[0,0,1],[0,w-1,1],[h-1,0,1],[h-1,w-1,1]])
    #H_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=ransac_thres)
    H_pred, inliers = pydegensac.findHomography(mkpts0, mkpts1, ransac_thres)

    if H_pred is None:
        return np.nan
    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    #print(H_pred)
    #print(corner_dist)
    return corner_dist

def batch_get_mkpts(matches, query, refer, patch_size=8):
    #grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    #_pts = grid[match_mask]
    #matches = match_mask.detach().cpu().numpy()
    
    #初始化部分
    #query_pts 和 refer_pts：用于存储匹配的查询图像和参考图像的关键点
    query_pts = []
    refer_pts = []
    #查询图像和参考图像特征图的宽度（输入为形状 [B, C, H, W] 的特征图，其中 W 是宽度）
    wq = query.shape[2]
    wr = refer.shape[2]
    #将图像特征划分为 patch_size 大小的小块，得到查询图像和参考图像中每行的小块数量
    qcols = wq // patch_size
    rcols = wr // patch_size
    """
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + torch.div(pt[1], qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + torch.div(pt[2], rcols) * patch_size
        query_pts.append(torch.Tensor([pt[0], x0, y0]))
        refer_pts.append(torch.Tensor([pt[0], x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    """
    #计算匹配点坐标
    x0 = patch_size/2 + (matches[:, 1] % qcols) * patch_size
    y0 = patch_size/2 + torch.div(matches[:, 1], qcols) * patch_size
    x1 = patch_size/2 + (matches[:, 2] % qcols) * patch_size
    y1 = patch_size/2 + torch.div(matches[:, 2], qcols) * patch_size
    #将计算出的批次索引、x 坐标和 y 坐标合并，生成 query_pts 和 refer_pts，形状为 [N, 3]，其中 N 是匹配点的数量
    query_pts = torch.cat((matches[:, 0].unsqueeze(1), x0.unsqueeze(1), y0.unsqueeze(1)), 1)
    refer_pts = torch.cat((matches[:, 0].unsqueeze(1), x1.unsqueeze(1), y1.unsqueeze(1)), 1)
    #检查 query_pts 是否为空。如果为空，则创建形状为 [0, 2] 的空张量
    if len(query_pts) < 0:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts






