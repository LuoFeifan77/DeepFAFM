import os
import scipy.io as sio
from utils.tensor_util import to_numpy
from pickle import TRUE
from re import I
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from networks.filter_network import Meyer


@MODEL_REGISTRY.register()
class FAPModel(BaseModel):  # Base model
    def __init__(self, opt):
        self.partial = opt.get('partial', False)
        self.sm_feat = opt.get('sm_feat', False)
        self.norm_filter = opt.get('norm_filter', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.using_nn_search = opt.get('using_nn_search', False)
        self.with_refine = opt.get('refine', -1)
        self.opt = opt
        if self.with_refine > 0:
            opt['is_train'] = True
        super(FAPModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        #1 feature extractor for mesh 
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        #2 get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        # 使用smooth features
        if self.sm_feat: 
            feat_x = torch.bmm(evecs_x, torch.bmm(evecs_trans_x, feat_x))
            feat_y = torch.bmm(evecs_y, torch.bmm(evecs_trans_y, feat_y))

        #3 calculate C_desc by desc preservation+spectral communication！
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        #4 calculate C_filter by MCFP
        basis_x, basis_y = self.networks['conv'](evals_x, evals_y)

        #5 compute filter by Jacobi 
        gs_x, gs_y = self.networks['comb'](basis_x, basis_y)

        # # 应该是对学出来的滤波器进行L2-normalization, 这个对近似等距有效果吗，能泛化能力吗
        if self.norm_filter:  # for isometric
            a1, _ = torch.max(gs_x, dim=-1)  # max values
            a2, _ = torch.max(gs_y, dim=-1)

            b1, _ = torch.min(gs_x, dim=-1)  # min values
            b2, _ = torch.min(gs_y, dim=-1)

            gs_x = (gs_x-b1.unsqueeze(-1))/(a1.unsqueeze(-1) - b1.unsqueeze(-1))  # 缩放到 [0,1] 
            gs_y = (gs_y-b2.unsqueeze(-1))/(a2.unsqueeze(-1) - b2.unsqueeze(-1)) 

        #6 non filtering for estimation  skip it
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))   # [1, K, K]
        Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))   # [1, K, K]

        # loss
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)  # for estimation, bij+orth

        #7 MSFOP  # 直接试一下滤波的结果，在非等距的数据上！
        Cxy_filtering = self.MCFP(gs_y, gs_x, Cxy_est)  # [1, K ,K]
        Cyx_filtering = self.MCFP(gs_x, gs_y, Cyx_est)  # [1, K, K]

        #8.1 frequency awareness couple loss
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_filtering)  # key ideas

        #8.2 calculate loss
        if not self.partial and not self.with_refine:
            if Cyx != None:
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_filtering)  #添加双向

        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['L'], data_y['L']  # 这里的loss不对！
            verts_x, verts_y = data_x['verts'], data_y['verts']

            # if using_dirichlet_loss: 
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)  # 只用单向
            self.loss_metrics['l_d'] += self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) 
                                        

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)

        #1 feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces')) #[1, Nx, D]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces')) #[1, Ny, D]

        #2 get spectral operators
        evals_x = data_x['evals']  #[1, K]
        evecs_x = data_x['evecs'].squeeze() 
        evecs_trans_x = data_x['evecs_trans'].squeeze() # [1, K, Nx]
       
        evals_y = (data_y['evals']) 
        evecs_y = (data_y['evecs']).squeeze() # [1, K, Nx]
        evecs_trans_y = (data_y['evecs_trans']).squeeze()  # [1, K, Ny]


        # smooth features or not
        if self.sm_feat: 
            feat_x = torch.bmm(evecs_x.unsqueeze(0), torch.bmm(evecs_trans_x.unsqueeze(0), feat_x))
            feat_y = torch.bmm(evecs_y.unsqueeze(0), torch.bmm(evecs_trans_y.unsqueeze(0), feat_y))

        basis_x, basis_y = self.networks['conv'](evals_x, evals_y)
        gs_x, gs_y = self.networks['comb'](basis_x, basis_y) # [1, 6, 200]

        # L1-normalization or not
        if self.norm_filter:
            a1, _ = torch.max(gs_x, dim=-1)  # max values
            a2, _ = torch.max(gs_y, dim=-1)

            b1, _ = torch.min(gs_x, dim=-1)  # min values
            b2, _ = torch.min(gs_y, dim=-1)

            gs_x = (gs_x-b1.unsqueeze(-1))/(a1.unsqueeze(-1) - b1.unsqueeze(-1))  # 缩放到 [0,1] 
            gs_y = (gs_y-b2.unsqueeze(-1))/(a2.unsqueeze(-1) - b2.unsqueeze(-1)) 

        feat_x = F.normalize(feat_x, dim=-1, p=2)  # 仅仅对非等距使用
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        p2p = nn_query(feat_x, feat_y).squeeze() # nearest neighbour query
        # p2pxy = nn_query(feat_y, feat_x).squeeze() # nearest neighbour query

        Cxy_est = evecs_trans_y @ evecs_x[p2p]   #[K, K]

        # 换一种思路
        if self.non_isometric:   # non_isometric matching 迭代一次即可
            iter_num = 1  # 1
        else :                   # near_isometric matching 迭代5次结果更好
            iter_num = 5  # 3~5    

        # using learned filter functions to refine
        for _ in range(iter_num):  # 迭代几次
            # convert functional map to point-to-point map by nnsearch
            Cxy_est = self.MCFP(gs_y, gs_x, Cxy_est.unsqueeze(0)).squeeze()  # [K ,K], using learned filters
            p2p = fmap2pointmap(Cxy_est, evecs_x, evecs_y) # 
            Cxy_est = evecs_trans_y @ evecs_x[p2p]  #[K, K] 

        Pyx = evecs_y @ Cxy_est @ evecs_trans_x  #置换矩阵

        # finish record
        timer.record()

        # resume previous network state dict, restar
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)

        # return
        return p2p, Pyx, Cxy_est, gs_x, gs_y


    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    
    def MCFP(self, gs_x, gs_y, Cyx):
    # input:
    #   gs_x/y: [1, Nf, Kx/Ky]
    #   Cyx : [1, K, K]

        C_new = torch.zeros_like(Cyx)  # 声明成全零元素
        gs_y2 = torch.sum(gs_y**2, dim=1)  # 

        # MWP filters
        Nf = gs_x.size(1)  # 
        for s in range(Nf):
            C_new = C_new + gs_x[:,s,:].t()*Cyx*gs_y[:,s,:]

        C_new=C_new*(1/gs_y2)   #
    
        return C_new  # [1, K, K]


    def refine(self, data):  # optimal parameters
        # pass 
        # refinement has been closed during testing
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False


    @torch.no_grad()  # 
    def validation(self, dataloader, tb_logger, update=True): 
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(FAPModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True

