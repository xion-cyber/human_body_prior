# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time     : 12/16/24 9:38 AM
# @Author   : YeYiqi
# @Email    : yeyiqi@stu.pku.edu.cn
# @File     : ik_mmpose_joints.py
# @desc     :


from typing import Union

import numpy as np
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from torch import nn

from human_body_prior.models.ik_engine import IK_Engine
from os import path as osp


class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int = 24,
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []
        self.n_joints = n_joints
        self.kpts_colors = np.array([Color('grey').rgb for _ in range(n_joints)])

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts': new_body.Jtr[:, :self.n_joints], 'body': new_body,
                'params': body_parms}

support_dir = '../support_data/dowloads'
vposer_expr_dir = osp.join(support_dir,'vposer_v2_05')
bm_fname =  osp.join(support_dir,'/home/yeyiqi/Documents/models/SMPLX/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz')
n_joints = 24
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loss = torch.nn.MSELoss(reduction='sum')
stepwise_weights = [
    {
        'data':10,
        'Poz_body':.01,
        'betas':.5
    }
]
optimizer_args = {
    'type':'LBFGS',
    'max_iter':300,
    'lr':1,
    'tolerance_change':1e-4,
    'history_size':200
}
ik_engine = IK_Engine(
    vposer_expr_dir=vposer_expr_dir,
    verbosity=2,
    display_rc=(2, 2),
    data_loss=data_loss,
    stepwise_weights=stepwise_weights,
    optimizer_args=optimizer_args
).to(comp_device)

source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints).to(comp_device)
sample_amass = {
    'poses':torch.zeros(1, 66).type(torch.float),
    'trans':torch.zeros(1, 3).type(torch.float)
}
target_bm = BodyModel(bm_fname)(**{
    'pose_body': torch.tensor(sample_amass['poses'][:,3:66]).type(torch.float),
    'root_orient': torch.tensor(sample_amass['poses'][:,:3]).type(torch.float),
    'trans': torch.tensor(sample_amass['trans']).type(torch.float)
})

source_pts = SourceKeyPoints(bm=bm_fname, n_joints=24).to(comp_device)
target_pts = target_bm.Jtr[:, :n_joints].detach().to(comp_device)
ik_res = ik_engine(source_pts, target_pts)

ik_res_detached = {k: v.detach() for k, v in ik_res.items()}

