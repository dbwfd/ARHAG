import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HeterogeneousGnn(nn.Module):
    def __init__(self, config, att_index):
        super(HeterogeneousGnn, self).__init__()
        self.is_layernorm = config.is_layernorm
        self.is_prenorm = config.is_prenorm
        self.num_attclass = config.num_attclass
        self.num_att = config.num_attribute
        self.input_dim = config.tf_common_dim
        self.hdim = config.tf_common_dim
        self.att_index = att_index
        self.fl_type = config.fl_type
        self.a2h_type = config.a2h_type
        self.h2a_type = config.h2a_type
        self.is_attention_norm = config.is_attention_norm

        self.is_a2h_mask1 = config.is_a2h_mask1
        self.is_h2a_mask1 = config.is_h2a_mask1

        self.is_a2h_mask2 = config.is_a2h_mask2
        self.is_h2a_mask2 = config.is_h2a_mask2

        self.avgmask = nn.Parameter(torch.zeros(self.num_attclass, self.num_att), requires_grad=False)

        for i in range(self.num_attclass):
            self.avgmask[i][att_index == i] = 1 / len(att_index[att_index == i])
        self.adj_ha = nn.Parameter(torch.cat((self.avgmask.detach().clone(), torch.eye(self.num_attclass)), dim=1),
                                   requires_grad=False)
        self.adj_ah = nn.Parameter(torch.cat((self.avgmask.detach().clone().T, torch.eye(self.num_att)), dim=1),
                                   requires_grad=False)

        self.layer1 = HeterogeneousGnnLayer(input_dim=self.input_dim, h_dim=self.hdim, avgmask=self.avgmask,
                                            adj_ha=self.adj_ha, adj_ah=self.adj_ah, a2h_type=self.fl_type,
                                            h2a_type=self.h2a_type, is_layernorm=self.is_layernorm,
                                            is_prenorm=self.is_prenorm, is_attention_norm=self.is_attention_norm,
                                            is_a2h_mask=self.is_a2h_mask1, is_h2a_mask=self.is_h2a_mask1)
        self.layer2 = HeterogeneousGnnLayer(input_dim=self.input_dim, h_dim=self.hdim, avgmask=self.avgmask,
                                            adj_ha=self.adj_ha, adj_ah=self.adj_ah, a2h_type=self.a2h_type,
                                            h2a_type=self.h2a_type, is_layernorm=self.is_layernorm,
                                            is_prenorm=self.is_prenorm, is_attention_norm=self.is_attention_norm,
                                            is_a2h_mask=self.is_a2h_mask2, is_h2a_mask=self.is_h2a_mask2)

    def forward(self, att_vf, att_confidence=None):
        mask_avg, mask_ha, mask_ah = None, None, None
        if att_confidence is not None:
            mask_avg, mask_ha, mask_ah = self.get_masks(att_confidence)

        out_hl, out_att = self.layer1(att_vf, hl_vf=None, mask_avg=mask_avg, mask_ha=mask_ha, mask_ah=mask_ah)
        out_hl, out_att = self.layer2(out_att, out_hl, mask_avg=mask_avg, mask_ha=mask_ha, mask_ah=mask_ah)

        return out_att

    def get_masks(self, att_confidence):
        bsz,_=att_confidence.shape
        threshold = torch.mean(att_confidence,dim=-1,keepdim=True)

        att_confidence_positive = att_confidence > threshold
        att_confidence_negative = att_confidence <= threshold

        att_confidence_positive = torch.unsqueeze(att_confidence_positive,dim=1)
        att_confidence_negative = torch.unsqueeze(att_confidence_negative, dim=1)
        att_confidence_positive = att_confidence_positive.repeat(1,self.num_attclass, 1)
        att_confidence_negative = att_confidence_negative.repeat(1,self.num_attclass, 1)

        avg_confidence_mask_positive = torch.where(att_confidence_positive, self.avgmask, 0)
        avg_confidence_mask_negative = torch.where(att_confidence_negative, self.avgmask, 0)

        mask_positive = avg_confidence_mask_positive > 0


        length_positive = torch.unsqueeze(torch.count_nonzero(avg_confidence_mask_positive,dim=-1),dim=-1)



        avg_confidence_mask_positive=torch.where(mask_positive, 1/length_positive, 0)

        avg_confidence_mask_positive=torch.nan_to_num(avg_confidence_mask_positive,nan=1e-6,posinf=1e-6,neginf=1e-6)





        mask_ha = torch.cat((avg_confidence_mask_positive ,
                             torch.unsqueeze(torch.eye(self.num_attclass).cuda(),dim=0).repeat(bsz,1,1)), dim=-1)
        mask_ah = torch.cat((torch.permute(avg_confidence_mask_negative,(0,2,1)),
                             torch.unsqueeze(torch.eye(self.num_att).cuda(), dim=0).repeat(bsz, 1, 1)), dim=-1)

        return avg_confidence_mask_positive, mask_ha, mask_ah


class HeterogeneousGnnLayer(nn.Module):
    def __init__(self, input_dim, h_dim, avgmask, adj_ha, adj_ah, a2h_type='avg', h2a_type='attention',
                 is_layernorm=True, is_prenorm=True, is_attention_norm=True, is_a2h_mask=False, is_h2a_mask=False):
        self.is_prenorm = is_prenorm
        self.is_layernorm = is_layernorm
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.is_attention_norm = is_attention_norm
        # for a2h
        super(HeterogeneousGnnLayer, self).__init__()
        self.W_a2h = nn.Parameter(nn.init.normal_(
            torch.empty(input_dim, h_dim)), requires_grad=True)
        # for h2h
        self.W_h2h = nn.Parameter(nn.init.normal_(
            torch.empty(h_dim, h_dim)), requires_grad=True)
        # for h2a
        self.W_h2a = nn.Parameter(nn.init.normal_(
            torch.empty(h_dim, input_dim)), requires_grad=True)

        self.type_a2h = a2h_type
        self.type_h2a = h2a_type

        self.is_a2h_mask = is_a2h_mask
        self.is_h2a_mask = is_h2a_mask
        #
        self.avgmask = avgmask
        self.adj_ha = adj_ha
        self.adj_ah = adj_ah
        # for layernorm
        if is_prenorm:
            self.layernorm1 = nn.LayerNorm(input_dim)
            self.layernorm2 = nn.LayerNorm(h_dim)
            self.layernorm3 = nn.LayerNorm(h_dim)
        else:
            self.layernorm1 = nn.LayerNorm(h_dim)
            self.layernorm2 = nn.LayerNorm(h_dim)
            self.layernorm3 = nn.LayerNorm(input_dim)

        # for dropout
        # self.dropout=dropout

    def forward(self, att_vf, hl_vf=None, mask_avg=None, mask_ha=None, mask_ah=None):
        """
        :param att_vf: att based visual feature: batch * num_att * feature_dim
        :param att_index: attribute classfication: batch * num_attclass * h_dim
        :param hl_vf: hign-level attribute visual feature, for the first layer = None, else the previous layer hl_vf
        :return: (updated) high-level attribute visual feature:batch * num_attclass * h_dim, reasoned_att_vf: batch * num_att * input_dim
        """

        hl_vf = self.a2h_vf_gnn(att_vf, hl_vf, mask_avg=mask_avg, mask_ha=mask_ha)
        updated_hlvf = self.h2h_vf_gnn(hl_vf)
        updated_attvf = self.h2a_vf_gnn(updated_hlvf, att_vf, mask_avg=mask_avg, mask_ah=mask_ah)

        return updated_hlvf, updated_attvf

    def a2h_vf_gnn(self, att_vf, hl_vf=None, mask_avg=None, mask_ha=None):
        """

        :param att_vf: att based visual feature: batch * num_att * feature_dim
        :param hl_vf: hign-level attribute visual feature, for the first layer = None, else the previous layer hl_vf
        :return: hl_vf: hl_vf constructed or updated by att_vf
        """
        if self.is_prenorm and self.is_layernorm:
            att_vf = self.layernorm1(att_vf)
        if self.type_a2h == 'avg':
            if self.is_a2h_mask:
                hl_vf = torch.matmul(mask_avg, att_vf)
            else:
                hl_vf = torch.matmul(self.avgmask, att_vf)
        elif self.type_a2h == 'wavg':
            hl_vf = torch.matmul(att_vf, self.W_a2h)
            if self.is_a2h_mask:
                hl_vf = torch.matmul(mask_avg, hl_vf)
            else:
                hl_vf = torch.matmul(self.avgmask, hl_vf)
        elif self.type_a2h == 'attention':
            hidden_vf = torch.matmul(att_vf, self.W_a2h)
            hidden_vf = torch.cat((hidden_vf, hl_vf), dim=1)
            s = torch.matmul(hl_vf, torch.permute(hidden_vf, (0, 2, 1)))
            if self.is_attention_norm:
                s = s / torch.sqrt(torch.tensor(self.h_dim).cuda())
            zero_vec = -9e15 * torch.ones_like(s)
            if self.is_a2h_mask:
                s = torch.where(mask_ha > 0, s, zero_vec)
            else:
                s = torch.where(self.adj_ha > 0, s, zero_vec)
            a = F.softmax(s, dim=-1)
            hl_vf = torch.matmul(a, hidden_vf)
        elif self.type_a2h == 'attention_add':
            # unfinished
            hidden_vf = torch.matmul(att_vf, self.W_a2h)
            s = torch.matmul(hl_vf, torch.permute(hidden_vf, (0, 2, 1)))
            if self.is_attention_norm:
                s = s / torch.sqrt(torch.tensor(self.h_dim).cuda())
            zero_vec = -9e15 * torch.ones_like(s)
            if self.is_a2h_mask:
                s = torch.where(mask_avg > 0, s, zero_vec)
            else:
                s = torch.where(self.avgmask > 0, s, zero_vec)

            a = F.softmax(s, dim=-1)
            hl_vf = torch.matmul(a, hidden_vf) + hl_vf

        if (not self.is_prenorm) and self.is_layernorm:
            hl_vf = self.layernorm1(hl_vf)
        return hl_vf

    def h2h_vf_gnn(self, hl_vf):
        if self.is_prenorm and self.is_layernorm:
            hl_vf = self.layernorm2(hl_vf)
        hide = torch.matmul(hl_vf, self.W_h2h)
        s = torch.matmul(hide, torch.permute(hide, (0, 2, 1)))
        if self.is_attention_norm:
            s = s / torch.sqrt(torch.tensor(self.h_dim).cuda())
        a = F.softmax(s, dim=-1)
        updated_hlvf = torch.matmul(a, hl_vf)

        if (not self.is_prenorm) and self.is_layernorm:
            updated_hlvf = self.layernorm1(updated_hlvf)

        return updated_hlvf

    def h2a_vf_gnn(self, updated_hlvf, att_vf, mask_avg=None, mask_ah=None):
        if self.is_prenorm and self.is_layernorm:
            updated_hlvf = self.layernorm3(updated_hlvf)
        if self.type_h2a == 'attention':
            hidden_vf = torch.matmul(updated_hlvf, self.W_h2a)
            hidden_vf = torch.cat((hidden_vf, att_vf), dim=1)
            s = torch.matmul(att_vf, torch.permute(hidden_vf, (0, 2, 1)))
            if self.is_attention_norm:
                s = s / torch.sqrt(torch.tensor(self.input_dim).cuda())
            zero_vec = -9e15 * torch.ones_like(s)
            if self.is_h2a_mask:
                s = torch.where(mask_ah > 0, s, zero_vec)
            else:
                s = torch.where(self.adj_ah > 0, s, zero_vec)

            a = F.softmax(s, dim=-1)
            updated_avf = torch.matmul(a, hidden_vf)
        elif self.type_h2a == 'attention_add':
            hidden_vf = torch.matmul(updated_hlvf, self.W_h2a)
            s = torch.matmul(hidden_vf, torch.permute(att_vf, (0, 2, 1)))
            if self.is_attention_norm:
                s = s / torch.sqrt(torch.tensor(self.input_dim).cuda())
            zero_vec = -9e15 * torch.ones_like(s)
            if self.is_h2a_mask:
                s = torch.where(mask_avg > 0, s, zero_vec)
            else:
                s = torch.where(self.avgmask > 0, s, zero_vec)

            a = F.softmax(s, dim=-1)
            updated_avf = torch.matmul(torch.permute(a, (0, 2, 1)), hidden_vf) + att_vf

        if (not self.is_prenorm) and self.is_layernorm:
            updated_avf = self.layernorm1(updated_avf)

        return updated_avf
