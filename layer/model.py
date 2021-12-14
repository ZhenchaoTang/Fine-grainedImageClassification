from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from layer import resnet
from layer.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), 200)
        self.partcls_net = nn.Linear(512 * 4, 200)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda().long()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feats=part_feature
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, top_n_cdds, part_feats]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)

def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size

import dgl
import networkx as nx
import dgl.function as fn

class CNNtoGraph(nn.Module):
    def __init__(self, hidden_dim=1024, cls_dim=200, pretrain=True, freeze=True):
        super().__init__()
        self.pretrain=pretrain
        self.freeze=freeze

        self.net = attention_net(topN=PROPOSAL_NUM)
        if self.pretrain:
            pretrain_model=r"./checkpoints/model.ckpt"
            ckpt = torch.load(pretrain_model)
            self.net.load_state_dict(ckpt['net_state_dict'])
        if self.freeze:
            for param in self.net.parameters():
                param.requires_grad=False

        self.fc=nn.Linear(2048*2,hidden_dim)
        self.classify=nn.Linear(hidden_dim,cls_dim)

    def construct_graph(self, top_n_cdds:"ndarry(N,topN6,6)", part_feats:"tensor(N,topN6,2048)", node_num=6):
        batch_size=top_n_cdds.shape[0]
        graph_batch=[]
        for sample_id in range(batch_size):
            nx_g = nx.complete_graph(node_num)
            graph = dgl.from_networkx(nx_g).to("cuda:0")
            graph.ndata["feature"]=part_feats[sample_id]
            cdds=torch.from_numpy(top_n_cdds[sample_id]).float().cuda()
            graph.ndata["location"]=cdds[:,1:5]
            graph_batch.append(graph)

        return dgl.batch(graph_batch)

    def apply_edges(self,edges):
        loc_u_x=edges.src["location"][:,1]+(edges.src["location"][:,3]-edges.src["location"][:,1])/2
        loc_u_y=edges.src["location"][:,0]+(edges.src["location"][:,2]-edges.src["location"][:,0])/2

        loc_v_x=edges.dst["location"][:,1]+(edges.dst["location"][:,3]-edges.dst["location"][:,1])/2
        loc_v_y=edges.dst["location"][:,0]+(edges.dst["location"][:,2]-edges.dst["location"][:,0])/2

        weight = torch.exp(
            -0.015 * (
                torch.sqrt(
                    (loc_u_x - loc_v_x) ** 2 + (loc_u_y - loc_v_y) ** 2
                )
            )
        ).view(-1, 6)

        mean_weight=torch.mean(weight.view(-1,30),-1).view(-1,1)
        weight = 24 * (weight.view(-1, 30) - mean_weight).clamp(min=0)

        return {"weight":weight.view(-1,1)}

    def forward(self, x):
        raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, top_n_cdds, part_feats = self.net(x)
        graph=self.construct_graph(top_n_cdds,part_feats)

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            graph.update_all(fn.u_mul_e("feature","weight","message"),
                             fn.mean("message","hidden"))
            final_ft=self.fc(torch.cat((graph.ndata["hidden"],graph.ndata["feature"]),-1))
            graph.ndata["nodeft"]=final_ft
            g_ft=dgl.mean_nodes(graph,"nodeft")
            return [self.classify(g_ft), raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, top_n_cdds, part_feats]