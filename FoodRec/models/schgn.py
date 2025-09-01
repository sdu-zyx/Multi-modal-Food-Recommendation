import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geometric
import numpy as np
from tqdm import tqdm

from FoodRec.common.abstract_recommender import GeneralRecommender
from FoodRec.common.module import Encoder, LayerNorm


def l2_loss(t):
    return torch.sum(t ** 2)


def truncated_normal_(tensor, mean=0, std=0.01):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class GraphConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = geometric.nn.GCNConv(in_channel, out_channel)
        truncated_normal_(self.conv1.lin.weight, std=np.sqrt(2.0 / (self.in_channel + self.out_channel)))
        truncated_normal_(self.conv1.bias, std=np.sqrt(2.0 / (self.in_channel + self.out_channel)))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        return x


class SCHGN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SCHGN, self).__init__(config, dataset)
        self.device = config['device']
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_cold = dataset.cold_num
        self.n_health = dataset.num_calories_level
        self.n_ingredients = dataset.num_ingredients
        self.img_size = dataset.image_size

        self.ingre_encoder = Encoder(n_layers=config['num_hidden_layers'],
                                     n_heads=config['num_attention_heads'],
                                     hidden_size=config['embedding_size'],
                                     inner_size=config['inner_size'],
                                     hidden_dropout_prob=config['hidden_dropout_prob'],
                                     attn_dropout_prob=config['attention_probs_dropout_prob'],
                                     hidden_act=config['hidden_act'],
                                     layer_norm_eps=1e-12)
        self.apply(self.init_weights)

        self.g2i_edges, self.i2u_edges = self.load_graph(self.dataset)
        input_channel = 64
        out_channel = 64
        self.new_gcn = GraphConv(input_channel, out_channel)

        self.emb_size = config['embedding_size']
        self.regs = config['regs']
        self.reg_image = config['reg_image']
        self.reg_w = config['reg_w']
        self.reg_g = config['reg_g']
        self.reg_health = config['reg_health']
        self.ssl = config['ssl']

        self.user_embed = torch.nn.Parameter(torch.FloatTensor(self.n_users, self.emb_size), requires_grad=True)
        self.item_embed = torch.nn.Parameter(torch.FloatTensor(self.n_items, self.emb_size),
                                             requires_grad=True)
        self.ingre_embed_first = torch.nn.Parameter(torch.FloatTensor(self.n_ingredients, self.emb_size),
                                                    requires_grad=True)

        self.ingre_embed_second = torch.nn.Parameter(torch.zeros(1, self.emb_size, dtype=self.ingre_embed_first.dtype),
                                                     requires_grad=False)
        self.ingre_embed_mask = torch.nn.Parameter(torch.FloatTensor(1, self.emb_size), requires_grad=True)
        self.health_embed = torch.nn.Parameter(torch.FloatTensor(self.n_health, self.emb_size), requires_grad=True)

        self.img_trans = nn.Linear(self.img_size, self.emb_size)
        truncated_normal_(self.img_trans.weight, std=np.sqrt(2.0 / (self.img_size + self.emb_size)))
        truncated_normal_(self.img_trans.bias, std=np.sqrt(2.0 / (self.img_size + self.emb_size)))

        # parameters for ingredient level attention
        self.W_att_ingre = torch.nn.Linear(self.emb_size * 3, self.emb_size)
        truncated_normal_(self.W_att_ingre.weight, std=np.sqrt(2.0 / (self.emb_size * 4)))
        truncated_normal_(self.W_att_ingre.bias, std=np.sqrt(2.0 / (self.emb_size + self.emb_size)))
        self.h_att_ingre = torch.nn.Linear(self.emb_size, 1, bias=False)
        nn.init.ones_(self.h_att_ingre.weight)

        # parameters for component level attention
        self.W_att_comp = torch.nn.Linear(self.emb_size * 2, self.emb_size)
        truncated_normal_(self.W_att_comp.weight, std=np.sqrt(2.0 / (self.emb_size * 3)))
        truncated_normal_(self.W_att_comp.bias, std=np.sqrt(2.0 / (self.emb_size + self.emb_size)))
        self.h_att_comp = torch.nn.Linear(self.emb_size, 1, bias=False)
        nn.init.ones_(self.h_att_comp.weight)

        self.W_concat = nn.Linear(self.emb_size * 3, self.emb_size)
        truncated_normal_(self.W_concat.weight, std=np.sqrt(2.0 / (self.emb_size * 4)))
        truncated_normal_(self.W_concat.bias, std=np.sqrt(2.0 / (self.emb_size + self.emb_size)))
        self.output_mlp = nn.Linear(self.emb_size, 1, bias=False)
        truncated_normal_(self.output_mlp.weight, std=np.sqrt(2.0 / (self.emb_size * 2)))

        self.mip_norm = nn.Linear(self.emb_size, self.emb_size)
        self.criterion = nn.BCELoss(reduction='none')

        self._init_weight()

    def _init_weight(self):
        truncated_normal_(self.user_embed, std=0.01)
        truncated_normal_(self.item_embed, std=0.01)
        truncated_normal_(self.ingre_embed_first, std=0.01)
        truncated_normal_(self.ingre_embed_mask, std=0.01)
        truncated_normal_(self.health_embed, std=0.01)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            truncated_normal_(module.weight, std=0.01)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_graph(self, dataset):
        i2u_index, g2i_index = [], []
        for uid, iid in tqdm(dataset.uRecipe_triples, ascii=True):
            i2u_index.append([iid + self.n_users, uid])

        for h_id, t_id in tqdm(dataset.rIngre_triples, ascii=True):
            g2i_index.append([t_id + self.n_users + self.n_items, h_id + self.n_users])

        for h_id, t_id in tqdm(dataset.rCalories_triples, ascii=True):
            g2i_index.append([t_id + self.n_users + self.n_items + self.n_ingredients, h_id + self.n_users])
        g2i_edges = torch.tensor(np.array(i2u_index)).to(self.device, dtype=torch.long)
        i2u_edges = torch.tensor(np.array(g2i_index)).to(self.device, dtype=torch.long)
        return g2i_edges, i2u_edges

    def sequence_mask(self, lengths, max_len):
        row = torch.arange(0, max_len, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row < matrix
        return mask.type(torch.float)

    def attention_ingredient_level(self, ingre_emb, u_emb, img_emb, ingre_num):
        b = ingre_emb.shape[0]
        n = ingre_emb.shape[1]

        expand_u_emb = u_emb.unsqueeze(1)

        tile_u_emb = expand_u_emb.repeat(1, n, 1)
        expand_img = img_emb.unsqueeze(1)
        tile_img = expand_img.repeat(1, n, 1)

        concat_v = torch.cat([ingre_emb, tile_u_emb, tile_img], dim=2)

        MLP_output = torch.tanh(self.W_att_ingre(concat_v))

        A_ = self.h_att_ingre(MLP_output).squeeze()
        smooth = -1e12

        mask_mat = self.sequence_mask(ingre_num, max_len=n)
        mask_mat = torch.ones_like(mask_mat) - mask_mat
        mask_mat = mask_mat * smooth

        A = F.softmax(A_ + mask_mat, dim=1)

        A = A.unsqueeze(2)

        return torch.sum(A * ingre_emb, dim=1)

    def attention_id_ingre_image(self, u_emb, i_emb, ingre_att_emb, img_emb, hl_emb):
        b = u_emb.shape[0]

        cp1 = torch.cat([u_emb, i_emb], dim=1)
        cp2 = torch.cat([u_emb, ingre_att_emb], dim=1)
        cp3 = torch.cat([u_emb, img_emb], dim=1)
        cp4 = torch.cat([u_emb, hl_emb], dim=1)

        cp = torch.cat([cp1, cp2, cp3, cp4], dim=0)

        c_hidden_output = torch.tanh(self.W_att_comp(cp))

        c_mlp_output = self.h_att_comp(c_hidden_output).view(b, -1)

        B = F.softmax(c_mlp_output, dim=1).unsqueeze(2)
        ce1 = i_emb.unsqueeze(1)  # [b, 1, e]
        ce2 = ingre_att_emb.unsqueeze(1)  # [b, 1, e]
        ce3 = img_emb.unsqueeze(1)  # [b, 1, e]
        ce4 = hl_emb.unsqueeze(1)  # [b, 1, e]
        ce = torch.cat([ce1, ce2, ce3, ce4], dim=1)  # [b, 4, e]
        return torch.sum(B * ce, dim=1)  # [b, e]

    def masked_ingre_prediction(self, ingre_emb, target_emb):
        ingre_emb = self.mip_norm(ingre_emb.view([-1, self.emb_size]))
        target_emb = target_emb.view([-1, self.emb_size])
        score = torch.mul(ingre_emb, target_emb)
        return torch.sigmoid(torch.sum(score, -1))

    def compute_ssl_loss(self, ingre_embedding, ingre_embedding_gcn, masked_ingre_seq, pos_ingre, neg_ingre):

        ingre_emb = ingre_embedding_gcn[masked_ingre_seq]

        seq_mask = (masked_ingre_seq == self.n_ingredients).float() * -1e8
        seq_mask = torch.unsqueeze(torch.unsqueeze(seq_mask, 1), 1)
        encoded_embs = self.ingre_encoder(ingre_emb, seq_mask, output_all_encoded_layers=True)
        new_ingre_emb = encoded_embs[-1]

        pos_ingre_emb = ingre_embedding[pos_ingre]
        neg_ingre_emb = ingre_embedding[neg_ingre]
        pos_score = self.masked_ingre_prediction(new_ingre_emb, pos_ingre_emb)
        neg_score = self.masked_ingre_prediction(new_ingre_emb, neg_ingre_emb)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_ingre_seq == self.n_ingredients + 1).float()
        num_tokens = torch.sum(mip_mask)
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())
        return mip_loss

    def compute_score(self, user, item, ingre, ingre_num, img, hl, is_training, g2i_edges, i2u_edges, ingre_embedding):

        u_emb = self.user_embed[user]
        i_emb = self.item_embed[item]
        ingre_emb = ingre_embedding[ingre]
        hl_emb = self.health_embed[hl]
        img_emb = self.img_trans(img.to(torch.float32))
        edge_index = torch.cat([g2i_edges, i2u_edges], dim=0)

        x = torch.cat([
            self.user_embed, self.item_embed,
            self.ingre_embed_first, self.health_embed
        ], dim=0)
        gcn_emb = self.new_gcn(x, edge_index.t().contiguous())  # [n_nodes, emb_size]
        user_embed_gcn, item_embed_gcn, ingre_embed_gcn, hl_emb_gcn = torch.split(
            gcn_emb, [self.n_users, self.n_items, self.n_ingredients, self.n_health], dim=0
        )
        ingre_embedding_gcn = torch.cat([ingre_embed_gcn, self.ingre_embed_second, self.ingre_embed_mask], dim=0)

        u_gcn_emb = user_embed_gcn[user]
        i_gcn_emb = item_embed_gcn[item]
        ingre_gcn_emb = ingre_embedding_gcn[ingre]
        hl_gcn_emb = hl_emb_gcn[hl]

        u_emb_final = u_emb + u_gcn_emb
        i_emb_final = i_emb + i_gcn_emb
        ingre_emb_final = ingre_emb + ingre_gcn_emb
        hl_emb_final = hl_emb + hl_gcn_emb

        ingre_att_emb = self.attention_ingredient_level(ingre_emb_final, u_emb_final, img_emb, ingre_num)
        item_att_emb = self.attention_id_ingre_image(u_emb_final, i_emb_final, ingre_att_emb, img_emb, hl_emb_final)
        ui_concat_emb = torch.cat([u_emb_final, item_att_emb, u_emb_final * item_att_emb], dim=1)
        hidden_input = self.W_concat(ui_concat_emb)
        MLP_ouput = F.relu(F.dropout(hidden_input, p=0.5, training=is_training))
        score = self.output_mlp(MLP_ouput).squeeze()

        return score, u_emb, i_emb, ingre_emb, hl_emb, ingre_embedding_gcn, item_att_emb

    def calculate_loss(self, batch_data):
        user = batch_data['u_id']
        pos_item, pos_ingre, pos_ingre_num, pos_img \
            = batch_data['pos_i_id'], batch_data['pos_ingre_code'], batch_data['pos_ingre_num'], batch_data['pos_img']
        neg_item, neg_ingre, neg_ingre_num, neg_img \
            = batch_data['neg_i_id'], batch_data['neg_ingre_code'], batch_data['neg_ingre_num'], batch_data['neg_img']

        pos_hl, neg_hl = batch_data['pos_cl'].long(), batch_data['neg_cl'].long()
        masked_ingre_seq, pos_ingre_seq, neg_ingre_seq \
            = batch_data['masked_ingre_seq'], batch_data['pos_ingre_seq'], batch_data['neg_ingre_seq']

        ingre_embedding = torch.cat([self.ingre_embed_first, self.ingre_embed_second, self.ingre_embed_mask], dim=0)
        pos_scores, user_emb, pos_item_emb, pos_ingre_emb, pos_hl_emb, ingre_embedding_g, _ = self.compute_score(user,
                                                                                                                 pos_item,
                                                                                                                 pos_ingre,
                                                                                                                 pos_ingre_num,
                                                                                                                 pos_img,
                                                                                                                 pos_hl,
                                                                                                                 True,
                                                                                                                 self.g2i_edges,
                                                                                                                 self.i2u_edges,
                                                                                                                 ingre_embedding)
        neg_scores, user_emb, neg_item_emb, neg_ingre_emb, neg_hl_emb, _, _ = self.compute_score(user, neg_item,
                                                                                                 neg_ingre,
                                                                                                 neg_ingre_num,
                                                                                                 neg_img, neg_hl, True,
                                                                                                 self.g2i_edges,
                                                                                                 self.i2u_edges,
                                                                                                 ingre_embedding)

        ssl_loss = self.ssl * self.compute_ssl_loss(ingre_embedding, ingre_embedding_g, masked_ingre_seq, pos_ingre_seq,
                                                    neg_ingre_seq)

        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.sum(bpr_loss)

        reg_loss = self.regs * (l2_loss(user_emb) + l2_loss(pos_item_emb) + l2_loss(neg_item_emb) + l2_loss(
            pos_ingre_emb) + l2_loss(neg_ingre_emb))
        reg_loss += self.reg_health * (l2_loss(pos_hl_emb) + l2_loss(neg_hl_emb))
        reg_loss += self.reg_image * (l2_loss(self.img_trans.weight))
        reg_loss += self.reg_w * (l2_loss(self.W_concat.weight) + l2_loss(
            self.output_mlp.weight))
        reg_loss += self.reg_g * (l2_loss(self.new_gcn.conv1.lin.weight))

        return bpr_loss, reg_loss, ssl_loss

    def full_sort_predict(self, batch_data):
        user = batch_data['u_id'].repeat(self.dataset.n_items)
        itemsList = list(range(self.dataset.n_items))
        item = torch.tensor(np.array(itemsList), dtype=torch.long).to(self.device)
        ingreList = []
        ingreNum = []
        imageList = []
        healthList = []

        for i in itemsList:
            ingreList.append(self.dataset.ingredientCodeDict[i])
            ingreNum.append(self.dataset.ingredientNum[i])
            imageList.append(self.dataset.embImage[i])
            healthList.append(self.dataset.cal_level[i])

        img = torch.tensor(np.array(imageList), dtype=torch.float32).to(self.device)
        ingre_num = torch.tensor(np.array(ingreNum), dtype=torch.long).to(self.device)
        ingre = torch.tensor(np.array(ingreList), dtype=torch.long).to(self.device)
        hl = torch.tensor(np.array(healthList), dtype=torch.long).to(self.device)

        ingre_embedding = torch.cat([self.ingre_embed_first, self.ingre_embed_second], dim=0)
        predictions, user_emb, item_emb, ingre_emb, hl_emb, _, r_final_emb = self.compute_score(user, item, ingre,
                                                                                                ingre_num, img, hl,
                                                                                                False, self.g2i_edges,
                                                                                                self.i2u_edges,
                                                                                                ingre_embedding)

        return predictions

    def sample_sort_predict(self, batch_data):
        user = batch_data['u_id']
        pos_item, pos_ingre, pos_ingre_num, pos_img \
            = batch_data['pos_i_id'], batch_data['pos_ingre_code'], batch_data['pos_ingre_num'], batch_data['pos_img']
        neg_item, neg_ingre, neg_ingre_num, neg_img \
            = batch_data['neg_i_id'], batch_data['neg_ingre_code'], batch_data['neg_ingre_num'], batch_data['neg_img']

        pos_hl, neg_hl = batch_data['pos_cl'].long(), batch_data['neg_cl'].long()

        items = torch.cat([neg_item, pos_item.unsqueeze(1)], dim=1).view(-1)
        new_batch = items.shape[0]
        ingres = torch.cat([neg_ingre, pos_ingre.unsqueeze(1)], dim=1).view(new_batch, -1)
        ingres_num = torch.cat([neg_ingre_num, pos_ingre_num.unsqueeze(1)], dim=1).view(-1)
        img = torch.cat([neg_img, pos_img.unsqueeze(1)], dim=1).view(new_batch, -1)
        hl = torch.cat([neg_hl, pos_hl.unsqueeze(1)], dim=1).view(-1)
        n = user.size(0)
        m = self.config['neg_sample_num'] + 1
        users = user.view(n, 1).expand(n, m).contiguous().view(-1)
        ingre_embedding = torch.cat([self.ingre_embed_first, self.ingre_embed_second], dim=0)
        predictions, user_emb, item_emb, ingre_emb, hl_emb, _, r_final_emb = self.compute_score(users, items, ingres,
                                                                                                ingres_num, img, hl,
                                                                                                False, self.g2i_edges,
                                                                                                self.i2u_edges,
                                                                                                ingre_embedding)
        return predictions.view(n, m)

    def inference_by_user(self, batch_data):
        user = batch_data['user_input']
        item = batch_data['item_input']
        img = batch_data['img_input']
        ingre_num = batch_data['ingre_num_input']
        ingre = batch_data['ingre_input']
        hl = batch_data['cal_level_input']

        ingre_embedding = torch.cat([self.ingre_embed_first, self.ingre_embed_second], dim=0)
        predictions, user_emb, item_emb, ingre_emb, hl_emb, _, r_final_emb = self.compute_score(user, item, ingre,
                                                                                                ingre_num, img, hl,
                                                                                                False, self.g2i_edges,
                                                                                                self.i2u_edges,
                                                                                                ingre_embedding)

        return predictions

