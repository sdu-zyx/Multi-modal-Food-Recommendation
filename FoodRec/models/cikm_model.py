import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from FoodRec.common.abstract_recommender import GeneralRecommender
from FoodRec.common.init import xavier_uniform_initialization
from FoodRec.common.loss import EmbLoss, BPRLoss


class CIKM_Model(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CIKM_Model, self).__init__(config, dataset)
        self.device = config['device']
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_ingredients = dataset.num_ingredients
        self.n_cal_level = dataset.num_calories_level
        self.n_health_level = len(dataset.health_level_multi_hot[0]) if config[
            'use_health_level_multi_hot'] else dataset.num_health_level

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config["embedding_size"],
                                                        nhead=config['num_attention_heads'],
                                                        dim_feedforward=4 * config["embedding_size"],
                                                        dropout=config['attention_probs_dropout_prob'],
                                                        activation=config['hidden_act'])
        self.ingr_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_hidden_layers'])

        self.mm_target_atten = target_attention_layer(config["embedding_size"], config["embedding_size"],
                                                      config['num_attention_heads'], linear_projection=False,
                                                      atten_mode='ln', padding_idx=self.n_ingredients)

        self.ingre_target_atten = target_attention_layer(config["embedding_size"], config["embedding_size"],
                                                         config['num_attention_heads'], linear_projection=False,
                                                         atten_mode='ln', padding_idx=self.n_ingredients)

        self.health_mlp = nn.Sequential(nn.Linear(config["embedding_size"], config["embedding_size"]),
                                        nn.ReLU(),
                                        nn.Linear(config["embedding_size"], self.n_health_level))

        self.criterion = nn.BCELoss(reduction='none')

        # load dataset info
        self.interaction_matrix = self.dataset.train_coo_matrix

        # load parameters info
        self.latent_dim = config["embedding_size"]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.ui_layers = config['ui_layers']
        self.reg_weight = config["reg_weight"]  # float32 type: the weight decay for l2 normalization
        self.loss_kd = config["loss_kd"]
        self.loss_health = config["loss_health"]
        self.kd_threshold = config["kd_threshold"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.ingre_embedding = torch.nn.Embedding(self.n_ingredients + 1, self.latent_dim,
                                                  padding_idx=self.n_ingredients)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        ri_coo_matrix = self.load_graph(dataset)
        self.ri_norm_adj = self.get_norm_adj_recipe_ing(ri_coo_matrix, self.n_items + self.n_ingredients).to(
            self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.latent_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.latent_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

    def load_graph(self, dataset):
        g2i_index = []

        for h_id, t_id in tqdm(dataset.rIngre_triples, ascii=True):
            g2i_index.append([t_id + self.n_items, h_id])

        g2i_edges = torch.tensor(np.array(g2i_index)).to(self.device, dtype=torch.long)

        rows = g2i_edges[:, 0]  # 提取所有行索引
        cols = g2i_edges[:, 1]  # 提取所有列索引
        values = torch.ones_like(rows)  # 假设所有的值都是1
        total_node = self.n_items + self.n_ingredients
        # 创建COO格式的稀疏矩阵
        coo_matrix = sp.coo_matrix((values.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
                                   shape=(total_node, total_node))
        return coo_matrix

    def get_norm_adj_recipe_ing(self, interaction_matrix, n_nodes):
        A = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self):

        ir_ego_embeddings = torch.cat((self.item_embedding.weight, self.ingre_embedding.weight[:-1, :]), dim=0)
        ir_embeddings_list = [ir_ego_embeddings]
        for layer_idx in range(self.n_layers):
            ir_ego_embeddings = torch.sparse.mm(self.ri_norm_adj, ir_ego_embeddings)
            ir_embeddings_list.append(ir_ego_embeddings)
        ir_all_embeddings = torch.stack(ir_embeddings_list, dim=1)
        ir_all_embeddings = torch.mean(ir_all_embeddings, dim=1)
        item_ir_embeddings, ingre_ir_embeddings = torch.split(
            ir_all_embeddings, [self.n_items, self.n_ingredients]
        )

        all_embeddings = torch.cat([self.user_embedding.weight, item_ir_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.ui_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings, ingre_ir_embeddings

    def calculate_loss(self, batch_data):

        user = batch_data['u_id']
        pos_item = batch_data['pos_i_id']
        neg_item = batch_data['neg_i_id']
        pos_ingredients = batch_data['pos_ingre_code']
        pos_ingre_num = batch_data['pos_ingre_num']
        pos_health_level = batch_data['pos_hl_mh']

        teacher_neg_ingredients = batch_data['neg_ingre_code']
        teacher_neg_ingre_num = batch_data['neg_ingre_num']
        teacher_neg_health_level = batch_data['neg_hl_mh']

        user_all_embeddings, item_all_embeddings, ingr_all_embeddings = self.forward()
        ingr_all_embeddings = self.ingre_embedding.weight

        health_level = torch.cat([pos_health_level, teacher_neg_health_level], dim=0)

        ingredients = torch.cat([pos_ingredients, teacher_neg_ingredients], dim=0)
        ingre_num = torch.cat([pos_ingre_num, teacher_neg_ingre_num], dim=0)
        ingr_embeddings = ingr_all_embeddings[ingredients]

        ingr_mask = (ingredients == self.n_ingredients)
        ingr_embeddings_permute = ingr_embeddings.permute(1, 0, 2)

        encoded_output = self.ingr_encoder(ingr_embeddings_permute,
                                           src_key_padding_mask=ingr_mask)

        encoded_output = encoded_output.permute(1, 0, 2).contiguous()

        text_feats = self.text_trs(self.text_embedding.weight)
        image_feats = self.image_trs(self.image_embedding.weight)
        batch_image_feats = image_feats[torch.cat([pos_item, neg_item], dim=0)].unsqueeze(1)
        batch_text_feats = text_feats[torch.cat([pos_item, neg_item], dim=0)].unsqueeze(1)
        mm_query_feats = torch.cat([batch_image_feats, batch_text_feats], dim=1)
        item_health, att_vec = self.mm_target_atten(mm_query_feats, encoded_output, ingredients)
        item_mm, att_vec = self.ingre_target_atten(encoded_output, mm_query_feats)

        norm_item_mm = F.normalize(item_mm)
        item_know = norm_item_mm.sum(1, keepdim=False) / ingre_num.unsqueeze(1)

        health_pred = torch.sigmoid(self.health_mlp(F.normalize(item_health).mean(dim=1, keepdim=False)))

        health_loss = torch.sum(self.criterion(health_pred, health_level))

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        kd_loss = 1 - cosine_similarity(item_know, torch.cat([pos_embeddings, neg_embeddings], dim=0), dim=-1).mean()
        kd_loss = self.norm_loss(kd_loss, self.kd_threshold)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        pos_ing_embeddings = self.ingre_embedding(pos_ingredients)
        neg_ing_embeddings = self.ingre_embedding(teacher_neg_ingredients)

        reg_loss = self.reg_weight * self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            pos_ing_embeddings,
            neg_ing_embeddings
        )

        return mf_loss, self.loss_health * health_loss, self.loss_kd * kd_loss, reg_loss

    def inference_by_user(self, batch_data):
        user = batch_data['user_input']
        item = batch_data['item_input']
        user_all_embeddings, item_all_embeddings, ingr_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores

    def inference_fast(self, batch_data, user_emb, item_emb):
        user = batch_data['user_input']
        item = batch_data['item_input']

        u_embeddings = user_emb[user]
        i_embeddings = item_emb[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores

    def norm_loss(self, kd_loss, threshold):
        # 计算两个张量之间的距离
        # 计算损失
        loss = torch.max(torch.tensor(0.0), kd_loss - threshold)
        return loss


class target_attention_layer(nn.Module):
    def __init__(self, model_dims, hidden, num_head, linear_projection, atten_mode, padding_idx):
        super(target_attention_layer, self).__init__()

        self.linear_projection = linear_projection
        self.num_split = int(hidden / num_head)
        self.num_head = num_head
        self.q_fc = nn.Linear(model_dims, hidden)
        self.k_fc = nn.Linear(model_dims, hidden)
        self.v_fc = nn.Linear(model_dims, hidden)
        self.atten_mode = atten_mode
        self.padding_idx = padding_idx
        if self.atten_mode == 'ln':
            self.ln = nn.LayerNorm(self.num_split, eps=1e-12)

    def forward(self, target_query, item_vec, seq_ids=None):
        queries = target_query
        keys = item_vec
        values = item_vec
        queries_len = queries.shape[1]
        keys_len = keys.shape[1]

        if self.linear_projection:
            Q = self.q_fc(queries)
            K = self.k_fc(keys)
            V = self.v_fc(values)
        else:
            Q = queries
            K = keys
            V = values

        Q_ = torch.cat(torch.chunk(Q, self.num_head, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_head, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_head, dim=2), dim=0)
        if self.atten_mode == 'ln':
            Q_ = self.ln(Q_)
            K_ = self.ln(K_)

            outputs = torch.matmul(Q_, K_.permute(0, 2, 1))
            outputs = outputs * (K_.shape[-1] ** (-0.5))
        else:
            outputs = torch.matmul(Q_, K_.permute(0, 2, 1))
            outputs = outputs * (K_.shape[-1] ** (-0.5))

        if seq_ids != None:
            key_masks = ((seq_ids == self.padding_idx).float() * (-2 ** 32 + 1)).view(-1, 1, keys_len).repeat(
                self.num_head,
                queries_len,
                1)
            outputs = (seq_ids != self.padding_idx).float().view(-1, 1, keys_len).repeat(self.num_head, queries_len,
                                                                                         1) * outputs + key_masks

        outputs = torch.softmax(outputs, dim=-1)
        att_vec = outputs

        outputs = torch.matmul(outputs, V_)
        outputs = torch.cat(torch.chunk(outputs, self.num_head, dim=0), dim=2).squeeze()

        return outputs, att_vec
