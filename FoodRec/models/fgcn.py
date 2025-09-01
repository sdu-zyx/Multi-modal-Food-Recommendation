import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

import numpy as np
from tqdm import tqdm

from FoodRec.common.abstract_recommender import GeneralRecommender
from FoodRec.common.loss import EmbLoss, BPRLoss
from FoodRec.common.init import xavier_normal_initialization


class FGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FGCN, self).__init__(config, dataset)
        self.device = config['device']
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_cold = dataset.cold_num
        self.n_health = dataset.num_calories_level
        self.n_ingredients = dataset.num_ingredients
        self.img_size = dataset.image_size

        self.emb_size = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.aggregator_type = config['aggregator_type']
        self.mess_dropout = config['mess_dropout']
        self.layers = config['layers']
        self.n_layers = config['n_layers']
        self.n_nodes = self.n_users + self.n_items + self.n_ingredients

        self.ii_aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            self.ii_aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, 'gcn'
                )
            )
        self.ir_aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            self.ir_aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )
        self.ru_aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            self.ru_aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )

        self.ru_coo_matrix, self.ir_coo_matrix, self.ii_coo_matrix = self.load_graph(self.dataset)
        self.ru_norm_adj = self.get_norm_adj_mat(self.ru_coo_matrix, self.n_users+self.n_items).to(self.device)
        self.ir_norm_adj = self.get_norm_adj_mat(self.ir_coo_matrix, self.n_items+self.n_ingredients).to(self.device)
        self.ii_norm_adj = self.get_norm_adj_mat(self.ii_coo_matrix, self.n_ingredients).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.item_id_embedding = nn.Embedding(self.n_items, self.emb_size)
        self.ingre_embedding = nn.Embedding(self.n_ingredients + 1, self.emb_size, padding_idx=self.n_ingredients)

        self.w1_conv = nn.Linear(self.emb_size, self.emb_size)
        self.reg_loss = EmbLoss()
        self.apply(xavier_normal_initialization)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_norm_adj_mat(self, interaction_matrix, n_nodes):
        A = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)

        A_coo = A.tocoo()
        rowsum = np.array(A_coo.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(A_coo).tocoo()

        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, torch.Size((n_nodes, n_nodes)))

        return adj_matrix_tensor

    def load_graph(self, dataset):
        r2u_index, i2r_index = [], []
        for uid, rid in tqdm(dataset.uRecipe_triples, ascii=True):
            r2u_index.append([rid + self.n_users, uid])

        for rid, iid in tqdm(dataset.rIngre_triples, ascii=True):
            i2r_index.append([iid + self.n_items, rid])

        r2u_edges = torch.tensor(np.array(r2u_index)).to(self.device, dtype=torch.long)
        i2r_edges = torch.tensor(np.array(i2r_index)).to(self.device, dtype=torch.long)

        i2i_index = []
        for h_id, t_id in tqdm(dataset.iIngre_triples, ascii=True):
            i2i_index.append([t_id, h_id])

        i2i_edges = torch.tensor(np.array(i2i_index)).to(self.device, dtype=torch.long)

        ru_rows = r2u_edges[:, 0]  # 提取所有行索引
        ru_cols = r2u_edges[:, 1]  # 提取所有列索引
        ru_values = torch.ones_like(ru_rows)  # 假设所有的值都是1
        ru_total_node = self.n_users + self.n_items
        # 创建COO格式的稀疏矩阵
        ru_coo_matrix = sp.coo_matrix((ru_values.cpu().numpy(), (ru_rows.cpu().numpy(), ru_cols.cpu().numpy())),
                                      shape=(ru_total_node, ru_total_node))

        ir_rows = i2r_edges[:, 0]  # 提取所有行索引
        ir_cols = i2r_edges[:, 1]  # 提取所有列索引
        ir_values = torch.ones_like(ir_rows)  # 假设所有的值都是1
        ir_total_node = self.n_items + self.n_ingredients
        # 创建COO格式的稀疏矩阵
        ir_coo_matrix = sp.coo_matrix((ir_values.cpu().numpy(), (ir_rows.cpu().numpy(), ir_cols.cpu().numpy())),
                                      shape=(ir_total_node, ir_total_node))

        ii_rows = i2i_edges[:, 0]
        ii_cols = i2i_edges[:, 1]
        ii_values = torch.ones_like(ii_rows)
        ii_coo_matrix = sp.coo_matrix((ii_values.cpu().numpy(), (ii_rows.cpu().numpy(), ii_cols.cpu().numpy())),
                                      shape=(self.n_ingredients, self.n_ingredients))

        return ru_coo_matrix, ir_coo_matrix, ii_coo_matrix

    def gnn_encode(self):

        ii_ego_embeddings = self.ingre_embedding.weight[:-1, :]
        ii_embeddings_list = [ii_ego_embeddings]
        for i in range(self.n_layers):
            ii_ego_embeddings = self.w1_conv(ii_ego_embeddings)
            ii_ego_embeddings = torch.sparse.mm(self.ii_norm_adj, ii_ego_embeddings)
            ii_embeddings_list.append(ii_ego_embeddings)
        ingre_ii_embeddings = torch.stack(ii_embeddings_list, dim=1)
        ingre_ii_embeddings = torch.mean(ingre_ii_embeddings, dim=1)

        ir_ego_embeddings = torch.cat((self.item_id_embedding.weight, ingre_ii_embeddings), dim=0)
        ir_embeddings_list = [ir_ego_embeddings]
        for aggregator in self.ir_aggregator_layers:
            ir_ego_embeddings = aggregator(self.ir_norm_adj, ir_ego_embeddings)
            norm_embeddings = F.normalize(ir_ego_embeddings, p=2, dim=1)
            ir_embeddings_list.append(norm_embeddings)
        ir_all_embeddings = torch.stack(ir_embeddings_list, dim=1)
        ir_all_embeddings = torch.mean(ir_all_embeddings, dim=1)
        item_ir_embeddings, ingre_ir_embeddings = torch.split(
            ir_all_embeddings, [self.n_items, self.n_ingredients]
        )

        ru_ego_embeddings = torch.cat((self.user_embedding.weight, item_ir_embeddings), dim=0)
        ru_embeddings_list = [ru_ego_embeddings]
        for aggregator in self.ru_aggregator_layers:
            ru_ego_embeddings = aggregator(self.ru_norm_adj, ru_ego_embeddings)
            norm_embeddings = F.normalize(ru_ego_embeddings, p=2, dim=1)
            ru_embeddings_list.append(norm_embeddings)
        ru_all_embeddings = torch.stack(ru_embeddings_list, dim=1)
        ru_all_embeddings = torch.mean(ru_all_embeddings, dim=1)
        user_ru_embeddings, item_ru_embeddings = torch.split(
            ru_all_embeddings, [self.n_users, self.n_items]
        )
        return user_ru_embeddings, self.item_id_embedding.weight, ingre_ir_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, batch_data):
        user = batch_data['u_id']
        pos_item = batch_data['pos_i_id']
        neg_item = batch_data['neg_i_id']
        user_all_embeddings, item_all_embeddings, ingre_all_embeddings = self.gnn_encode()

        u_g_embeddings = user_all_embeddings[user]
        pos_i_g_embeddings = item_all_embeddings[pos_item]
        neg_i_g_embeddings = item_all_embeddings[neg_item]

        bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                      neg_i_g_embeddings)
        reg_loss = self.reg_weight * self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return bpr_loss, reg_loss

    def inference_by_user(self, user_batch):
        user = user_batch['user_input']
        item = user_batch['item_input']
        user_all_embeddings, item_all_embeddings, ingre_all_embeddings = self.gnn_encode()
        score_mat_ui = torch.mul(user_all_embeddings[user], item_all_embeddings[item]).sum(dim=1)

        return score_mat_ui


class Aggregator(nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings
