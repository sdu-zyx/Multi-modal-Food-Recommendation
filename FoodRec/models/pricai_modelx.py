import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from FoodRec.common.abstract_recommender import GeneralRecommender
from FoodRec.common.init import xavier_uniform_initialization
from FoodRec.common.loss import EmbLoss, BPRLoss


class PRICAI_ModelX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PRICAI_ModelX, self).__init__(config, dataset)
        self.device = config['device']
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_ingredients = dataset.num_ingredients
        self.n_cal_level = dataset.num_calories_level
        self.n_health_level = len(dataset.health_level_multi_hot[0]) if config['use_health_level_multi_hot'] else dataset.num_health_level

        # load dataset info
        self.interaction_matrix = self.dataset.train_coo_matrix

        # load parameters info
        self.latent_dim = config["embedding_size"]  # int type:the embedding size of lightGCN
        self.n_ri_layers = config["n_ri_layers"]
        self.n_mm_layers = config["n_mm_layers"]
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]  # float32 type: the weight decay for l2 normalization
        self.loss_cl = config['loss_cl']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.n_cluster = config['n_cluster']

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
        image_coo_matrix = self.load_graph(dataset.image_cluster_triples)
        self.image_norm_adj = self.get_norm_adj_recipe_infor(image_coo_matrix, self.n_items + self.n_cluster).to(
            self.device)
        text_coo_matrix = self.load_graph(dataset.text_cluster_triples)
        self.text_norm_adj = self.get_norm_adj_recipe_infor(text_coo_matrix, self.n_items + self.n_cluster).to(
            self.device)
        ingre_coo_matrix = self.load_graph(dataset.rIngre_triples)
        self.ingre_norm_adj = self.get_norm_adj_recipe_infor(ingre_coo_matrix, self.n_items + self.n_ingredients).to(
            self.device)

        self.proj_ingre = nn.Linear(self.latent_dim, self.latent_dim)
        self.proj_text = nn.Linear(self.latent_dim, self.latent_dim)
        self.proj_image = nn.Linear(self.latent_dim, self.latent_dim)
        self.image_prototype_embedding = torch.nn.Embedding(self.n_cluster, self.latent_dim)
        self.text_prototype_embedding = torch.nn.Embedding(self.n_cluster, self.latent_dim)

        self.apply(xavier_uniform_initialization)

        self.v_center, self.t_center = None, None
        if config['use_center_embedding']:
            self.v_center = torch.tensor(np.load(config['interaction_data_path'] + 'mm_cluster/image_center.npy').astype(np.float32)).to(self.device)
            self.t_center = torch.tensor(np.load(config['interaction_data_path'] + 'mm_cluster/text_center.npy').astype(np.float32)).to(self.device)
        if self.v_center is not None:
            self.image_prototype_embedding = nn.Embedding.from_pretrained(self.v_center, freeze=False)
            self.image_trs = nn.Linear(self.v_center.shape[1], self.latent_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_center is not None:
            self.text_prototype_embedding = nn.Embedding.from_pretrained(self.t_center, freeze=False)
            self.text_trs = nn.Linear(self.t_center.shape[1], self.latent_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

    def load_graph(self, triples):
        g2i_index = []

        for h_id, t_id in tqdm(triples, ascii=True):
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

    def get_norm_adj_recipe_infor(self, interaction_matrix, n_nodes):
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
        ingre_ego_embeddings = torch.cat((self.item_embedding.weight, self.ingre_embedding.weight[:-1, :]), dim=0)
        ingre_embeddings_list = [ingre_ego_embeddings]
        for layer_idx in range(self.n_ri_layers):
            ingre_ego_embeddings = torch.sparse.mm(self.ingre_norm_adj, ingre_ego_embeddings)
            ingre_embeddings_list.append(ingre_ego_embeddings)
        ingre_all_embeddings = torch.stack(ingre_embeddings_list, dim=1)
        ingre_all_embeddings = torch.mean(ingre_all_embeddings, dim=1)
        item_ingre_embeddings, ingre_embeddings = torch.split(
            ingre_all_embeddings, [self.n_items, self.n_ingredients]
        )
        if self.v_center is not None:
            image_prototype = self.image_trs(self.image_prototype_embedding.weight)
            image_ego_embeddings = torch.cat((self.item_embedding.weight, image_prototype), dim=0)
        else:
            image_ego_embeddings = torch.cat((self.item_embedding.weight, self.image_prototype_embedding.weight), dim=0)
        image_embeddings_list = [image_ego_embeddings]
        for layer_idx in range(self.n_ri_layers):
            image_ego_embeddings = torch.sparse.mm(self.image_norm_adj, image_ego_embeddings)
            image_embeddings_list.append(image_ego_embeddings)
        image_all_embeddings = torch.stack(image_embeddings_list, dim=1)
        image_all_embeddings = torch.mean(image_all_embeddings, dim=1)
        item_image_embeddings, image_embeddings = torch.split(
            image_all_embeddings, [self.n_items, self.n_cluster]
        )
        if self.t_center is not None:
            text_prototype = self.text_trs(self.text_prototype_embedding.weight)
            text_ego_embeddings = torch.cat((self.item_embedding.weight, text_prototype), dim=0)
        else:
            text_ego_embeddings = torch.cat((self.item_embedding.weight, self.text_prototype_embedding.weight), dim=0)
        text_embeddings_list = [text_ego_embeddings]
        for layer_idx in range(self.n_ri_layers):
            text_ego_embeddings = torch.sparse.mm(self.text_norm_adj, text_ego_embeddings)
            text_embeddings_list.append(text_ego_embeddings)
        text_all_embeddings = torch.stack(text_embeddings_list, dim=1)
        text_all_embeddings = torch.mean(text_all_embeddings, dim=1)
        item_text_embeddings, text_embeddings = torch.split(
            text_all_embeddings, [self.n_items, self.n_cluster]
        )

        item_emb = item_ingre_embeddings + item_image_embeddings + item_text_embeddings
        all_embeddings = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_ui_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings, (item_image_embeddings, item_text_embeddings, item_ingre_embeddings)

    def calculate_loss(self, batch_data):

        user = batch_data['u_id']
        pos_item = batch_data['pos_i_id']
        neg_item = batch_data['neg_i_id']
        pos_ingredients = batch_data['pos_ingre_code']
        neg_ingredients = batch_data['neg_ingre_code']
        all_item = torch.cat([pos_item, neg_item], dim=0)

        user_all_embeddings, item_all_embeddings, hidden_views = self.forward()
        image_embeddings, text_embeddings, ingre_embeddings = hidden_views
        item_image = image_embeddings[all_item]
        item_text = text_embeddings[all_item]
        item_ingre = ingre_embeddings[all_item]
        # item_image = self.proj_image(item_image)
        # item_text = self.proj_text(item_text)
        # item_ingre = self.proj_ingre(item_ingre)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss_g = self.mf_loss(pos_scores, neg_scores)
        # cl_loss = self.CL_loss(torch.cat([item_image, item_text], dim=0)) + self.CL_loss(torch.cat([item_image, item_ingre], dim=0)) + self.CL_loss(torch.cat([item_ingre, item_text], dim=0))
        # cl_loss = self.poly_view_cl(item_image, item_text, item_ingre)
        # cl_loss = self.min_mutual_information(item_image, item_text, item_ingre)
        # cl_loss = self.OrthogonalLoss(item_image, item_text, item_ingre)
        cl_loss = self.correlation_distance(item_image, item_text) + self.correlation_distance(item_image, item_ingre) + self.correlation_distance(item_ingre, item_text)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_weight * self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings
        )

        return mf_loss_g, self.loss_cl*cl_loss, reg_loss

    def inference_fast(self, batch_data, user_emb, item_emb):
        user = batch_data['user_input']
        item = batch_data['item_input']

        u_embeddings = user_emb[user]
        i_embeddings = item_emb[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores

    def l2_normalize(self, tensor, axis=-1, epsilon=1e-12):
        """
        对 PyTorch 张量进行 L2 归一化。

        参数：
        - tensor: 输入的 PyTorch 张量
        - axis: 归一化的轴，默认为最后一个轴（-1）
        - epsilon: 避免除以零的小常数，默认为 1e-12

        返回值：
        - normalized_tensor: L2 归一化后的张量
        """
        # 计算 L2 范数
        norm = torch.norm(tensor, p=2, dim=axis, keepdim=True)

        # 避免除以零，并进行归一化
        normalized_tensor = tensor / (norm + epsilon)

        return normalized_tensor

    def get_mask(self, beta, k, m):
        """The self-supervised target is j=i, beta=alpha. Produce a mask that
        removes the contribution of j=i, beta!=alpha, i.e. return a [k,m,k]
        tensor of zeros with ones on:
        - The self-sample index
    - The betas not equal to alpha
        """
        # mask the sample
        diagonal_k = torch.eye(k)
        mask_sample = diagonal_k.view(k, 1, k)
        # mask the beta-th view
        one_m = torch.ones(m)
        mask_beta = one_m.view(1, m, 1)
        mask_beta[:, beta, :] = 0
        return mask_beta * mask_sample

    def poly_view_cl(self, i1, i2, i3, tau=0.5, method='arithmetic'):
        x_a = torch.concat([i1.unsqueeze(1), i2.unsqueeze(1), i3.unsqueeze(1)], dim=1)
        z = self.l2_normalize(x_a)
        scores = torch.einsum("jmd,knd->jmnk", z, z) / tau
        losses_alpha = list()
        m = x_a.shape[1]
        k = x_a.shape[0]
        for alpha in range(m):
            losses_alpha_beta = list()
            for beta in range(m):
                if alpha != beta:
                    logits = scores[:, alpha, :, :]
                    labels = torch.tensor(np.array(list(range(k))) + beta*k, device=self.device)
                    mask = self.get_mask(beta, k, m).to(self.device)
                    logits = (logits - mask * 1e6).view(k, m*k)
                    loss_alpha_beta = F.cross_entropy(logits, labels)
                    losses_alpha_beta.append(loss_alpha_beta)
            losses_alpha_b = torch.stack(losses_alpha_beta, dim=-1)

            if method == "arithmetic":
                loss_alpha = torch.logsumexp(losses_alpha_b, dim=-1) - torch.tensor(np.log(k), device=self.device)  # [k]
            elif method == "geometric":
                loss_alpha = torch.mean(losses_alpha_b, dim=-1)
            losses_alpha.append(loss_alpha)
        losses = torch.stack(losses_alpha, dim=-1)
        sample_losses = torch.mean(losses, dim=-1)
        loss = torch.mean(sample_losses)
        return loss


    def CL_loss(self, hidden, hidden_norm=True, temperature=0.5):
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9
        # inner dot or cosine
        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        hidden_list = torch.split(hidden, batch_size, dim=0)
        hidden1, hidden2 = hidden_list[0], hidden_list[1]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)/batch_size
        return loss

    def min_mutual_information(self, A, B, C):
        sim_AB = F.cosine_similarity(A, B)
        sim_AC = F.cosine_similarity(A, C)
        sim_BC = F.cosine_similarity(B, C)

        # 计算对比损失
        loss_AB = -torch.log(1 - sim_AB.mean() + 1e-8)
        loss_AC = -torch.log(1 - sim_AC.mean() + 1e-8)
        loss_BC = -torch.log(1 - sim_BC.mean() + 1e-8)

        # 总损失
        loss = (loss_AB + loss_AC + loss_BC) / 3

        return loss

    def OrthogonalLoss(self, A, B, C):
        # 计算A和B的点积
        AB_dot_product = torch.sum(A * B, dim=1)
        # 计算A和C的点积
        AC_dot_product = torch.sum(A * C, dim=1)
        # 计算B和C的点积
        BC_dot_product = torch.sum(B * C, dim=1)

        # 计算损失，最小化点积的平方和
        loss = (AB_dot_product ** 2).mean() + (AC_dot_product ** 2).mean() + (BC_dot_product ** 2).mean()

        return loss


    def correlation_distance(self, x, y):
        zero = torch.zeros(1, dtype=torch.float).to(x.device)
        def _create_centered_distance(X, zero):
            r = torch.sum(torch.square(X), 1, keepdim=True)
            X_t = X.permute(1, 0)
            r_t = r.permute(1, 0)
            D = torch.sqrt(torch.maximum(r-2*torch.matmul(X,X_t)+r_t,zero)+1e-8)
            D = D - torch.mean(D, 0, keepdim=True) - torch.mean(D, 1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(D1, D2, zero):
            n_samples = D1.shape[0]
            n_samples = torch.ones(1, dtype=torch.float).to(D1.device) * n_samples
            sum = torch.sum(D1*D2)
            sum = torch.div(sum, n_samples*n_samples)
            dcov = torch.sqrt(torch.maximum(sum, zero)+1e-8)
            return dcov

        D1 = _create_centered_distance(x, zero)
        D2 = _create_centered_distance(y, zero)

        dcov_12 = _create_distance_covariance(D1, D2, zero)
        dcov_11 = _create_distance_covariance(D1, D1, zero)
        dcov_22 = _create_distance_covariance(D2, D2, zero)

        dcor = torch.sqrt(torch.maximum(dcov_11*dcov_22, zero)+1e-10)
        dcor = torch.div(dcov_12, dcor)

        return dcor

