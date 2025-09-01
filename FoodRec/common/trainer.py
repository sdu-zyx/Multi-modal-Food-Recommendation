# coding: utf-8

import math
import os
import itertools
import torch
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import copy

from time import time
from logging import getLogger

from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm

from FoodRec.utils.dataloader import EvalByUserDataloader, TrainDataLoader
from FoodRec.utils.utils import get_local_time, early_stopping, dict2str
from FoodRec.utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


def get_auc_fast(rel_list, predictions, neg_num):
    neg_predictions = predictions[len(rel_list):]
    auc_value = np.sum([np.sum(neg_predictions < predictions[idx]) for idx in rel_list])
    return auc_value / (len(rel_list) * neg_num)


def metrics_by_user(doc_list, rel_list):
    dcg = 0.0
    hit_num = 0.0

    for i in range(len(doc_list)):
        if doc_list[i] in rel_list:
            dcg += 1 / (math.log(i + 2) / math.log(2))
            hit_num += 1

    idcg = 0.0
    for i in range(min(len(doc_list), len(rel_list))):
        idcg += 1 / (math.log(i + 2) / math.log(2))
    ndcg = dcg / idcg
    recall = hit_num / len(rel_list)
    return recall, ndcg


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.test_step = min(config['test_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        # fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']  # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        total_sim = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            if torch.cuda.is_available():
                interaction = {k: v.to(self.device, non_blocking=True) for k, v in interaction.items()}
            self.optimizer.zero_grad()
            second_inter = copy.deepcopy(interaction)
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)

            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses

                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()

            if self.config['calcu_cos_similarity']:
                out = self.calcu_similarity()
                if isinstance(out, tuple):
                    sim_tuple = tuple(per_sim.item() for per_sim in out)
                    total_sim = sim_tuple if total_sim is None else tuple(map(sum, zip(total_sim, sim_tuple)))

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            # if batch_idx == 0:
            #    break
        return total_loss, loss_batches, total_sim

    def _valid_by_user_epoch(self, valid_data, is_test=False):
        res = []
        user_idx = 0
        desc = 'testing' if is_test else 'evaluating'
        if self.config['graph_inference_fast']:
            user_emb, item_emb, ingre_emb = self.model.forward()
        for user_batch in tqdm(valid_data, ascii=True, desc=desc):
            if not is_test:
                pos_items = self.model.dataset.validRatings[user_idx]
            else:
                pos_items = self.model.dataset.testRatings[user_idx]
            user_batch = {k: v.to(self.device, non_blocking=True) for k, v in user_batch.items()}
            if self.config['graph_inference_fast']:
                predictions = self.model.inference_fast(user_batch, user_emb, item_emb)
            else:
                predictions = self.model.inference_by_user(user_batch)
            predictions = predictions.cpu().detach().numpy().copy()

            pos_num = len(pos_items)
            # neg_pred, pos_pred = predictions[:-pos_num], predictions[-pos_num:]
            gt_idx = range(pos_num)

            pred_idx = np.argsort(predictions)[::-1]
            recall, ndcg, auc = [], [], []
            neg_num = self.config['neg_sample_num']
            try:
                auc_value = get_auc_fast(gt_idx, predictions, neg_num)
            except ZeroDivisionError:
                print(pos_items)
                print(user_batch['user_input'])
                print(predictions.shape)

            for k in range(10, 21, 10):
                selected_idx = pred_idx[:k]
                rec_val, ndcg_val = metrics_by_user(selected_idx, gt_idx)
                recall.append(rec_val)
                ndcg.append(ndcg_val)
                auc.append(auc_value)

            res.append((recall, ndcg, auc))
            user_idx += 1
        res = np.array(res)
        recalls, ndcgs, aucs = (res.mean(axis=0)).tolist()
        metrics = {}
        metrics['AUC'] = aucs[0]
        metrics['Recall@10'] = recalls[0]
        metrics['Recall@20'] = recalls[1]
        metrics['NDCG@10'] = ndcgs[0]
        metrics['NDCG@20'] = ndcgs[1]
        valid_result = metrics
        valid_score = valid_result['NDCG@20']
        return valid_score, valid_result

    def _valid_epoch(self, valid_data, is_test=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, is_test)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _valid_sample(self, valid_data, is_test=False):
        desc = 'testing' if is_test else 'evaluating'
        pred_list = None
        for idx, user_batch in tqdm(enumerate(valid_data), ascii=True, desc=desc):

            user_batch = {k: v.to(self.device, non_blocking=True) for k, v in user_batch.items()}

            predictions = self.model.sample_sort_predict(user_batch)

            predictions = predictions.cpu().detach().numpy().copy()
            if idx == 0:
                pred_list = predictions
            else:
                pred_list = np.append(pred_list, predictions, axis=0)

        valid_result = self.get_sample_scores(pred_list)
        valid_score = valid_result['NDCG@20']
        return valid_score, valid_result

    def get_sample_scores(self, pred_list):

        auc_value = np.sum([np.sum(prediction[0:-1] < prediction[-1]) for prediction in pred_list])
        AUC = auc_value / len(pred_list) / len(pred_list[0][0:-1])
        pred_list = (-pred_list).argsort().argsort()[:, -1]
        HIT_1, NDCG_1, MRR = self.get_metric_sample(pred_list, 1)
        HIT_5, NDCG_5, MRR = self.get_metric_sample(pred_list, 5)
        HIT_10, NDCG_10, MRR = self.get_metric_sample(pred_list, 10)
        HIT_20, NDCG_20, MRR = self.get_metric_sample(pred_list, 20)
        metrics = {}
        metrics['AUC'] = AUC
        metrics['MRR'] = MRR
        metrics['HIT@1'] = HIT_1
        metrics['HIT@5'] = HIT_5
        metrics['HIT@10'] = HIT_10
        metrics['HIT@20'] = HIT_20
        metrics['NDCG@1'] = NDCG_1
        metrics['NDCG@5'] = NDCG_5
        metrics['NDCG@10'] = NDCG_10
        metrics['NDCG@20'] = NDCG_20
        return metrics

    def get_metric_sample(self, pred_list, topk):
        NDCG = 0.0
        HIT = 0.0
        MRR = 0.0
        # [batch] the answer's rank
        for rank in pred_list:
            MRR += 1.0 / (rank + 1.0)
            if rank < topk:
                NDCG += 1.0 / np.log2(rank + 2.0)
                HIT += 1.0
        return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            # raise ValueError('Training loss is nan')
            return True

    def _genrata_emb_cos_sim_output(self, epoch_idx, len_dataloader, emb_cos_sim):
        if emb_cos_sim is None:
            pass
        else:
            emb_cos_sim_output = 'epoch %d training [' % (epoch_idx)
            if isinstance(emb_cos_sim, tuple):
                emb_cos_sim_output += ', '.join(
                    'similarity%d: %.4f' % (idx + 1, sim / len_dataloader) for idx, sim in enumerate(emb_cos_sim))
            emb_cos_sim_output = emb_cos_sim_output + ']'
            self.logger.info(emb_cos_sim_output)

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output += ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, dataset, valid_data=None, test_data=None, hyper_tuple=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            hyper_tuple(tuple, optional): the hyper-parameters

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        checkpointname = '{}-{}-{}={}.pt'.format(self.config['model'], self.config['dataset'],
                                                 self.config['hyper_parameters'], hyper_tuple)
        CKPROOT = self.config['ckp_root']
        dir_name = os.path.dirname(CKPROOT)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        checkpoint_path = os.path.join(CKPROOT, checkpointname)

        (train_data_pre, train_data_post) = (
            TrainDataLoader(self.config, dataset, use_neg_list=False),
            TrainDataLoader(self.config, dataset, use_neg_list=True))
        train_sampler = RandomSampler(train_data_pre)
        train_dataloader = DataLoader(train_data_pre, sampler=train_sampler, batch_size=self.config['train_batch_size'])

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _, emb_cos_sim = self._train_epoch(train_dataloader, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            for param_group in self.optimizer.param_groups:
                self.logger.info('======lr: %f' % param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            self._genrata_emb_cos_sim_output(epoch_idx, len(train_dataloader), emb_cos_sim)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                if self.config['eval_by_user']:
                    valid_feed_dicts = EvalByUserDataloader(self.model.dataset)
                    valid_score, valid_result = self._valid_by_user_epoch(valid_feed_dicts)
                else:
                    if self.config['full_sort']:
                        valid_score, valid_result = self._valid_epoch(valid_data)
                    else:
                        valid_score, valid_result = self._valid_sample(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)

                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if update_flag:
                    torch.save(self.model.state_dict(), checkpoint_path)
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        # test
        self.model.load_state_dict(torch.load(checkpoint_path))
        if self.config['eval_by_user']:
            test_feed_dicts = EvalByUserDataloader(self.model.dataset, is_test=True)
            _, test_result = self._valid_by_user_epoch(test_feed_dicts, is_test=True)
        else:
            if self.config['full_sort']:
                _, test_result = self._valid_epoch(test_data, is_test=True)
            else:
                _, test_result = self._valid_sample(test_data, is_test=True)
        self.logger.info('test result: \n' + dict2str(test_result))
        self.best_test_upon_valid = test_result
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        pos_items_list = []
        pos_num_list = []
        pos_user_list = []
        for batch_idx, batched_data in tqdm(enumerate(eval_data)):
            # predict: interaction without item ids
            if is_test == False:
                pos_items = self.model.dataset.validRatings[batch_idx]
            else:
                pos_items = self.model.dataset.testRatings[batch_idx]
            scores = self.model.full_sort_predict(batched_data)
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
            pos_items_list.append(pos_items)
            pos_num_list.append(len(pos_items))
            pos_user_list.append(batched_data['u_id'][0])
        return self.evaluator.evaluate(batch_matrix_list, (pos_user_list, pos_items_list, pos_num_list),
                                       is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def _valid_by_user_epoch_record_prediction(self, valid_data, is_test=False):
        res = []
        user_idx = 0
        desc = 'testing' if is_test else 'evaluating'
        predition_list = []
        pred_idx_list = []
        if self.config['graph_inference_fast']:
            user_emb, item_emb, ingre_emb = self.model.forward()
        for user_batch in tqdm(valid_data, ascii=True, desc=desc):
            if not is_test:
                pos_items = self.model.dataset.validRatings[user_idx]
            else:
                pos_items = self.model.dataset.testRatings[user_idx]
            user_batch = {k: v.to(self.device, non_blocking=True) for k, v in user_batch.items()}
            if self.config['graph_inference_fast']:
                predictions = self.model.inference_fast(user_batch, user_emb, item_emb)
            else:
                predictions = self.model.inference_by_user(user_batch)
            predictions = predictions.cpu().detach().numpy().copy()

            pos_num = len(pos_items)
            # neg_pred, pos_pred = predictions[:-pos_num], predictions[-pos_num:]
            gt_idx = range(pos_num)

            pred_idx = np.argsort(predictions)[::-1]

            predition_list.append(predictions)
            pred_idx_list.append(pred_idx)

            recall, ndcg, auc = [], [], []
            neg_num = self.config['neg_sample_num']
            try:
                auc_value = get_auc_fast(gt_idx, predictions, neg_num)
            except ZeroDivisionError:
                print(pos_items)
                print(user_batch['user_input'])
                print(predictions.shape)

            for k in range(10, 21, 10):
                selected_idx = pred_idx[:k]
                rec_val, ndcg_val = metrics_by_user(selected_idx, gt_idx)
                recall.append(rec_val)
                ndcg.append(ndcg_val)
                auc.append(auc_value)

            res.append((recall, ndcg, auc))
            user_idx += 1
        res = np.array(res)
        recalls, ndcgs, aucs = (res.mean(axis=0)).tolist()
        metrics = {}
        metrics['AUC'] = aucs[0]
        metrics['Recall@10'] = recalls[0]
        metrics['Recall@20'] = recalls[1]
        metrics['NDCG@10'] = ndcgs[0]
        metrics['NDCG@20'] = ndcgs[1]
        valid_result = metrics
        valid_score = valid_result['NDCG@20']
        return valid_score, valid_result, predition_list, pred_idx_list

    def calcu_similarity(self):
        # user_emb = self.model.user_emb
        id_emb = self.model.id_emb
        text_emb = self.model.text_emb
        image_emb = self.model.image_emb
        # ingre_emb = self.model.ingre_emb
        # user_emb_grad = self.model.user_emb.grad
        id_emb_grad = self.model.id_emb.grad
        text_emb_grad = self.model.text_emb.grad
        image_emb_grad = self.model.image_emb.grad
        # ingre_emb_grad = self.model.ingre_emb.grad

        # u_id = cosine_similarity(user_emb, id_emb, dim=-1).mean()
        # u_id_grad = cosine_similarity(user_emb_grad, id_emb_grad, dim=-1).mean()
        #
        # u_text = cosine_similarity(user_emb, text_emb, dim=-1).mean()
        # u_text_grad = cosine_similarity(user_emb_grad, text_emb_grad, dim=-1).mean()
        #
        # u_image = cosine_similarity(user_emb, image_emb, dim=-1).mean()
        # u_image_grad = cosine_similarity(user_emb_grad, image_emb_grad, dim=-1).mean()

        id_text = cosine_similarity(id_emb, text_emb, dim=-1).mean()
        id_text_grad = cosine_similarity(id_emb_grad, text_emb_grad, dim=-1).mean()

        id_image = cosine_similarity(id_emb, image_emb, dim=-1).mean()
        id_image_grad = cosine_similarity(id_emb_grad, image_emb_grad, dim=-1).mean()

        id_norms = torch.norm(id_emb, dim=-1, keepdim=True)
        id_norms = id_emb / id_norms

        text_norms = torch.norm(text_emb, dim=-1, keepdim=True)
        text_norms = text_emb / text_norms

        image_norms = torch.norm(image_emb, dim=-1, keepdim=True)
        image_norms = image_emb / image_norms

        diff_id_text = text_norms - id_norms
        pos_id_text = diff_id_text[diff_id_text > 0].size(0) / diff_id_text.numel()
        pos_id_text = torch.tensor(pos_id_text, dtype=torch.float32, device=self.device)

        diff_id_image = image_norms - id_norms
        pos_id_image = diff_id_image[diff_id_image > 0].size(0) / diff_id_image.numel()
        pos_id_image = torch.tensor(pos_id_image, dtype=torch.float32, device=self.device)

        # return u_id, u_id_grad, u_text, u_text_grad, u_image, u_image_grad, id_text, id_text_grad, id_image, id_image_grad
        return id_text, id_text_grad, id_image, id_image_grad, pos_id_text, pos_id_image

    def _valid_by_user_cold_start_study(self, test_data, is_cold=True):
        res = []
        user_idx = 0
        desc = 'cold item testing' if is_cold else 'warm item testing'
        predition_list = []
        pred_idx_list = []
        if self.config['graph_inference_fast']:
            user_emb, item_emb, ingre_emb = self.model.forward()
        for user_batch in tqdm(test_data, ascii=True, desc=desc):
            if is_cold:
                pos_items = self.model.dataset.coldRatings[user_idx]
            else:
                pos_items = self.model.dataset.warmRatings[user_idx]
            user_batch = {k: v.to(self.device, non_blocking=True) for k, v in user_batch.items()}
            if self.config['graph_inference_fast']:
                predictions = self.model.inference_fast(user_batch, user_emb, item_emb)
            else:
                predictions = self.model.inference_by_user(user_batch)
            predictions = predictions.cpu().detach().numpy().copy()

            pos_num = len(pos_items)
            # neg_pred, pos_pred = predictions[:-pos_num], predictions[-pos_num:]
            gt_idx = range(pos_num)

            pred_idx = np.argsort(predictions)[::-1]

            predition_list.append(predictions)
            pred_idx_list.append(pred_idx)

            recall, ndcg, auc = [], [], []
            neg_num = self.config['neg_sample_num']
            try:
                auc_value = get_auc_fast(gt_idx, predictions, neg_num)
            except ZeroDivisionError:
                print(pos_items)
                print(user_batch['user_input'])
                print(predictions.shape)

            for k in range(10, 21, 10):
                selected_idx = pred_idx[:k]
                rec_val, ndcg_val = metrics_by_user(selected_idx, gt_idx)
                recall.append(rec_val)
                ndcg.append(ndcg_val)
                auc.append(auc_value)

            res.append((recall, ndcg, auc))
            user_idx += 1
        res = np.array(res)
        recalls, ndcgs, aucs = (res.mean(axis=0)).tolist()
        metrics = {}
        metrics['AUC'] = aucs[0]
        metrics['Recall@10'] = recalls[0]
        metrics['Recall@20'] = recalls[1]
        metrics['NDCG@10'] = ndcgs[0]
        metrics['NDCG@20'] = ndcgs[1]
        valid_result = metrics
        valid_score = valid_result['NDCG@20']
        return valid_score, valid_result, predition_list, pred_idx_list

    def _valid_by_user_health_level_study(self, test_data, health_level):
        res = []
        user_idx = 0
        desc = r'health_level_{} testing'.format(health_level)
        predition_list = []
        pred_idx_list = []
        if self.config['graph_inference_fast']:
            user_emb, item_emb, ingre_emb = self.model.forward()
        rating = self.model.dataset.healthRatings[health_level]
        for user_batch in tqdm(test_data, ascii=True, desc=desc):
            pos_items = rating[user_idx]
            user_batch = {k: v.to(self.device, non_blocking=True) for k, v in user_batch.items()}
            if self.config['graph_inference_fast']:
                predictions = self.model.inference_fast(user_batch, user_emb, item_emb)
            else:
                predictions = self.model.inference_by_user(user_batch)
            predictions = predictions.cpu().detach().numpy().copy()

            pos_num = len(pos_items)
            # neg_pred, pos_pred = predictions[:-pos_num], predictions[-pos_num:]
            gt_idx = range(pos_num)

            pred_idx = np.argsort(predictions)[::-1]

            predition_list.append(predictions)
            pred_idx_list.append(pred_idx)

            recall, ndcg, auc = [], [], []
            neg_num = self.config['neg_sample_num']
            try:
                auc_value = get_auc_fast(gt_idx, predictions, neg_num)
            except ZeroDivisionError:
                print(pos_items)
                print(user_batch['user_input'])
                print(predictions.shape)

            for k in range(10, 21, 10):
                selected_idx = pred_idx[:k]
                rec_val, ndcg_val = metrics_by_user(selected_idx, gt_idx)
                recall.append(rec_val)
                ndcg.append(ndcg_val)
                auc.append(auc_value)

            res.append((recall, ndcg, auc))
            user_idx += 1
        res = np.array(res)
        recalls, ndcgs, aucs = (res.mean(axis=0)).tolist()
        metrics = {}
        metrics['AUC'] = aucs[0]
        metrics['Recall@10'] = recalls[0]
        metrics['Recall@20'] = recalls[1]
        metrics['NDCG@10'] = ndcgs[0]
        metrics['NDCG@20'] = ndcgs[1]
        valid_result = metrics
        valid_score = valid_result['NDCG@20']
        return valid_score, valid_result, predition_list, pred_idx_list

    def _valid_by_user_sense_study(self, test_data, is_sense=True):
        res = []
        user_idx = 0
        desc = 'sense user testing' if is_sense else 'unsense user testing'
        predition_list = []
        pred_idx_list = []
        if self.config['graph_inference_fast']:
            user_emb, item_emb, ingre_emb = self.model.forward()
        for user_batch in tqdm(test_data, ascii=True, desc=desc):
            if is_sense:
                pos_items = self.model.dataset.senseRatings[user_idx]
            else:
                pos_items = self.model.dataset.unsenseRatings[user_idx]
            user_batch = {k: v.to(self.device, non_blocking=True) for k, v in user_batch.items()}
            if self.config['graph_inference_fast']:
                predictions = self.model.inference_fast(user_batch, user_emb, item_emb)
            else:
                predictions = self.model.inference_by_user(user_batch)
            predictions = predictions.cpu().detach().numpy().copy()

            pos_num = len(pos_items)
            # neg_pred, pos_pred = predictions[:-pos_num], predictions[-pos_num:]
            gt_idx = range(pos_num)

            pred_idx = np.argsort(predictions)[::-1]

            predition_list.append(predictions)
            pred_idx_list.append(pred_idx)

            recall, ndcg, auc = [], [], []
            neg_num = self.config['neg_sample_num']
            try:
                auc_value = get_auc_fast(gt_idx, predictions, neg_num)
            except ZeroDivisionError:
                print(pos_items)
                print(user_batch['user_input'])
                print(predictions.shape)

            for k in range(10, 21, 10):
                selected_idx = pred_idx[:k]
                rec_val, ndcg_val = metrics_by_user(selected_idx, gt_idx)
                recall.append(rec_val)
                ndcg.append(ndcg_val)
                auc.append(auc_value)

            res.append((recall, ndcg, auc))
            user_idx += 1
        res = np.array(res)
        recalls, ndcgs, aucs = (res.mean(axis=0)).tolist()
        metrics = {}
        metrics['AUC'] = aucs[0]
        metrics['Recall@10'] = recalls[0]
        metrics['Recall@20'] = recalls[1]
        metrics['NDCG@10'] = ndcgs[0]
        metrics['NDCG@20'] = ndcgs[1]
        valid_result = metrics
        valid_score = valid_result['NDCG@20']
        return valid_score, valid_result, predition_list, pred_idx_list