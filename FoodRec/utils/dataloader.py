import pickle

import torch
import torch.utils.data as data
import random
import numpy as np

from FoodRec.utils.utils import get_neg_ingre


class TrainDataLoader(data.Dataset):
    def __init__(self, args_config, dataset, use_neg_list=False):
        super(TrainDataLoader, self).__init__()
        self.args_config = args_config
        self.dataset = dataset
        self.n_ingredients = dataset.num_ingredients
        self.max_len = 20
        self.masked_p = 0.2
        self._user_input, self._item_input_pos, self._ingre_input_pos, self._ingre_num_pos, self._image_input_pos = self.init_samples()
        self.use_neg_list = use_neg_list
        self.neg_list = self.init_neg_list()
        if args_config['health_neg_sample']:
            with open(args_config['graph_data_path'] + 'health_sample_dict.pkl', 'rb') as f:
                self.neg_sample_set, self.health_0, self.health_1, self.health_2, self.health_3, self.health_4, self.health_5 = pickle.load(
                    f)

    def __len__(self):
        return len(self._user_input)

    def init_samples(self):
        _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos = [], [], [], [], []
        for (u, i) in self.dataset.trainMatrix.keys():
            _user_input.append(u)
            _item_input_pos.append(i)
            _ingre_input_pos.append(self.dataset.ingredientCodeDict[i])
            _ingre_num_pos.append(self.dataset.ingredientNum[i])
            _image_input_pos.append(self.dataset.embImage[i])
        return _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos

    def init_neg_list(self):
        neg_list = []
        for user in self._user_input:
            pos_items = self.dataset.trainList[user]
            pos_validTest = self.dataset.validTestRatings[user]
            neg_item = self.get_random_neg(pos_items, pos_validTest)
            neg_list.append(neg_item)
        neg_lists = random.sample(neg_list, len(neg_list))
        return neg_lists

    def __getitem__(self, index):
        # users,
        # pos_items, pos_image, pos_hl, pos_cate,
        # neg_items, neg_image, neg_hl, neg_cate,

        out_dict = {}
        u_id = self._user_input[index]
        out_dict['u_id'] = u_id
        pos_i_id = self._item_input_pos[index]
        out_dict['pos_i_id'] = pos_i_id
        out_dict['pos_ingre_code'] = self._ingre_input_pos[index]
        out_dict['pos_ingre_num'] = self._ingre_num_pos[index]
        out_dict['pos_img'] = self._image_input_pos[index]
        # out_dict['pos_ingre_emb'] = self.dataset.ingre_emb[pos_i_id]
        if self.args_config['SCHGN_ssl']:
            out_dict['masked_ingre_seq'], out_dict['pos_ingre_seq'], out_dict['neg_ingre_seq'] = self.ssl_task(
                out_dict['pos_ingre_code'], out_dict['pos_ingre_num'])
        if self.use_neg_list:
            neg_i_id = self.neg_list[index]
        else:
            pos_items = self.dataset.trainList[u_id]
            pos_validTest = self.dataset.validTestRatings[u_id]
            neg_i_id = self.get_random_neg(pos_items, pos_validTest)
        out_dict['neg_i_id'] = neg_i_id
        out_dict['neg_ingre_code'] = self.dataset.ingredientCodeDict[neg_i_id]
        out_dict['neg_ingre_num'] = self.dataset.ingredientNum[neg_i_id]
        out_dict['neg_img'] = self.dataset.embImage[neg_i_id]
        # out_dict['neg_ingre_emb'] = self.dataset.ingre_emb[neg_i_id]
        if self.args_config['use_cal_level']:
            out_dict['pos_cl'] = self.dataset.cal_level[pos_i_id]
            out_dict['neg_cl'] = self.dataset.cal_level[neg_i_id]
        if self.args_config['use_health_level']:
            out_dict['pos_hl'] = self.dataset.health_level[pos_i_id]
            out_dict['neg_hl'] = self.dataset.health_level[neg_i_id]
        if self.args_config['use_health_level_multi_hot']:
            out_dict['pos_hl_mh'] = torch.tensor(self.dataset.health_level_multi_hot[pos_i_id], dtype=torch.float)
            out_dict['neg_hl_mh'] = torch.tensor(self.dataset.health_level_multi_hot[neg_i_id], dtype=torch.float)
        if self.args_config['health_neg_sample']:
            pos_items = self.dataset.trainList[u_id]
            pos_validTest = self.dataset.validTestRatings[u_id]

            while True:
                if u_id in self.neg_sample_set:
                    if self.dataset.health_level[pos_i_id] == 0:
                        health_neg = random.choice(self.health_0)
                    elif self.dataset.health_level[pos_i_id] == 1:
                        health_neg = random.choice(self.health_1)
                    elif self.dataset.health_level[pos_i_id] == 2:
                        health_neg = random.choice(self.health_2)
                    elif self.dataset.health_level[pos_i_id] == 3:
                        health_neg = random.choice(self.health_3)
                    elif self.dataset.health_level[pos_i_id] == 4:
                        health_neg = random.choice(self.health_4)
                    else:
                        health_neg = random.choice(self.health_5)
                else:
                    health_neg = random.choice(self.dataset.train_item_list)
                if health_neg not in pos_items and health_neg not in pos_validTest:
                    break
            out_dict['health_neg'] = health_neg
            out_dict['health_neg_ingre_code'] = self.dataset.ingredientCodeDict[health_neg]
            out_dict['health_neg_ingre_num'] = self.dataset.ingredientNum[health_neg]
            out_dict['health_neg_img'] = self.dataset.embImage[health_neg]
            out_dict['health_neg_cl'] = self.dataset.cal_level[health_neg]
            out_dict['health_neg_hl'] = self.dataset.health_level[health_neg]
        return out_dict

    def ssl_task(self, ingre_seq, ingre_num):
        masked_ingre_seq = []
        neg_ingre = []
        pos_ingre = []
        ingre_set = set(ingre_seq[:ingre_num])
        for idx, ingre in enumerate(ingre_seq):
            if idx < ingre_num:
                pos_ingre.append(ingre)
                prob = random.random()
                if prob < self.masked_p:
                    masked_ingre_seq.append(self.n_ingredients + 1)
                    neg_ingre.append(get_neg_ingre(ingre_set, self.n_ingredients))
                else:
                    masked_ingre_seq.append(ingre)
                    neg_ingre.append(ingre)
            else:
                pos_ingre.append(ingre)
                masked_ingre_seq.append(ingre)
                neg_ingre.append(ingre)

        assert len(masked_ingre_seq) == self.max_len
        assert len(pos_ingre) == self.max_len
        assert len(neg_ingre) == self.max_len

        return torch.tensor(masked_ingre_seq, dtype=torch.long), \
            torch.tensor(pos_ingre, dtype=torch.long), \
            torch.tensor(neg_ingre, dtype=torch.long)

    def get_random_neg(self, train_pos, validTest_pos):
        while True:
            neg_i_id = np.random.randint(self.dataset.num_items)
            # neg_i_id = random.choice(self.dataset.train_item_list)
            if neg_i_id not in train_pos and neg_i_id not in validTest_pos:
                break
        return neg_i_id


class EvalDataLoader(data.Dataset):
    def __init__(self, args_config, dataset, phase='val', full_sort=False):
        super(EvalDataLoader, self).__init__()
        self.args_config = args_config
        self.full_sort = full_sort
        self.dataset = dataset
        if not full_sort:
            self._user_input, self._item_input_pos, self._ingre_input_pos, self._ingre_num_pos, self._image_input_pos = (
                self.init_eval(phase))
        if phase == 'val':
            self.user_ids = self.dataset.valid_users
        else:
            self.user_ids = list(range(self.dataset.n_users))

    def __len__(self):
        if self.full_sort:
            return len(self.user_ids)
        else:
            return len(self._user_input)

    def init_eval(self, phase):
        _ingre_input_pos, _ingre_num_pos, _image_input_pos = [], [], []
        if phase == 'val':
            _user_input = list(self.dataset.valid_data[:, 0])
            _item_input_pos = list(self.dataset.valid_data[:, 1] - self.dataset.n_users)
        else:
            _user_input = list(self.dataset.test_data[:, 0])
            _item_input_pos = list(self.dataset.test_data[:, 1] - self.dataset.n_users)
        assert len(_user_input) == len(_item_input_pos)
        for i in _item_input_pos:
            _ingre_input_pos.append(self.dataset.ingredientCodeDict[i])
            _ingre_num_pos.append(self.dataset.ingredientNum[i])
            _image_input_pos.append(self.dataset.embImage[i])
        return _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos

    def __getitem__(self, index):
        # users,
        # pos_items, pos_image, pos_hl, pos_cate,
        # neg_items, neg_image, neg_hl, neg_cate,
        out_dict = {}
        if not self.full_sort:
            u_id = self._user_input[index]
            out_dict['u_id'] = u_id
            pos_i_id = self._item_input_pos[index]
            out_dict['pos_i_id'] = pos_i_id
            out_dict['pos_ingre_code'] = self._ingre_input_pos[index]
            out_dict['pos_ingre_num'] = self._ingre_num_pos[index]
            out_dict['pos_img'] = self._image_input_pos[index]

            neg_items = self.dataset.testNegatives[u_id]
            ingre_code_list, ingre_num_list, img_list = [], [], []
            for i in neg_items:
                ingre_code_list.append(self.dataset.ingredientCodeDict[i])
                ingre_num_list.append(self.dataset.ingredientNum[i])
                img_list.append(self.dataset.embImage[i])
            out_dict['neg_ingre_code'], out_dict['neg_ingre_num'], out_dict['neg_img'] = \
                torch.tensor(np.array(ingre_code_list), dtype=torch.long), torch.tensor(np.array(ingre_num_list), dtype=
                torch.long), torch.tensor(np.array(img_list), dtype=
                torch.float32)
            out_dict['neg_i_id'] = torch.tensor(neg_items, dtype=torch.long)

            if self.args_config['use_cal_level']:
                out_dict['pos_cl'] = self.dataset.cal_level[pos_i_id]
                cal_level_list = []
                for i in neg_items:
                    cal_level_list.append(self.dataset.cal_level[i])
                out_dict['neg_cl'] = torch.tensor(np.array(cal_level_list), dtype=torch.long)

        else:
            u_id = self.user_ids[index]
            out_dict['u_id'] = u_id
        return out_dict


def EvalByUserDataloader(dataset, is_test=False):
    if not is_test:
        valid_users = dataset.valid_users
        for idx in range(len(valid_users)):
            user = valid_users[idx]
            pos_item_list = dataset.validRatings[idx]
            neg_item_list = dataset.validNegatives[idx]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = pos_item_list + neg_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            cal_level_input = []
            health_level_input = []
            if dataset.args_config['use_cal_level']:
                for item in item_input:
                    cal_level_input.append(dataset.cal_level[item])
                    health_level_input.append(dataset.cal_level[item])
                cal_level_input = np.array(cal_level_input)
                health_level_input = np.array(health_level_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(),
                'cal_level_input': torch.tensor(cal_level_input).long(),
                'health_level_input': torch.tensor(health_level_input).long()
            }
            yield user_batch
    else:
        for user in range(dataset.num_users):
            pos_item_list = dataset.testRatings[user]
            neg_item_list = dataset.testNegatives[user]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = pos_item_list + neg_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            cal_level_input = []
            health_level_input = []
            if dataset.args_config['use_cal_level']:
                for item in item_input:
                    cal_level_input.append(dataset.cal_level[item])
                    health_level_input.append(dataset.cal_level[item])
                cal_level_input = np.array(cal_level_input)
                health_level_input = np.array(health_level_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(),
                'cal_level_input': torch.tensor(cal_level_input).long(),
                'health_level_input': torch.tensor(health_level_input).long()
            }
            yield user_batch


def EvalByUserColdStartDataloader(dataset, is_cold=True):
    if is_cold:
        test_users = dataset.cold_users
        for idx in range(len(test_users)):
            user = test_users[idx]
            pos_item_list = dataset.coldRatings[idx]
            neg_item_list = dataset.coldNegatives[idx]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = pos_item_list + neg_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            cal_level_input = []
            health_level_input = []
            if dataset.args_config['use_cal_level']:
                for item in item_input:
                    cal_level_input.append(dataset.cal_level[item])
                    health_level_input.append(dataset.cal_level[item])
                cal_level_input = np.array(cal_level_input)
                health_level_input = np.array(health_level_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(),
                'cal_level_input': torch.tensor(cal_level_input).long(),
                'health_level_input': torch.tensor(health_level_input).long()
            }
            yield user_batch
    else:
        test_users = dataset.warm_users
        for idx in range(len(test_users)):
            user = test_users[idx]
            pos_item_list = dataset.warmRatings[idx]
            neg_item_list = dataset.warmNegatives[idx]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = pos_item_list + neg_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            cal_level_input = []
            health_level_input = []
            if dataset.args_config['use_cal_level']:
                for item in item_input:
                    cal_level_input.append(dataset.cal_level[item])
                    health_level_input.append(dataset.cal_level[item])
                cal_level_input = np.array(cal_level_input)
                health_level_input = np.array(health_level_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(),
                'cal_level_input': torch.tensor(cal_level_input).long(),
                'health_level_input': torch.tensor(health_level_input).long()
            }
            yield user_batch

def EvalByUserSenseDataloader(dataset, is_sense=True):
    if is_sense:
        test_users = dataset.sense_users
        for idx in range(len(test_users)):
            user = test_users[idx]
            pos_item_list = dataset.senseRatings[idx]
            neg_item_list = dataset.senseNegatives[idx]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = pos_item_list + neg_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            cal_level_input = []
            health_level_input = []
            if dataset.args_config['use_cal_level']:
                for item in item_input:
                    cal_level_input.append(dataset.cal_level[item])
                    health_level_input.append(dataset.cal_level[item])
                cal_level_input = np.array(cal_level_input)
                health_level_input = np.array(health_level_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(),
                'cal_level_input': torch.tensor(cal_level_input).long(),
                'health_level_input': torch.tensor(health_level_input).long()
            }
            yield user_batch
    else:
        test_users = dataset.unsense_users
        for idx in range(len(test_users)):
            user = test_users[idx]
            pos_item_list = dataset.unsenseRatings[idx]
            neg_item_list = dataset.unsenseNegatives[idx]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = pos_item_list + neg_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            cal_level_input = []
            health_level_input = []
            if dataset.args_config['use_cal_level']:
                for item in item_input:
                    cal_level_input.append(dataset.cal_level[item])
                    health_level_input.append(dataset.cal_level[item])
                cal_level_input = np.array(cal_level_input)
                health_level_input = np.array(health_level_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(),
                'cal_level_input': torch.tensor(cal_level_input).long(),
                'health_level_input': torch.tensor(health_level_input).long()
            }
            yield user_batch

def EvalByUserHealthLevelDataloader(dataset, health_level):
    test_users = dataset.healthUsers[health_level]
    rating = dataset.healthRatings[health_level]
    negative = dataset.healthNegatives[health_level]
    for idx in range(len(test_users)):
        user = test_users[idx]
        pos_item_list = rating[idx]
        neg_item_list = negative[idx]
        for pos_item in pos_item_list:
            if pos_item in neg_item_list:
                neg_item_list.remove(pos_item)
        item_input = pos_item_list + neg_item_list
        user_input = np.full(len(item_input), user, dtype='int')
        item_input = np.array(item_input)
        img_input = []
        ingre_input = []
        ingre_num_input = []
        for item in item_input:
            img_input.append(dataset.embImage[item])
            ingre_num_input.append(dataset.ingredientNum[item])
            ingre_input.append(dataset.ingredientCodeDict[item])
        img_input = np.array(img_input)
        ingre_num_input = np.array(ingre_num_input)
        ingre_input = np.array(ingre_input)
        cal_level_input = []
        health_level_input = []
        if dataset.args_config['use_cal_level']:
            for item in item_input:
                cal_level_input.append(dataset.cal_level[item])
                health_level_input.append(dataset.cal_level[item])
            cal_level_input = np.array(cal_level_input)
            health_level_input = np.array(health_level_input)
        user_batch = {
            'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
            'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
            'ingre_input': torch.tensor(ingre_input).long(), 'cal_level_input': torch.tensor(cal_level_input).long(),
            'health_level_input': torch.tensor(health_level_input).long()
        }
        yield user_batch
