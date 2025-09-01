import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.utils.data as data
import random
import pickle


class InteractionData(object):
    """
    Loading the interaction data file
    """

    def __init__(self, args_config):
        self.args_config = args_config
        interaction_path = args_config['interaction_data_path']
        ingre_path = args_config['ingre_data_path']
        self.user_range = []
        self.item_range = []
        self.n_users, self.n_items, self.n_train, self.n_valid, self.n_test, self.inter_num = 0, 0, 0, 0, 0, 0
        self.num_ingredients = 0

        self.trainMatrix = self.load_training_file_as_matrix(interaction_path + 'data.train.rating')
        self.trainList = self.load_training_file_as_list(interaction_path + 'data.train.rating')
        self.testRatings = self.load_training_file_as_list(interaction_path + 'data.test.rating')
        self.testNegatives = self.load_negative_file(interaction_path + 'data.test.negative')
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

        self.validRatings, self.valid_users = self.load_valid_file_as_list(interaction_path + 'data.valid.rating')
        self.validNegatives = self.load_negative_file(interaction_path + 'data.valid.negative')
        assert len(self.validRatings) == len(self.validNegatives)
        self.validTestRatings = self.load_valid_test_file_as_dict(interaction_path + 'data.valid.rating',
                                                                  interaction_path + 'data.test.rating')
        self.cold_list, self.cold_num, self.train_item_list = self.get_cold_start_item_num()

        self.train_data = self.generate_interactions(interaction_path + 'data.train.rating')
        self.valid_data = self.generate_interactions(interaction_path + 'data.valid.rating')
        self.test_data = self.generate_interactions(interaction_path + 'data.test.rating')

        self.train_user_dict, self.valid_user_dict, self.test_user_dict = self.generate_user_dict()

        self.embImage = np.load(interaction_path + 'data_image_features_float.npy')
        self.image_size = self.embImage.shape[1]

        self.embText = np.load(ingre_path + 'data_text_features_t5.npy')
        self.text_size = self.embText.shape[1]

        self.ingredientNum = self.load_id_ingredient_num(ingre_path + 'data_id_ingre_num_file')
        self.ingredientCodeDict = np.load(ingre_path + 'data_ingre_code_file.npy')
        self.num_ingredients = np.max(self.ingredientCodeDict)

        self.statistic_interactions()
        if args_config['interaction_data_path'] != args_config['graph_data_path']:
            coo_matrix_path = args_config['interaction_data_path'] + 'inter_coo_matrix.pkl'
        else:
            coo_matrix_path = args_config['graph_data_path'] + 'inter_coo_matrix.pkl'
        self.train_coo_matrix = self.load_train_coo_matrix(coo_matrix_path)

        if args_config['cold_study']:
            cold_path = args_config['interaction_data_path'] + 'cold_start/'
            self.coldRatings, self.cold_users = self.load_valid_file_as_list(cold_path + 'data.cold.rating')
            self.coldNegatives = self.load_negative_file(cold_path + 'data.cold.negative')
            self.warmRatings, self.warm_users = self.load_valid_file_as_list(cold_path + 'data.warm.rating')
            self.warmNegatives = self.load_negative_file(cold_path + 'data.warm.negative')
            assert len(self.coldRatings) == len(self.coldNegatives) == len(self.cold_users)
            assert len(self.warmRatings) == len(self.warmNegatives) == len(self.warm_users)
        if args_config['sense_study']:
            sense_path = args_config['interaction_data_path'] + 'sense_user/'
            self.senseRatings, self.sense_users = self.load_valid_file_as_list(sense_path + 'data.sense.rating')
            self.senseNegatives = self.load_negative_file(sense_path + 'data.sense.negative')
            self.unsenseRatings, self.unsense_users = self.load_valid_file_as_list(sense_path + 'data.unsense.rating')
            self.unsenseNegatives = self.load_negative_file(sense_path + 'data.unsense.negative')
            assert len(self.senseRatings) == len(self.senseNegatives) == len(self.sense_users)
            assert len(self.unsenseRatings) == len(self.unsenseNegatives) == len(self.unsense_users)
        if args_config['health_level_study']:
            health_level_path = args_config['interaction_data_path'] + 'health_level/'
            self.healthRatings = defaultdict(list)
            self.healthNegatives = defaultdict(list)
            self.healthUsers = defaultdict(list)
            for health_level in range(6):
                self.healthNegatives[health_level] = self.load_negative_file(health_level_path + r'data_health{}.negative'.format(health_level))
                self.healthRatings[health_level], self.healthUsers[health_level] = self.load_valid_file_as_list(health_level_path + r'data_health{}.rating'.format(health_level))
                assert len(self.healthRatings[health_level]) == len(self.healthNegatives[health_level]) == len(self.healthUsers[health_level])

    def load_train_coo_matrix(self, filename):
        with open(filename, 'rb') as f:
            train_interactions = pickle.load(f).astype(np.float32)
        return train_interactions

    def load_valid_test_file_as_dict(self, valid_file, test_file):

        validTestRatings = {}
        for u in range(self.num_users):
            validTestRatings[u] = set()

        fv = open(valid_file, 'r')
        for line in fv:
            arr = line.split('\t')
            u, i = int(arr[0]), int(arr[1])
            validTestRatings[u].add(i)
        fv.close()

        ft = open(test_file, 'r')
        for line in ft:
            arr = line.split('\t')
            u, i = int(arr[0]), int(arr[1])
            validTestRatings[u].add(i)
        ft.close()

        return validTestRatings

    def load_valid_file_as_list(self, filename):
        lists, items, user_list = [], [], []
        with open(filename, 'r') as f:
            line = f.readline()
            index = 0
            last_u = int(line.split('\t')[0])
            while line is not None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                if last_u < u:
                    index = 0
                    lists.append(items)
                    user_list.append(last_u)
                    items = []
                    last_u = u
                index += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        user_list.append(u)
        return lists, user_list

    def load_training_file_as_list(self, filename):
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        return lists

    def load_training_file_as_matrix(self, filename):
        num_users, num_items = 0, 0
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    @staticmethod
    def generate_interactions(filename):
        inter_mat = []
        lines = open(filename, 'r').readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            u_id, pos_id = int(tokens[0]), int(tokens[1])
            inter_mat.append([u_id, pos_id])
        return np.array(inter_mat)

    def generate_user_dict(self):
        def generate_dict(inter_mat):
            user_dict = defaultdict(list)
            for u_id, i_id in inter_mat:
                user_dict[u_id].append(i_id)
            return user_dict

        num_users = max(max(self.train_data[:, 0]), max(self.valid_data[:, 0]), max(self.test_data[:, 0])) + 1

        self.train_data[:, 1] = self.train_data[:, 1] + num_users
        self.valid_data[:, 1] = self.valid_data[:, 1] + num_users
        self.test_data[:, 1] = self.test_data[:, 1] + num_users

        train_user_dict = generate_dict(self.train_data)
        valid_uesr_dict = generate_dict(self.valid_data)
        test_user_dict = generate_dict(self.test_data)

        return train_user_dict, valid_uesr_dict, test_user_dict

    def load_id_ingredient_num(self, filename):
        fr = open(filename, 'r')
        ingredientNumList = []
        for line in fr:
            arr = line.strip().split('\t')
            ingredientNumList.append(int(arr[1]))
        return ingredientNumList

    def statistic_interactions(self):
        def id_range(train_mat, valid_mat, test_mat, idx):
            min_id = min(min(train_mat[:, idx]), min(valid_mat[:, idx]), min(test_mat[:, idx]))
            max_id = max(max(train_mat[:, idx]), max(valid_mat[:, idx]), max(test_mat[:, idx]))
            n_id = max_id - min_id + 1
            return (min_id, max_id), n_id

        self.user_range, self.n_users = id_range(
            self.train_data, self.valid_data, self.test_data, idx=0
        )
        self.item_range, self.n_items = id_range(
            self.train_data, self.valid_data, self.test_data, idx=1
        )
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)
        self.inter_num = self.n_train + self.n_valid + self.n_test

        print("-" * 50)
        print("-     user_range: (%d, %d)" % (self.user_range[0], self.user_range[1]))
        print("-     item_range: (%d, %d)" % (self.item_range[0], self.item_range[1]))
        print("-        n_train: %d" % self.n_train)
        print("-        n_valid: %d" % self.n_valid)
        print("-         n_test: %d" % self.n_test)
        print("-        n_users: %d" % self.n_users)
        print("-        n_items: %d" % self.n_items)
        print("-        n_cold: %d" % self.cold_num)
        print("-        n_ingredients: %d" % self.num_ingredients)
        print("-" * 50)

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def get_cold_start_item_num(self):
        train_item_list = []
        for i_list in self.trainList:
            train_item_list.extend(i_list)
        test_item_list = []
        for r in self.testRatings:
            test_item_list.extend(r)
        valid_item_list = []
        for r in self.validRatings:
            valid_item_list.extend(r)
        c_list = list((set(test_item_list) | set(valid_item_list)) - set(train_item_list))
        t_list = list(set(train_item_list))
        return c_list, len(c_list), t_list


class GraphData(object):
    def __init__(self, args_config):
        self.num_health_level = 0
        self.num_calories_level = 0
        self.args_config = args_config

        self.n_relations = 0
        graph_data_path = args_config['graph_data_path']
        interaction_path = args_config['interaction_data_path']
        ingre_path = args_config['ingre_data_path'] if args_config['small_ingre'] else graph_data_path

        print("-" * 50)
        if args_config['load_UserRecipe_graph']:
            self.uRecipe_triples = self.load_graph_triples(graph_data_path + 'ur_graph.txt')
            self.n_relations += 1
            print("-   n_UserRecipe_triples: %d" % len(self.uRecipe_triples))
        if args_config['load_RecipeRecipe_graph']:
            self.rRecipe_triples = self.load_graph_triples(graph_data_path + 'rr_graph.txt')
            self.n_relations += 1
            print("-   n_RecipeRecipe_triples: %d" % len(self.rRecipe_triples))
        if args_config['load_RecipeIngre_graph']:
            self.rIngre_triples = self.load_graph_triples(ingre_path + 'ri_graph.txt')
            self.n_relations += 1
            print("-   n_RecipeIngre_triples: %d" % len(self.rIngre_triples))
        if args_config['load_IngreIngre_graph']:
            self.iIngre_triples = self.load_graph_triples(graph_data_path + 'ii_graph.txt')
            self.n_relations += 1
            print("-   n_IngreIngre_triples: %d" % len(self.iIngre_triples))
        if args_config['load_RecipeCalories_graph']:
            self.rCalories_triples = self.load_graph_triples(graph_data_path + 'rc_graph.txt')
            self.num_calories_level = max(self.rCalories_triples[:, 1]) + 1
            self.n_relations += 1
            print("-   n_RecipeCalories_triples: %d" % len(self.rCalories_triples))
            print("-   num_calories_level: %d" % self.num_calories_level)
        if args_config['load_RecipeHealth_graph']:
            self.rHealth_triples = self.load_graph_triples(graph_data_path + 'rh_graph.txt')
            self.num_health_level = max(self.rHealth_triples[:, 1]) + 1
            self.n_relations += 1
            print("-   n_RecipeHealth_triples: %d" % len(self.rHealth_triples))
        if args_config['use_cal_level']:
            self.cal_level = self.load_dict(graph_data_path + 'recipe_cal_level_dict.pkl')
        if args_config['use_health_level']:
            self.health_level = self.load_dict(graph_data_path + 'recipe_health_level_dict.pkl')
        if args_config['use_health_level_multi_hot']:
            self.health_level_multi_hot = self.load_dict(graph_data_path + 'recipe_health_level_multi_hot_dict.pkl')
        if args_config['load_RecipeRecipeCo_graph']:
            self.rr_co_triples = np.loadtxt(graph_data_path + 'rr_co_graph.txt')
            self.n_relations += 1
            print("-   n_RecipeRecipeCo_triples: %d" % len(self.rr_co_triples))
        if args_config['load_RecipeRecipeIng_graph']:
            self.rr_ing_triples = np.loadtxt(graph_data_path + 'rr_ing_graph.txt')
            self.n_relations += 1
            print("-   n_RecipeRecipeIng_triples: %d" % len(self.rr_ing_triples))
        if args_config['load_RecipeRecipeHealth_graph']:
            self.rr_health_triples = np.loadtxt(graph_data_path + 'rr_health_graph.txt')
            self.n_relations += 1
            print("-   n_RecipeRecipeHealth_triples: %d" % len(self.rr_health_triples))
        if args_config['load_ImageCluster_graph']:
            self.image_cluster_triples = np.loadtxt(interaction_path + 'cluster/image_cluster_edge.txt')
            self.n_relations += 1
            print("-   n_ImageCluster_triples: %d" % len(self.image_cluster_triples))
        if args_config['load_TextCluster_graph']:
            self.text_cluster_triples = np.loadtxt(interaction_path + 'cluster/text_cluster_edge.txt')
            self.n_relations += 1
            print("-   n_TextCluster_triples: %d" % len(self.text_cluster_triples))
        print("-   n_relations: %d" % self.n_relations)
        print("-" * 50)

    def load_graph_triples(self, path):
        triples = np.loadtxt(path, dtype=np.int_)
        return triples

    def load_dict(self, filename):
        with open(filename, 'rb') as f:
            item_dict = pickle.load(f)
        return item_dict


class FoodData(InteractionData, GraphData):
    def __init__(self, args_config):
        self.args_config = args_config
        InteractionData.__init__(self, args_config=args_config)
        GraphData.__init__(self, args_config=args_config)

    def __str__(self):
        info = [self.args_config['dataset']]

        avg_actions_of_users = self.inter_num / self.n_users
        info.extend(['The number of users: {}'.format(self.n_users),
                     'Average actions of users: {}'.format(avg_actions_of_users)])
        avg_actions_of_items = self.inter_num / self.n_items
        info.extend(['The number of items: {}'.format(self.n_items),
                     'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))

        sparsity = 1 - self.inter_num / self.n_users / self.n_items
        info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
