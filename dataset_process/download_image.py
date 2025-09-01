# coding: utf-8

# 1. 加入下载超时退出，要不一直卡死 stuck
# 2. 多线程下载
import pickle
import os
import pandas as pd
import urllib.request
import ast
from itertools import product

import requests
from bs4 import BeautifulSoup

Dir = r'/home/zyx2509/Food_dataset/Foodcom/'
image_dir = 'image_dataset'
K = 60

os.chdir(Dir)
# Working dir
print(f'Current Dir: {os.getcwd()}')
dst_dir = Dir + image_dir
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)

####################### Get files
#meta_file = 'meta-games.csv'

with open(Dir+'processed_dataset/mapping_dict.pkl', 'rb') as f:
    _, item_to_idx, _ = pickle.load(f)

items = item_to_idx.keys()

raw_recipe = pd.read_csv(Dir+'raw_dataset/RAW_recipes.csv')
target_items = raw_recipe[raw_recipe['id'].isin(items)]

oris = target_items.shape
print(f'Original shape: {oris}')
id, name = 'id', 'name'
# Change to current dir
os.chdir(Dir+image_dir)
print(f'Changed to Dir: {os.getcwd()}')
print('==============')


def format_image_url(id, name, url):
    name = name.replace(" ", "-")
    url_ = url + name + '-' + str(id)
    return url_


# 实际下载线程
error_image_id = []
def dld_images(i, row):

    id = row['id']
    name = row['name']
    url = 'https://www.food.com/recipe/'
    url = format_image_url(id, name, url)
    header = {
        'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
    img_filename = os.path.join(dst_dir + f'/{id}.jpg')
    if not os.path.exists(img_filename):
        try:
            # 发送HTTP请求获取网页内容
            response = requests.get(url, headers=header)
        except Exception as e:
            print(f'{id}-image request content, {id}, error: {e}')
            error_image_id.append(id)
            return
        html_content = response.text

        # 使用Beautiful Soup解析HTML内容
        soup = BeautifulSoup(html_content, 'html.parser')

        current_recipe_div = soup.find('div', class_='primary-image')

        # 找到所有图片标签
        img_tag = current_recipe_div.find_all('img')[0]

        img_url = img_tag['src']
        if not img_url.endswith('.jpg'):
            print(f'{id}-image format error')
            error_image_id.append(id)
            return
        # 下载图片
        try:
            # 发送HTTP请求获取网页内容
            img_response = requests.get(img_url, headers=header)
        except Exception as e:
            print(f'{id}-image request image, {id}, error: {e}')
            error_image_id.append(id)
            return

        # 保存图片到本地
        with open(img_filename, 'wb') as f:
            f.write(img_response.content)
            print(f"Downloaded {img_filename}")
    else:
        print(f'{id}-image文件已存在，跳过下载。')




############### MAIN
import multiprocessing as mp
import socket

target_items.reset_index(drop=True, inplace=True)

# Set the default timeout in seconds
timeout = 45
socket.setdefaulttimeout(timeout)

# download images
pool = mp.Pool(processes=K)
res = pool.starmap(dld_images, target_items.iterrows())
print(f'\n============\n======Finished=======')

