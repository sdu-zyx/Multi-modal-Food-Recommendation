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

file_names = os.listdir(dst_dir)

finish_image = [int(os.path.splitext(file_name)[0]) for file_name in file_names]

unfinish = [x for x in items if x not in finish_image]
print(len(finish_image))
print(len(unfinish))

no_image_list = []

# 打开文本文件以读取模式
with open(Dir+'processed_dataset/no_image.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 去除换行符并将 id 添加到列表中
        no_image_list.append(int(line.strip()))

# 打印 id 列表
print(len(no_image_list))

new_image = [x for x in unfinish if x not in no_image_list]
print(len(new_image))

raw_recipe = pd.read_csv(Dir+'raw_dataset/RAW_recipes.csv')
target_items = raw_recipe[raw_recipe['id'].isin(new_image)]

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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
    img_filename = os.path.join(dst_dir + f'/{id}.jpg')
    if not os.path.exists(img_filename):
        try:
            # 发送HTTP请求获取网页内容
            response = requests.get(url, headers=headers)
        except Exception as e:
            print(f'{id}-image request content, {id}, error: {e}')
            error_image_id.append(id)
            return
        html_content = response.text

        # 使用Beautiful Soup解析HTML内容
        soup = BeautifulSoup(html_content, 'html.parser')

        current_recipe_div = soup.find('div', class_='primary-image')

        # 找到所有图片标签
        try:
            img_tag = current_recipe_div.find_all('img')[0]
        except Exception as e:
            print(url, e)
            return

        img_url = img_tag['src']
        if "recipe-default-images" in img_url:
            print(f'{id}-image is default-images')
            error_image_id.append(id)
            with open(Dir+'preprocess_5-core/no_image.txt', 'a') as no_image_file:
                no_image_file.write(str(id)+'\n')
            return
        # 下载图片
        try:
            # 发送HTTP请求获取网页内容
            img_response = requests.get(img_url, headers=headers)
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
print(error_image_id)
print(len(error_image_id))

print(f'\n============\n======Finished=======')

