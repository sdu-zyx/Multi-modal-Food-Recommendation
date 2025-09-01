# Multi-modal Food Recommendation

This is the PyTorch implementation of HealthRec and CLUSSL described in the papers:

>Yixin Zhang, Xin Zhou, Fanglin Zhu, Ning Liu, Wei Guo, Yonghui Xu, Zhiqi Shen, and Lizhen Cui. [Multi-modal food recommendation with health-aware knowledge distillation.](https://dl.acm.org/doi/abs/10.1145/3627673.3679580) In CIKM 2024.

>Yixin Zhang, Xin Zhou, Qianwen Meng, Fanglin Zhu, Yonghui Xu, Zhiqi Shen, Lizhen Cui. [Multi-modal Food Recommendation Using Clustering and Self-supervised Learning.](https://link.springer.com/chapter/10.1007/978-981-96-0116-5_22) In PRICAI 2024.

## Dataset 
Please download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1xCKlKpTnHpiNwjTNLT1yIDB-Srwsv0yz?usp=sharing).

### Dataset Preprocessing
If you have downloaded the processed datasets, you can directly use them for reproduction and further experiments.
If you want to know the details of data preprocessing, please see the instructions below.
#### Download and process raw datasets
Please download the raw datasets from the original website.

For the Allrecipes dataset, please download from [https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1/data](https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1/data). And then execute [allrecipes_process.ipynb](..%2Fdataset_process%2Fallrecipes_process.ipynb) to process data.

For the Foodcom dataset, please download from [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data). And execute [foodcom_process.ipynb](..%2Fdataset_process%2Ffoodcom_process.ipynb) to process data.

For the image of Foodcom dataset, please download using [download_image.py](..%2Fdataset_process%2Fdownload_image.py), and [download_check.py](..%2Fdataset_process%2Fdownload_check.py).

Only for CLUSSL model, Construct Bipartite Graph via Continuous Features, execute [allrecipes_kmeans.ipynb](..%2Fdataset_process%2Fallrecipes_kmeans.ipynb) and [foodcom_kmeans.ipynb](..%2Fdataset_process%2Ffoodcom_kmeans.ipynb) to obtain modality-specific graph edges.

### Dataset path
Change the data_path according to your absolute path in configs/overall.yaml

And change the below code in utils/quick_start.py 

```
    config['interaction_data_path'] = config['data_path'] + dataset + '/processed_dataset/'
    config['graph_data_path'] = config['data_path'] + dataset + '/processed_dataset/graph_edge/'
    config['ingre_data_path'] = config['data_path'] + dataset + '/processed_dataset/'
```

## Run the code

Make sure all dependencies package are installed

```
python runner.py
```

## Build your model

Build your model like models/cikm_model.py or models/schgn.py
And use yaml files to configure your model parameters，like configs/cikm_model.yaml or configs/SCHGN.yaml


## Acknowledgement
The implementation is based on the open-source recommendation library [MMRec](https://github.com/enoche/MMRec).

If you find our work useful in your research, please consider citing the paper:

```
@inproceedings{HealthRec,
  title={Multi-modal food recommendation with health-aware knowledge distillation},
  author={Zhang, Yixin and Zhou, Xin and Zhu, Fanglin and Liu, Ning and Guo, Wei and Xu, Yonghui and Shen, Zhiqi and Cui, Lizhen},
  booktitle={Proceedings of the 33rd ACM international conference on information and knowledge management},
  pages={3279--3289},
  year={2024}
}
```

```
@inproceedings{CLUSSL,
  title={Multi-modal food recommendation using clustering and self-supervised learning},
  author={Zhang, Yixin and Zhou, Xin and Meng, Qianwen and Zhu, Fanglin and Xu, Yonghui and Shen, Zhiqi and Cui, Lizhen},
  booktitle={Pacific Rim International Conference on Artificial Intelligence},
  pages={269--281},
  year={2024},
  organization={Springer}
}
```
