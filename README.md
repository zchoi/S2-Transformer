# S<sup>2</sup> Transformer for Image Captioning [IJCAI 2022]

<!-- [![](https://img.shields.io/badge/python-3.7.11-orange.svg)](https://www.python.org/)
[![](https://img.shields.io/apm/l/vim-mode.svg)](https://github.com/zchoi/S2-Transformer/blob/main/LICENSE)
[![](https://img.shields.io/badge/Pytorch-1.7.1-red.svg)](https://pytorch.org/) -->

> Official code implementation for the paper [**S<sup>2</sup> Transformer for Imgae Captioning**](https://www.ijcai.org/proceedings/2022/0224.pdf)<br>
> Pengpeng Zeng, Haonan Zhang, Jingkuan Song, and Lianli Gao

<p align="center">
  <img src="framework.png" alt="Relationship-Sensitive Transformer" width="850"/>
</p>

## Table of Contents
- [Environment setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reference and Citation](#reference-and-citation)
- [Acknowledgements](#acknowledgements)

## Environment setup

Clone this repository and create the `m2release` conda environment using the `environment.yml` file:
```
conda env create -f environment.yaml
conda activate m2release
```

Then download spacy data by executing the following command:
```python
python -m spacy download en_core_web_md
```

> [!NOTE]
> Python 3 is required to run our code. If you suffer network problems, please download ```en_core_web_md``` library from [here](https://drive.google.com/file/d/1jf6ecYDzIomaGt3HgOqO_7rEL6oiTjgN/view?usp=sharing), unzip and place it to ```/your/anaconda/path/envs/m2release/lib/python*/site-packages/```


## Data Preparation

* **Annotation**. Download the annotation file [m2_annotations](https://drive.google.com/file/d/12EdMHuwLjHZPAMRJNrt3xSE2AMf7Tz8y/view?usp=sharing) [1]. Extract and put it in the project root directory.
* **Feature**. Download processed image features [ResNeXt-101](https://pan.baidu.com/s/1avz9zaQ7c36XfVFK3ZZZ5w) and [ResNeXt-152](https://pan.baidu.com/s/1avz9zaQ7c36XfVFK3ZZZ5w) features [2] (code `9vtB`), put it in the project root directory.  
> **Update:** Image features on [OneDrive](https://1drv.ms/f/s!Am8OAR5-SGckhdBjE1v6GXpNp4jaTw?e=kgTLn3)
<!-- * **Evaluation**. Download the evaluation tools [here](https://pan.baidu.com/s/1xVZO7t8k4H_l3aEyuA-KXQ). Acess code: jcj6. Extarct and put it in the project root directory. -->

## Training
Run `python train_transformer.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 50) |
| `--workers` | Number of workers, accelerate model training in the xe stage.|
| `--head` | Number of heads (default: 8) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to visual features file (h5py)|
| `--annotation_folder` | Path to annotations |
| `--num_clusters` | Number of pseudo regions |

For example, to train the model, run the following command:
```python
python train_transformer.py --exp_name S2 --batch_size 50 --m 40 --head 8 --features_path /path/to/features --num_clusters 5
```
or just run:
```
bash train.sh
```
> [!NOTE]
> We apply `torch.distributed` to train our model, you can set the `worldSize` in [train_transformer.py]() to determine the number of GPUs for your training.

## Evaluation
### Offline Evaluation.
Run `python test_transformer.py` to evaluate the model using the following arguments:
```
python test_transformer.py --batch_size 10 --features_path /path/to/features --model_path /path/to/saved_transformer_models/ckpt --num_clusters 5
```
  
> [!NOTE]
> We have removed the ```SPICE``` evaluation metric during training because it is time-cost. You can add it when evaluate the model: download this [file](https://drive.google.com/file/d/1vEVsbEFjDstmSvoWhu4UdKaJjX1jJXpR/view?usp=sharing) and put it in ```/path/to/evaluation/```, then uncomment codes in [__init__.py](https://github.com/zchoi/S2-Transformer/blob/master/evaluation/__init__.py).

We provide pretrained model [here](https://drive.google.com/file/d/1Y133r4Wd9ediS1Jqlwc1qtL15vCK_Mik/view?usp=sharing), you will get following results (second row) by evaluating the pretrained model:

| Model 	| B@1 	| B@4 	|        M   	| R 	| C 	| S |
|:---------:	|:-------:	|:-:	|:---------------:	|:--------------------------:	|:-------:	| :-------:|
| Our Paper (ResNext101) 	|     81.1   	| 39.6 	|               29.6 	|              59.1             	|    133.5   	|  23.2|
|   Reproduced Model (ResNext101) 	|   81.2   	| 39.9 	| 29.6 	|   59.1  	|  133.7 	|  23.3|



### Online Evaluation
We also report the performance of our model on the online COCO test server with an ensemble of four S<sup>2</sup> models. The detailed online test code can be obtained in this [repo](https://github.com/zhangxuying1004/RSTNet).

## Reference and Citation
### Reference
[1] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
[2] Xuying Zhang, Xiaoshuai Sun, Yunpeng Luo, Jiayi Ji, Yiyi Zhou, Yongjian Wu, Feiyue
Huang, and Rongrong Ji. Rstnet: Captioning with adaptive attention on visual and non-visual words. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15465–15474, 2021.
### Citation
```bibtex
@inproceedings{S2,
  author    = {Pengpeng Zeng* and
               Haonan Zhang* and
               Jingkuan Song and 
               Lianli Gao},
  title     = {S2 Transformer for Image Captioning},
  booktitle = {IJCAI},
  pages     = {1608--1614}
  year      = {2022}
}
```
## Acknowledgements
Thanks Zhang _et.al_ for releasing the visual features (ResNeXt-101 and ResNeXt-152). Our code implementation is also based on their [repo](https://github.com/zhangxuying1004/RSTNet).   
Thanks for the original annotations prepared by [M<sup>2</sup> Transformer](https://github.com/aimagelab/meshed-memory-transformer), and effective visual representation from [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa).
