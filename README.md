# Bokeh effect 学习与研究

基于SBTNet进行了研究学习，结合DPT（Midas 3.0）对模型进行了改进。

## SBTNet源码地址，下载pretrained model，请放到`checkpoints`目录下

https://github.com/JuewenPeng/SBTNet

## Intel lab DPT源码地址，下载pretrained model，请放到`model_cache`目录下

https://github.com/intel-isl/DPT


## 运行SBTNet模型的命令
```
python evaluation.py --root_folder TEST_ROOT_FOLDER --save_folder SAVE_FOLDER
```
## 运行改进模型的命令，光圈根据实际修改调整
```
python demo.py --src_f 16.0 --tgt_f 1.8
```

## 以下为SBTNet的原始README.MD
```
*************************************************************************************
```


# SBTNet: Selective Bokeh Effect Transformation

Our solution in competition NTIRE 2023 Bokeh Effect Transformation: https://codalab.lisn.upsaclay.fr/competitions/10229.

<!-- ## News -->
<!-- - *2023/3/29:* Update the test results and the pretrained model (**disable AlphaNet while testing real-world images**). -->

## Test Results
Download the test results from [Google Drive](https://drive.google.com/drive/folders/1ZTwTKC-NOEPne38cWRrrzweHBbZNItFB?usp=share_link).

## Usage
Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1ZTwTKC-NOEPne38cWRrrzweHBbZNItFB?usp=share_link), and place it in the folder `checkpoints`. 
Run the following code to generate test results.
```
python evaluation.py --root_folder 'TEST_ROOT_FOLDER' --save_folder 'SAVE_FOLDER'
```
- `root_folder`:  root folder of the test dataset.
- `save_folder`: folder to save the results.

## Citation
If you find our work useful in your research, please cite our paper.
```
@inproceedings{Peng2023Selective,
  title = {Selective Bokeh Effect Transformation},
  author = {Peng, Juewen and Pan, Zhiyu and Liu, Chengxin and Luo, Xianrui and Sun, Huiqiang and Shen, Liao and Xian, Ke and Cao, Zhiguo},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year = {2023}
}
```
