# G-MONet

This repository contains the testing code of **G-MONet**, a CNN based solution for SAR Amplitude Despeckling.

if you find it useful and use it for you research, please cite as the following 
```
@ARTICLE{10250969,
  author={Vitale, Sergio and Ferraioli, Giampaolo and Frery, Alejandro C. and Pascazio, Vito and Yue, Dong-Xiao and Xu, Feng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={{SAR} Despeckling Using Multiobjective Neural Network Trained With Generic Statistical Samples}, 
  year={2023},
  volume={61},
  number={},
  pages={1-12},
  doi={10.1109/TGRS.2023.3314857}}
```


GGCS-MONet inglobe the benefit of a multi-objective architecture of [MONet](https://ieeexplore.ieee.org/document/9261137) and a specific training strategy based on the [GGCS simulator](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8944059)

The 17 layers CNN architecture of MONet is trained in a supervised fashion with a training dataset specifically designed for including real properties or real SAR images.
Starting from a multi-temporal stack of real SAR images, the GGCS simulator is used for extracting statistical information from the data. Such features are used for simualting realistic speckle noise distribution intirinsic in the real data. This allow to overcome the limitation of fully developed hypothesis highlighted in [previous analysis](https://github.com/impress-parthenope/Analysis-on-the-Building-of-Training-Datatset-for-Deep-Learning-SAR-Despeckling)

_Architecture of **MONet**_

<p align="center">
  <img src="https://github.com/impress-parthenope/GGCS-MONet/assets/85936968/fc4c026f-27f0-4a9f-8a4d-112d1f7fd064.png"
  width=500>
<p>
  
_Procedure for **constructing realistic training dataset** with GGCS simulator and real SAR images_

<p align="center">
<img src="https://github.com/impress-parthenope/GGCS-MONet/assets/85936968/87f51ed6-44f0-4923-b38f-3e7506430703.png" height=400>
<p>


# Team members
 Sergio Vitale    (contact person, sergio.vitale@uniparthenope.it);
 Dong-Xiao Yue (yue_dong_xiao@163.com)
 Giampaolo Ferraioli (giampaolo.ferraioli@uniparthenope.it);
 Alejandro Frery (alejandro.frery@vuw.ac.nz);
 Feng Xu (fengxu@fudan.edu.cn);
 Vito Pascazio (vito.pascazio@uniparthenope.it);
 
# License
Copyright (c) 2023 Dipartimento di Ingegneria and Dipartimento di Scienze e Tecnologie of Università degli Studi di Napoli "Parthenope".

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)

# Usage 
* *model* contains trained weigths
* *model.py* contains the model implementation
* *testing.py* is the main script for testing

# Prerequisites
This code is written on Ubuntu system for Python3.7 and uses Pytorch library.
- python=3.7
- pythorc=3.9.1
- cuda = 10.2
  
For a correct usage of the code, please install the python environement saved in **./env/monet_pytorch.yml** with the following step:

**Installing Anaconda** (if not already installed)

1. download anaconda3 from https://www.anaconda.com/products/individual#linux
2. from command line, move to the download directory and install the package by:
> sh <Anaconda_downloaded_version>.sh 
and follow the instruction for installation
3. add conda to path
> PATH=~/anaconda3/bin:$PATH

**Installing the conda environment**

The file ./insarmonet_env.yml contains the environemnt for the testing the code. You can easily installing it by command line:

1. move to the folder containing the github repository and open the terminal
2. run the following command
 > conda env create -f insarmonet_env.yml


Once environment has been set up, activate it by command line as well:

1. activate the environemnt from the command line

> conda activate insarmonet_env

2. launch spyder

> spyder

3. goes to the folder containing **testing.py**, edit and run



