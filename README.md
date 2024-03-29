# Passive-None-Line-of-Sight-Recognition

# About
Code for the paper 'Accurate but Fragile Passive None-Line-of-Sight Recognition'.<br>
Implementation is done in Pytorch 1.1 and runs with 3.6. This repository contains implementation code for CNN NLOS Recognition and using white-box attacks to verify its robustness.<br>
Paper: [地址]<br>
The datasets generated during and/or analyzed during the current study are available at https://pan.baidu.com/s/13pAX1el0fH-lKhpqanSFfw the password of “iy7d”.
<br>
![二维码](https://user-images.githubusercontent.com/52912822/198873284-36e596b9-a35e-41f0-adfa-a3b7b6991799.png)<br>
Prepare dataset<br>
The files of the data folder are used to reproducing results in the manuscript (and its supplementary information).<br>
Here we demonstrate how to prepare data for NLOS Recognition.<br>
1. How to get simulation data<br>
Assume that you have already through below urls downloaded the Mnist data to Passive \data\raw.<br>
urls = [<br>
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',<br>
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',<br>
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',<br><br>
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',<br>
    ]<br>
Assumption that the prior information of the setting (supplementary information, Table S2) has known, then refer to https://github.com/Computational-Periscopy/Ordinary-Camera Ordinary-Camera/Functions/simulate_A.py to compute light transport matrix A.<br>
2. How to get reconstruction data<br>
Assume that camera measurement have obtained, refer to https://github.com/Computational-Periscopy/Ordinary-Camera  script 'fig4_column_c.m', then you can got reconstruction data.<br>
Note that: You should modify the parameter settings according to your experimental system.<br>
# Reproduction
If you have any questions about the reproduction of the article, please contact Yaqin Zhang. E-mail number: 493834755@qq.com or zhangyaqin202102@163.com.
