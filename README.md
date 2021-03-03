# Passive-None-Line-of-Sight-Recognition

# About
Code for the paper 'Passive None-Line-of-Sight Recognition: Accurate but Fragile'.<br>
Implementation is done in Pytorch 1.1 and runs with 3.6. This repository contains implementation code for CNN NLOS Recognition and using white-box attacks to verify its robustness.<br>
Paper: [地址]<br>
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
Note that: You should modify the parameter settings according to your experimental system.
