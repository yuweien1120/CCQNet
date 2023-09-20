# A Quadratic Network-based Class-weighted Supervised Contrastive Learning framework to Long-tailed Bearing Fault Diagnosis

This is the official repository of the paper "A Quadratic Network-based Class-weighted Supervised Contrastive Learning framework to Long-tailed Bearing Fault Diagnosis".

In this work,

1. We introduce the class-weighted contrastive learning quadratic network (CCQNet) for long-tailed bearing fault diagnosis. Our approach utilizes supervised contrastive learning and quadratic neurons to enhance the model's feature extraction capability and overcome the challenges of long-tailed bearing data through re-balanced loss functions. By combining these techniques, our model demonstrates superior performance in handling long-tailed data compared to other state-of-the-art methods. 
2. Methodologically, we have demonstrated the superiority of quadratic networks in signal feature representation by deriving and establishing the connection between autocorrelation and quadratic neurons.

All experiments are conducted with Linux-5.4.0 on an Intel Platinum 8255C CPU at 2.50GHz and one RTX 3080 10GB GPU. We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  

## CCQNet: Class-weighted Contrastive Learning Quadratic Network

The CCQNet consists of a quadratic convolutional residual network backbone, a class-weighted contrastive learning branch, and a classifier branch employing logit-adjusted cross-entropy loss. 

![](https://raw.githubusercontent.com/yuweien1120/readme-img/main/framework_00.png)

### Quadratic Residual Network

We employ a quadratic residual network as the backbone of the neural network because a quadratic network exhibits a superior feature extraction ability compared with conventional neural network. 

A quadratic neuron was proposed by [1], It computes two inner products  and  one  power  term  of  the  input  vector  and  integrates them for a nonlinear activation function. The output function of a quadratic neuron is expressed as 

![](https://raw.githubusercontent.com/yuweien1120/readme-img/main/quadratic%20neuron1.png)

where $\ast$ denotes the convolutional operation, $\odot$ denotes Hadamard product, $\boldsymbol w^r \in \mathbb{R}^{k\times 1}$, $\boldsymbol w^g \in \mathbb{R}^{k\times 1}$ and $\boldsymbol w^b \in \mathbb{R}^{k\times 1}$ denote the weights of three different convolution kernels, $\sigma(\cdot)$ is the activation function (e.g., ReLU), ${b}^r$, ${b}^g$ and ${c}$ denote biases corresponding to these convolution kernels. 

We construct a cross-layer connection strategy for the quadratic convolutional network to improve its stability. We employ two residual blocks, referred to as QResBlocks, each composed of two Qlayers. 

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/qresblock.png" style="zoom: 25%;" />

The structural parameters of the quadratic residual network backbone are presented as follows.

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/structural%20parameters.png" style="zoom: 50%;" />



###  Class-weighted Contrastive Learning

Inspired by the reweighting technique, we integrate this weight into the SCL loss function [2], and we thus have the class-weighted contrastive loss (CRCL) as:

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/CRCL.png" style="zoom: 33%;" />

Compared to the original SCL loss function, for each $z_a$, we assign a weight $ W_a = \frac{1}{|\boldsymbol{P}_a|}$ for class $a$, $|\boldsymbol{P}_a|$ indicates the number of samples belonging to class $a$.

### Classifier Learning

we use the logit adjusted cross-entropy loss function $\mathcal{L}^{LC}$ to drive the training of the classifier [3].

 <img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/LC.png" style="zoom: 60%;" />

where $j \in J \equiv \{1, 2, \cdots, n\}$ are the indices of the raw data ${\boldsymbol x_1, \boldsymbol x_2, \cdots, \boldsymbol x_N}$, $f(\boldsymbol x_j)$ denotes the output of the classifier, i.e. logit. $f_{y_j}(\boldsymbol x_j)$ denotes the value of the element in the logit vector classified as label $y_j$. Let $[L]$ be a collection of labels $y'$ and $y' \in [L] \equiv \{1, 2, \cdots, L\}$, $\pi_{y_j}$ denotes the prior probability of the label $y_j$, and $\tau$ indicates temperature coefficient which is set to $\tau = 1$ in the developed approach.

###  The superiority of quadratic networks

We find the connection between the quadratic convolution operation and signal autocorrelation. Mathematically, the quadratic convolutional operation can be decomposed into two parts: the sum of learnable autocorrelation and the conventional convolutional operation.

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/Qx.png" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/qj.png" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/qj1.png" style="zoom:67%;" />

With the above decomposition, a quadratic network offers advantages over conventional neural networks when processing signals. The autocorrelation operation within a quadratic neuron aids in extracting valuable signals with random noise, while such capability is missing in conventional neural networks. The operations of the quadratic neuron are shown as follows.

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/q_operation.png" style="zoom: 67%;" />

## Repository organization

### Requirements

We use PyCharm 2022.3 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.

* Python == 3.8
* PyTorch == 1.13.1
* CUDA == 11.7 if use GPU
* wandb == 0.13.5
* anaconda == 2022.10

### Organization

```
CCQNet
│   augmentation.py # data augmentation related functions
|   loss_func.py # loss function of CCQNet
|   main.py # training CCQNet
|   make_dataloader.py # making long-tailed dataloader
│   processing.py # spliting dataset to long-tailed train set, valid set and test set, then processing data
|   test.py # calculating the metrics of CCQNet on the test set
|   train_function.py # Train function for quadratic network
|   trainer.py # trainer used for training CCQNet
└─  utils
     │   metric.py # functions for calculating metrics
└─  Model
     │   ConvQuadraticOperation.py # the quadratic convolutional neuron function 
     │   QResNet.py # QResNet model
     │   CCQNet.py # CCQNet model
```

### Datasets

We use the CWRU dataset [4] and HIT dataset in our article. The CWRU dataset is a public bearing fault dataset  that can be found in [CWRU Dataset](https://github.com/s-whynot/CWRU-dataset).

### How to Use

Our deep learning models use **Weight & Bias** for training and fine-tuning. It is the machine learning platform for developers to build better models faster. [Here](https://docs.wandb.ai/quickstart) is a quick start for Weight & Bias. You need to create an account and install the CLI and Python library for interacting with the Weights and Biases API:

```
pip install wandb
```

Then login 

```
wandb login
```

After that, you can run our code for training.

1. For long-tailed dataset, you need to generate train dataset, valid dataset and test dataset by running ```make_dataloader.py```.

2. Run ```main.py``` to train CCQNet model. First, you need to fill in your username in the wandb initialization function:

   ```
   wandb.init(project="CCQNet", entity="username")
   ```

   The results will be saved to your **Weight & Bias** project online.  

   

## Main Results

### Classification Performance

Here we give the main results of our paper. We use accuracy (ACC), F1 score, and MCC (Matthews Correlation Coefficient) to validate the performance of the proposed method. All resutls are run 10 times to calculate the average. The proposed method outperforms other compared baseline methods.

![](https://raw.githubusercontent.com/yuweien1120/readme-img/main/hit_perform.png)

###  Feature map visualization

We compared feature maps of our quadratic residual network and a ResNet backbone with the same structure. Both networks preserve local features of high amplitude in the input signal caused by fault-induced vibrations. However, as the network gets deeper, the quadratic network places more emphasis on these features, outperforming the conventional network.

<img src="https://raw.githubusercontent.com/yuweien1120/readme-img/main/cmp_backbone.png" style="zoom: 67%;" />

### Learnable autocorrelation

We visualized the autocorrelation operation term that weights set as 1 and the learnable autocorrelation term in the first layer of a quadratic network. The results indicate that while the autocorrelation operation can effectively extract fault-related signals from the noise, it also amplifies certain parts of extraneous noise. This limitation arises because autocorrelation is primarily designed to enhance transient impulses in the signal. In contrast, the learnable autocorrelation term demonstrates a remarkable ability to extract fault-related signals. The weights learned by the neural network act as adaptive filters, which further enhances the feature extraction capability.

![](https://raw.githubusercontent.com/yuweien1120/readme-img/main/Learnable_autocorr.png)

## Contact

If you have any questions about our work, please contact the following email address:

22s001048@stu.hit.edu.cn

Enjoy your coding!

## Reference

[1] Fenglei Fan, Wenxiang Cong, and Ge Wang. A new type of neurons for machine learning. International journal for numericalmethods in biomedical engineering, 34(2):e2920, 2018.

[2] Khosla, Prannay, et al. "Supervised contrastive learning." *Advances in neural information processing systems* 33 (2020): 18661-18673.

[3] Menon, Aditya Krishna, et al. "Long-tail learning via logit adjustment." *arXiv preprint arXiv:2007.07314* (2020).

[4] https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
