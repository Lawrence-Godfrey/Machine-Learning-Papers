# Machine Learning Papers

## Convolutional Neural Networks
#### [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
By Yann LeCun, Leon Bottou, Yoshua Bengio and Patrick Haffner
 *  Reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task.
 
#### [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
By Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
 * Trained on ImageNet 2010
 * Novel Features: 
     * ReLu
     * Trained on multiple GPUs
     * Local Response Normalization
     * Overlapping Pooling
         * s=2 z=3
         * Harder to overfit
         
 * maximizes the multinomial logistic regression objective

#### [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
By Karen Simonyan and Andrew Zisserman
 * Introduces VGG16 and VGG19
 * Does not use Local Response Normalisation (LRN) normalisation
     * They found that it doesn’t increase performance, but uses more memory
 * Shows how increasing depth of network increases performance 
     * 11 to 19 layer networks
 * Uses smaller 3×3 filters with stride of 1 compared to much larger (7x7 and 11x11) filters previously used. 
 * Stacking 3 3x3 filters is almost the same as a single 7x7 filter, however, stacking filters also adds more ReLu activations between them, which makes the decision function more discriminative 
     * Also, a 3 stacked 3x3 filter uses 3C2  = 27C2 weight, while a 7x7 layer uses 72C2 = 49C2 weights
* One architecture uses 1x1 filter to increase non-linearity without affecting anything else 

#### [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
By Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun
 * Introduces “Shortcut” connections which skip layers and add activation of earlier layer to later layer. 
 * Reduces vanishing gradrient effect in larger networks
 
#### [Network In Network](https://arxiv.org/pdf/1312.4400.pdf)
By Min Lin, Qiang Chen and Shuicheng Yan
 * Adding Multi-Layer Perceptrons instead of traditional kernels between layers to increase model discriminability. 
 
#### [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
By Szegedy et al.
 * InceptionNet Architecture, which uses Inception modules. 
 * Similar to NIN, Inception modules use a number of kernels with different kernel sizes inbetween layers, adding their outputs together at the end. 
 * This allows for multiple receptive field sizes, which can increase accuracy since the important objects in an image might differ from image to image. 

#### [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)
By Mingxing Tan and Quoc V. Le
 * Efficient scaling of width, depth and resolution to maximize performance. 
 * Intuitively, only scaling one of these dimensions doesn't work: Increasing depth increases the receptive field, which requires a higher resolution input, which requires more parameters. 

#### [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)
By Sermanet et al.

#### [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
By Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi

#### [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
By Ross Girshick, Jeff Donahue, Trevor Darrell and Jitendra Malik
 * Introduces Region based CNN (R-CNN)

#### [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
By Ross Girshick

#### [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
By Shaoqing Ren, Kaiming He, Ross Girshick and Jian Sun

#### [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)
By Matthew D. Zeiler and Rob Fergus 
 * Projecting feature activations back to input pixel space to understand what causes activation
     * Using Deconvolutional Network (deconvnet)
 * Occluding certain parts of input and seeing how output changes
 * Viewing feature maps is common practice, but only works on the first few layers.

#### [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
By Leon A. Gatys, Alexander S. Ecker and Matthias Bethge

#### [Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
By Paul Viola and Michael Jones





## Fully Convolutional Networks 

#### [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
By Jonathan Long, Evan Shelhamer and Trevor Darrell

#### [Review and Evaluation of Deep Learning Architectures for Efficient Land Cover Mapping with UAS Hyper-Spatial Imagery: A Case Study Over a Wetland](https://www.mdpi.com/2072-4292/12/6/959/pdf)
By Mohammad Pashaei, Hamid Kamangir, Michael J. Starek and Philippe Tissot
* Review of Encoder-Decoder Semantic Segmentation Models

#### [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
By Augustus Odena, Vincent Dumoulin, Chris Olah


## 3D Convolutions 
#### [An Efficient 3D CNN for Action/Object Segmentation in Video](https://arxiv.org/pdf/1907.08895.pdf)
By Rui Hou, Chen Chen, Rahul Sukthankar and Mubarak Shah

#### [An End-to-end 3D Convolutional Neural Network for Action Detection and Segmentation in Videos](https://arxiv.org/pdf/1712.01111.pdf)
By Rui Hou, Rui Hou and Mubarak Shah

#### [Semantic Video Segmentation: A Review on Recent Approaches](https://arxiv.org/ftp/arxiv/papers/1806/1806.06172.pdf)
By Mohammad Hajizadeh Saffar, Mohsen Fayyaz, Mohammad Sabokrou and Mahmood Fathy

#### [STD2P: RGBD Semantic Segmentation using Spatio-Temporal Data-Driven Pooling](https://scalable.mpi-inf.mpg.de/files/2017/04/cvpr2017.pdf)
By Yang He, Wei-Chen Chiu, Margret Keuper and Mario Fritz

#### [Semi-CNN Architecture for Effective Spatio-Temporal Learning in Action Recognition](https://www.researchgate.net/publication/338552250_Semi-CNN_Architecture_for_Effective_Spatio-Temporal_Learning_in_Action_Recognition)
By Mei Chee Leong, Dilip K. Prasad, Dilip K. Prasad, Yong Tsui Lee, Yong Tsui Lee and Feng Lin

#### [Learning Spatiotemporal Features with 3D Convolutional Networks](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)
By Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani and Manohar Paluri
 * C3D Architecture 
 * https://github.com/facebookarchive/C3D

#### [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650.pdf)
By Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger

## Generative Adveserial Networks 
#### [Hyperspherical Variational Auto-Encoders](https://www.researchgate.net/publication/324182043_Hyperspherical_Variational_Auto-Encoders)
By Tim R. Davidson, Luca Falorsi, Nicola De Cao
 * VAE, not GAN, but also for generating data

#### [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf)
By Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila

#### [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/pdf/1903.07291.pdf)
By Taesung Park, Ming-Yu Liu, Ting-Chun Wang and Jun-Yan Zhu

#### [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/pdf/1905.08233.pdf)
By Egor Zakharov, Aliaksandra Shysheya, Egor Burkova and Victor Lempitsky

#### [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/abs/1610.07584)
By Wu, Zhang, Xue, Freeman, and Tenenbaum

#### [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
By Brock, Donahue, and Simonyan

#### [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
By Alec Radford, Luke Metz and Soumith Chintala


## GPU Acceleration 
#### [cuDNN: Efficient Primitives for Deep Learning](https://arxiv.org/pdf/1410.0759.pdf)
