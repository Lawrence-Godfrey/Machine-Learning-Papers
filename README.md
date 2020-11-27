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
## 3D Convolutions 
## Generative Adveserial Networks 
