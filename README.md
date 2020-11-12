# Machine Learning Papers

## Convolutional Neural Networks
[Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
 * Trained on ImageNet 2010
 * Novel Features: 
     * ReLu
     * Trained on multiple GPUs
     * Local Response Normalization
     * Overlapping Pooling
         * s=2 z=3
         * Harder to overfit
         
 * maximizes the multinomial logistic regression objective

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
 * Introduces VGG16 and VGG19
 * Does not use Local Response Normalisation (LRN) normalisation
     * They found that it doesn’t increase performance, but uses more memory
 * Shows how increasing depth of network increases performance 
     * 11 to 19 layer networks
 * Uses smaller 3×3 filters with stride of 1 compared to much larger (7x7 and 11x11) filters previously used. 
 * Stacking 3 3x3 filters is almost the same as a single 7x7 filter, however, stacking filters also adds more ReLu activations between them, which makes the decision function more discriminative 
     * Also, a 3 stacked 3x3 filter uses 3C2  = 27C2 weight, while a 7x7 layer uses 72C2 = 49C2 weights
* One architecture uses 1x1 filter to increase non-linearity without affecting anything else 



## Fully Convolutional Networks 
## 3D Convolutions 
## Generative Adveserial Networks 
