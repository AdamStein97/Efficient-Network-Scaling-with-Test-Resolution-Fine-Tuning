# Efficient Network Scaling with Image Test Resolution Fine Tuning

A project designed to explore how to gain optimum performance in computer vision based problems
when resources are limited. The explored use case is an image classifier scaled up using the algorithm outlined in the 
recent EfficientNet paper [1] with post-training fine tuning to fix the train-test resolution discrepancy and 
dramatically improve performance [2].

## Motivation

Neural Networks on computer vision problems can often take an extremely long time to
train. This is because convolution is a relatively slow operation especially as the 
resolution of our images increase. The solution is to do limited hyper-parameter tuning 
when designing large CNNs and to take low resolution crops of images to reduce 
dimensions. Both of these steps lead to non-optimal designed models that do not 
utilise all information available to them at test time.

The motivation behind this project is to demonstrate how to efficiently design large CNNs 
with limited hyper-parameter training and effectively use higher resolution images at test
time than we can afford to train with. The techniques used to achieve these two aims are 
taken from two 2019 papers that achieved high commendation from leading CV conferences.

The example use case to demonstrate these techniques will be an image classification 
problem of predicting if an image is either a cat or a dog.

## Data

The dataset used for training the models will be the "cats_vs_dogs" taken from the 
Tensorflow datasets catalogue. The dataset contains 23,000 images of varying size; 4,000
of these images are taken to be our test set. Example images from this dataset can be found
in the `example_images` directory. 

### Augmentation

To improve model performance, images are augmented with random flipping along with alterations 
to brightness and saturation. In addition to this, random crops are taken. The proportion 
of the image placed within the crop is uniformly sampled between 0.2 and 0.7. All crops are
square and the resolution of this crop is then scaled to 64x64. If the aspect ratio is too wide
to create a square crop with the samples proportion then the largest possible square is taken. 
Images are never padded.

## Background

### Network Scaling

Tan et al. proposed a framework of efficiently scaling up CNNs by grid searching scaling
parameters on small models and using these to extrapolate to very large models [1]. This 
framework allowed them to design EfficientNet which could gain similar performance to 
ResNet [3] models but with roughly 6x less latency.

A baseline architecture is designed and then we perform a grid search on a scaling parameter
for depth, width (# channels) and resolution referred to <img src="https://render.githubusercontent.com/render/math?math=\alpha">,
<img src="https://render.githubusercontent.com/render/math?math=\beta"> and <img src="https://render.githubusercontent.com/render/math?math=\gamma"> 
respectively. 

Such that:
<img src="https://render.githubusercontent.com/render/math?math=\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \alpha \geq 1, \beta \geq 1,\gamma \geq 1">

Once we have optimum values for <img src="https://render.githubusercontent.com/render/math?math=\alpha, \beta,\gamma"> 
we can scale up our network uniformly with a selected compound coefficient <img src="https://render.githubusercontent.com/render/math?math=\phi"> where:

depth = <img src="https://render.githubusercontent.com/render/math?math=\alpha^\phi"> 

width = <img src="https://render.githubusercontent.com/render/math?math=\beta^\phi"> 

resolution = <img src="https://render.githubusercontent.com/render/math?math=\gamma^\phi"> 

The total flops of the network will approximately increase by <img src="https://render.githubusercontent.com/render/math?math=2^\phi">. 
Performing the grid search on a small baseline network is considerably quicker than hyper-parameter tuning on the search
space for large CNNs. This leads to a far better use of resources as illustrated by the results in the paper.

Tan et al. derived their baseline model using a reinforcement learning approach but this
was seen to be too computationally expensive for this project. As an alternative a few 
different baseline models were designed by hand and evaluated to find the optimum. This 
will be discussed further in section X.X.

### Train-Test Resolution Discrepancy

For our training examples, we use the common approach of taking random crops on different
scales. At test time, we want to utilise as much of the image as possible rather than taking
a small crop that only encapsulates a small part of the image. We instead take a large centre crop
of the image. Both the small crops and the large centre are scaled to be the same resolution.

Touvron et al. argued that this different approach of preprocessing at train time 
compared to test time leads to poor performance [2]. The scale of components in the image 
appear different at test time and CNNs are not scale invariant. Touvron et al. propose 
increasing the resolution at test time, such that the scale of image components appear
as they did during training.
 
Having the images at a different resolution for testing skews the activation 
statistics of the model so fine tuning is needed. The classifier is trained normally
using the random crop method on the training set. The model is then fine tuned by 
performing the test time preprocessing on the training set and using this to train the 
model. The increased resolution of the images does increase the latency of the model
but the fine tuning phase is relatively short so is still possible when computational 
resources are limited.

This approach allows models to be trained on smaller images to significantly reduce 
training time. They achieved the highest ImageNet single-crop accuracy 
at the time of publication in 2019.

## Network Architecture

The baseline model was designed from MBConv blocks with the resolution decreasing through
the network and the width increasing. With the scaling strategy, choosing an optimum
baseline model is important but a complex search strategy is not feasible given the 
computing resources. Instead we devised three different architectures all with roughly 50,000
parameters and evaluated their performance. 

- __Model A__: Designed to be deeper with fewer channels
- __Model B__: Designed to have medium depth with larger kernel sizes and a medium amount of channels
- __Model C__: Designed to be shallower with more channels and smaller kernel sizes

### Results

| Model Name  | Test Loss  | Test Accuracy | 
|---|---|---|
| A  | 0.5246 |  68.47% |
| B | 0.4098 | 81.75%  |
| C  | __0.3561__ | __84.48%__  |

### Selected Architecture

| Stage  | Layer  | Resolution | Kernel Size | Channels | Layers
|---|---|---|---|---|---|
| 1  | Conv2D | 64 x 64 | 7 x 7 | 12 | 1
| 2  | MBConv t=1 | 32 x 32 | 3 x 3 | 16 | 1
| 3  | MBConv t=6 | 16 x 16 | 5 x 5 | 22 | 1
| 4  | MBConv t=6 | 16 x 16 | 3 x 3 | 26 | 2
| 5  | MBConv t=6 | 16 x 16 | 3 x 3 | 32 | 1
| 6  | Conv2D | 8 x 8 | 1 x 1 | 40 | 1

Stage 6 is followed with an additional batch normalisation, global average pooling 
and a dense layer with softmax for the two labels.

## Scaling Factor Selection

Given our baseline selected architecture, we perform a grid search on optimum values
for <img src="https://render.githubusercontent.com/render/math?math=\alpha, \beta,\gamma">. 
We scale up the network in the following way:

- Stage 1 and 6 remain the same
- Stage 2 is always 1 layer but the new of cahnnels can increase
- The resolution is scaled during preprocessing
- The scaled number of channels and layer is rounded to the nearest integer

|Experiment | <img src="https://render.githubusercontent.com/render/math?math=\alpha">  | <img src="https://render.githubusercontent.com/render/math?math=\beta">  | <img src="https://render.githubusercontent.com/render/math?math=\gamma"> | Test Loss  | Test Accuracy | 
|---|---|---|---|---|---|
| 1  | 1.2 | 1.05 | 1.23 | __0.2581__ | __89.09%__
| 2  | 1.2 | 1.23 | 1.05 | 0.2846 | 87.88%
| 3  | 1.4 | 1.02 | 1.17 | 0.2623 | __89.09%__
| 4  | 1.4 | 1.05 | 1.14 | 0.2659 | 87.95%
| 5  | 1.4 | 1.14 | 1.05 | 0.3356 | 85.64%
| 6  | 1.4 | 1.17 | 1.02 | 0.3230 | 85.58%

The scaling parameters in experiment 1 were selected as the optimum. 

### Train-Test Resolution

We need to calculate the scale to increase our test time resolution by such that images
appear at the same scale. Touvron et al. outlined an approach to do this. 

First we calculate the expected proportion, <img src="https://render.githubusercontent.com/render/math?math=\sigma">, 
of the training image used in the crop. We uniformly sample this during
training between 0.2 and 0.7 but occasionally the aspect ratio of an image does not allow
a square crop of our selected size. We calculate the estimated proportion by running the
preprocessing over our entire training set and taking an average of the proportion of the image used.

We found: 
<img src="https://render.githubusercontent.com/render/math?math=\sigma \approx 0.444">

At test time we have been resizing the image (preserving the aspect ratio) and then croping it to a square of size (`k_image_test`, `k_image_test`).
We then take a centre crop of size (`k_test`, `k_test`). To calculate the estimated change of scale between train and test time
we calculate: 

<img src="https://render.githubusercontent.com/render/math?math=r = \sigma \cdot \ k^{image}_{test}/k_{train}">
where <img src="https://render.githubusercontent.com/render/math?math=(k_{train}, k_{train})"> is the resolution of our 
train crops.

We scale up the size of our test time images by <img src="https://render.githubusercontent.com/render/math?math=1/r">.

In our case we estimate <img src="https://render.githubusercontent.com/render/math?math=r = 0.444 \cdot \ 72/ 64 \approx 0.5">. 
This means that the resolution of our images at test time should be twice that of our training images.


## Final Results

Up to this point, we have been using the same test set to select our optimum baseline model
and optimum scaling parameters. To gain a more realistic performance estimate we take a 
different test set of the same size at random.

The models were scaled up using <img src="https://render.githubusercontent.com/render/math?math=\phi=4">

| Model | Test Loss | Test Accuracy
|---|---|---|
| Trained without fine tuning | 0.2520 | 89.77% 
| Trained with fine tuning | 0.1277 | 94.66%  

## Install

## Run

## Library Structure

## References