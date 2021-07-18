# Rotten-Fruit-Classification-using-Neural-Network-Transfer-Learning-
Images are an important source of data and information in the agricultural sciences. Fruit classification is an important task in markets because of the similarity of different fruits. They could be used in industries to detect a specific fruit and segregate them separately from a conveyor with various fruits. Therefore, automatic fruit classification is necessary and urgent. <br />
 <br />
The main objective of this project is to classify whether the given image  of fruits (apples/orange/banana) are fresh or rotten. To detect the fruits are fresh or rotten by using transfer learning techniques. The main objective of this project is classify whether the given image is rotten apple/banana/orange or fresh apple/banana/orange.VGG16 convolution neural network is used. Transfer learning is done with a fully connected layer on top of vgg16
 <br />
 <br />
  <br />
 <b>Transfer Learning: <b />
  <br />
   <br />
   Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task. Transfer learning is an optimization that allows rapid progress or improved performance when modelling the second task. Transfer learning is the process that is widely used today in training deep neural networks so as to save the training time and resources and get good results. Different factors on which different transfer learning strategies are used based on the size of the data and similarity of the data by which one can quickly decide about the strategy that is to be used for transfer learning.
<br />
<br />
  It is common to perform transfer learning with predictive modeling problems that use image data as input. This may be a prediction task that takes photographs or video data as input. For these types of problems, it is common to use a deep learning model pre-trained for a large and challenging image classification task such as the ImageNet 1000-class photograph classification competition. The research organizations that develop models for this competition and do well often release their final model under a permissive license for reuse. These models can take days or weeks to train on modern hardware. These models can be downloaded and incorporated directly into new models that expect image data as input.
  
<br />
<br />
VGG16:

<br />
<br />
  VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous models submitted toILSRVC-2014. It makes the improvement over Alex Net by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.
  
<br />
<br />
  
<br />
<br />
  I have attached the source code for the project in .ipynb format and also in pdf format with detailed explanation below each line for better understanding. The results of the prediction and also loss function plot is also uploaded here for reference. Details like number of layer, number of neurons per layer are clearly understandable through the image of the model summary that is also uploaded here. We were able to derive accuracy of 92.14 %

  
