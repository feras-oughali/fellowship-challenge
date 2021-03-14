# Fellowship.ai Challenge

**The work can be summarized by:**
- exploring the dataset
- creating a baseline model
- using semi-supervised learning by implementing FixMatch to utilize the unlabeled images
- interpretation using GRADCAM

**My main contribution from this work was creating a FixMatch fastai callback that can be used effortlessly within the framework.**  

In this implementation, all data for training was used (5000 labels). This achieved **94.31%** accuracy on the test set. Using one of the predefined folds of 1000 labels drops the error rate by about 10%. This relatively poor performance was partly due to the use of resnet34. Note that no hyperparameters tuning was performed to report these results. The recommended hyperparameters from the [paper](https://arxiv.org/abs/2001.07685) was used. 

This implementation was based on FixMatch pytorch implementation from this [repo](https://github.com/kekmodel/FixMatch-pytorch), and utilizing RandAugment from `timm` library. 

Visualization of the final layer activations using GRADCAM for a mislabeled image of the weakest class was done using fastai hook callback. 

**Final remarks:**

- Some of the unlabeled images from this dataset come from a similar but different distribution from the labeled data which makes it a more realistic challenge.
- resnet34 model is not a good choice for interpretability purposes and performance wise. As images size is relatively small (96x96), the size of feature map at the last conv layer of the model is very small (only 3x3). This is not good enough for a detailed interpretability. Additionally, deeper networks are not likely to improve performance.
- A shallower and wider network would do better in this scenario. This is justified by the observations made below and by the model of choice when reporting SOTA resutls in published papers which is a wide resnet network(WRN-37-2).
- Great amount of debugging went into this work to make it happen. Tests for GPU memory leak was also performed to insure smooth training on long runs. 
- Additional work can be done to explore the utilization of the unlabeled images like, monitoring percentage of used images for pseudo labeling, visually analyzing a sample of utilized images, and the possibility of removing out of domain images to enhance the quality and speed of training. 
