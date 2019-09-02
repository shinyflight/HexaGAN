## HexaGAN: Generative Adversarial Nets for Real World Classification

### A Tensorflow implementation of HexaGAN (Pytorch version will be uploaded)

![HexaGAN_model](https://user-images.githubusercontent.com/25117385/64109029-66de4f80-cdb9-11e9-9e57-93797996de33.png)

When dealing with the real world data, we encounter three problems such as 1) missing data, 2) class imbalance, and 3) missing label problems. In this paper, we propose HexaGAN, a generative adversarial network framework that shows promising classification performance for all three problems. <br>


* Authors: Uiwon Hwang, Dahuin Jung, Sungroh Yoon

* Paper URL: http://proceedings.mlr.press/v97/hwang19a/hwang19a.pdf

* Appendix URL: http://proceedings.mlr.press/v97/hwang19a/hwang19a-supp.pdf

* Presentation PPT: https://icml.cc/media/Slides/icml/2019/halla(11-14-00)-11-15-15-4629-hexagan_genera.pdf


#### Files

* ops.py: various operations for building neural networks and data loading

* ops_cnn.py: various operations for convolutional neural networks (for the MNIST dataset)

* model.py: HexaGAN model (for the breast dataset)

* train_breast.py: classification on the breast dataset with 20% missingness

* train_mnist.py: missing data imputation on the MNIST dataset with 50% missingness (including HexaGAN model)
