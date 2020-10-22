Code for "Generative causal explanations of black-box classifiers" by Matt O'Shaughnessy, Greg Canal, Marissa Connor, Mark Davenport, and Chris Rozell.

## Requirements
- [numpy](https://numpy.org/)
- [pytorch](https://pytorch.org/)

## Guide to recreate plots
*Figure 3* (global explanations of MNIST digits):
- To train classifier: Run `MNIST_CNN_train.py`, set `model` to `mnist` and `class_use` to `np.array([3,8])`
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.
 
*Figure 4* (comparison with other methods):
- Left side of plot (local explanations from other popular techniques)
   - Run `comparisons_mnist.py`
- Right side of plot (our local explanation)
   - To train classifier: run `MNIST_CNN_train.py`, set `model` to `mnist` and `class_use` to `np.array([3,8])`
   - To train explanatory VAE: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
   - To make plot: run `comparisons_mnist_ours.py`
   
*Figure 5* (quantitative results with fashion MNIST)
 - Subfigures (a) and (b) (information flow and reduction in classifier
   accuracy)
   - To train classifier: Run `MNIST_CNN_train.py`, choose `model` to be `fmnist` and `class_use` to `np.array([0,3,4])`
   - To train explanatory VAE: Run `CVAE.py`, set `data_type` to `fmnist`, `decoder_net` to `VAE_fMNIST`, and `classifier` to `cnn_fmnist`
   - To make plots: run `make_fmnist_quant.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.
 - Subfigure (c) (global explanation of fashion MNIST images)
   - To train classifier: Run `MNIST_CNN_train.py`, choose `model` to be `fmnist` and `class_use` to `np.array([0,3,4])`
   - To train explanatory VAE: Run `CVAE.py`, set `data_type` to `fmnist`, `decoder_net` to `VAE_fMNIST`, and `classifier` to `cnn_fmnist`
   - To make plots: run `make_fmnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.
   
*Figure 8* (empirical results for causal objective with linear/gaussian generative map, linear classifier)
 - Run `visualize_causalobj_linear.py`. This will save results to `results/visualize_causalobj_linear.mat`.
 - To create plot: run first cell of `visualize_causalobj_plot.m`

*Figures 9-10* (empirical results for causal/combined objective with linear/gaussian generative map, AND classifier)
 - To generate data: run `visualize_causalobj_and.py`. This will save results to `results/visualize_causalobj_and.mat`.
 - To create plots: run second and third cells of `visualize_causalobj_plot.m`

*Figure 11* (snapshots of parameter tuning procedure for MNIST digits)
- To train classifier: Run `MNIST_CNN_train.py`, set `model` to `mnist` and `class_use` to `np.array([3,8])`
- To generate data: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plots: run `make_tuning_mnist.m`, changing the file path to the results saved by the previous step

*Figure 12* (additional results for global explanations of MNIST digits):
- To train classifier: Run `MNIST_CNN_train.py`, set `model` to `mnist`
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models. 

*Figure 13* (global explanations of 1/4/9 MNIST digits):
- To train classifier: Run `MNIST_CNN_train.py`, set `model` to `mnist` and `class_use` to `np.array([1,4,9])`
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models. 

*Figure 14* (zoomed global explanations of 1/4/9 MNIST digits):
- To train classifier: Run `MNIST_CNN_train.py`, set `model` to `mnist` and `class_use` to `np.array([1,4,9])`
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `make_mnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models. 

*Figure 15* (detailed plots of our method as used in local comparison)
- To train classifier: run `MNIST_CNN_train.py`, set `model` to `mnist` and `class_use` to `np.array([3,8])`
- To train explanatory VAE: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plot: run `comparisons_mnist_ours.py`

*Figure 16* (snapshots of parameter tuning procedure for fashion MNIST digits)
- To train classifier: Run `MNIST_CNN_train.py`, choose `model` to be `fmnist`
- To generate data: run `CVAE.py`, set `data_type` to `mnist`, `decoder_net` to `VAE_CNN`, and `classifier` to `cnn`
- To make plots: run `make_tuning_mnist.m`, changing the file path to the results saved by the previous step

*Figure 17* (detailed global explanations of fashion MNIST)
- To train classifier: Run `MNIST_CNN_train.py`, choose `model` to be `fmnist` and `class_use` to `np.array([0,3,4])`
- To train explanatory VAE: Run `CVAE.py`, set `data_type` to `fmnist`, `decoder_net` to `VAE_fMNIST`, and `classifier` to `cnn_fmnist`
- To make plots: run `make_fmnist_qual.py`. This script loads pretrained classifier and VAE models; change the paths and parameters at the top of the script to generate plots from different models.

*Figures 18-19* (decomposed objective and global explanations of fashion MNIST with varying VAE capacity)
- To train classifier: Run `MNIST_CNN_train.py`, choose `model` to be `fmnist` and `class_use` to `np.array([0,3,4])`
- To make plots: run `make_vae_capacity.py`. This script loads the pretrained classifier model, and trains explainers with different capacities and values of lambda. The file `make_vae_capacity.m` generates the plots in Figure 18.