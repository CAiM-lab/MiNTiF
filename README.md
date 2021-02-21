![Logo](MiNTiF_Utils/images_docs/logo_small.png)
## CNN Framework


MiNTiF ("MiNTiF Is Networks Tool In Fiji") will allow users to define, train and deploy tensorflow models within Fiji.

[Documentation](MiNTiF_Utils/images_docs//MINTIF_Documentation.pdf)
## Installation
* Download and install [FIJI](https://imagej.net/Fiji/Downloads).
* Download the [MiNTiF git repository](https://github.com/CAiM-lab/MiNTiF) and place the folder "MiNTiF” in the Fiji sub-folder:
  
      {Userspecific}\Fiji.app\plugins\Scripts\Plugins
* Place the folder "MiNTiF_Utils” in the Fiji sub-folder:
  
      {Userspecific}\Fiji.app
* Before you use MiNTiF please install \& update [Anaconda](https://www.anaconda.com/products/individual).
* Restart Fiji. Find the MiNTiF Plugin in Fiji under Plugins >   MiNTiF. \
  Install the necessary packages by running the command MiNTiF >  Utilities >  Install Environment



## Summary
* **Define Model**: Define the parameters to construct a MiNTiF Convolutional Neural Network (CNN).
* **Crate Dataset**: Create a new MiNTiF Dataset from an Image currently loaded in Fiji, or append the Image to an already existing MiNTiF file. These files will be used to train models and predict the labels of images.
* **Deploy Model**: Train a model on training datasets  or predict the labels of test datasets.
    * **Training**: Teach a model how to accomplish a task (segmentation/detection) by providing annotated data and letting the Network determine the relevant features
    * **Prediction**: Predict labels on new datasets. The network will use the trained parameters from the previous step to predict labels on previously unseen data.
* **Reconstruct**: Reconstruct the image form MiNTiF files after prediction. The reconstructed images will contain the predicted labels as separate image channels.

Use this example dataset to test MiNTiF:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4552197.svg)](https://doi.org/10.5281/zenodo.4552197)




## Changelog
    
* **Version 1.0 (Release)**: 
  * Detection Pipeline.
  * Support for larger than RAM images.
  * Prediction and transfer learning from pretrained models.
  * Various smaller improvements to usability and function.

* **Version 0.1 (Pre-Release)**: 
    * GUI to use cnnframework workflow for 2D Segmentation. 
    * Create Dataset for training and prediction.
    * Define parameters for preset models. 
    * Train and predict on datasets.
    * Reconstruct with labels.

    
[License](MiNTiF/MiNTiF/LICENSE)
