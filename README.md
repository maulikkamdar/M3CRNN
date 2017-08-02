M3CRNN - MRI to MGMT Methylation Status Prediction using Convolutional Recurrent Neural Network
=======

M3CRNN (spelled 'Macaroon'), is a Convolutional Recurrent Neural Network pipeline for predicting methylation status of MGMT regulatory regions from Brain MRI scans of Glioblastoma Multiforme patients. The source code provides the preprocessing scripts, the CRNN model, as well as a web-based visualization platform. 

The model and the visualization platform actually uses pre-processed MRI scans. You can actually download the entire TCIA GBM MRI dataset, as well as the methylation dataset for preprocessing using the scripts provided, or contact Lichy Han (lhan2@stanford.edu) or Maulik R. Kamdar (maulikrk@stanford.edu) for the preprocessed data cubes. As the size of the preprocessed data cubes was ~23 GB, we have not uploaded them here. The model also uses as input the Training, Validation and Testing data frames that are uploaded here. The model code also performs the data augmentation part.

The visualization platform can deploy the CRNN pipeline online using Python Flask, and can be used to click and load each MRI scan and visualize filter outputs. This platform is configurable for any volumetric biomedical data set, as well as a different CRNN pipeline. 

**Note**: The visualization platform was actually stripped from the umbrella OntoApps (http://onto-apps.stanford.edu/, so please contact Maulik R. Kamdar (maulikrk@stanford.edu) if there are difficulties deploying the pipeline online

**Live Demo :** http://onto-apps.stanford.edu/m3crnn

**Conference Paper :** MRI to MGMT: Predicting Methylation Status in Glioblastoma Patients using a Convolutional Recurrent Neural Network (under review, Pacific Symposium on Biocomputing, 2018)

**Poster :** http://stanford.edu/~maulikrk/posters/cs231n.pdf

**System Requirements :**
* Python Flask
* Tensorflow 1.1.0
* Pandas
* NumPy

**Dependencies :**
* jQuery 1.7
* D3 JS (http://d3js.org/)
* Twitter Bootstrap

**TODO Technicalities:**
* 



