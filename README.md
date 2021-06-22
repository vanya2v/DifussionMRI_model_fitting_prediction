# Difussion MRI Model Fitting and Prediction with Deep Networks

Deep-learning based model fitting and Gleason score lesion prediction.


### Referenceing
If you use this repository in your diffusion MRI work please refer to this citation:

Vanya Valindria, Saurabh Singh,  Eleni Chiou,  Eleftheria Panagiotaki, et al. "Non-invasive Gleason Score Classification with VERDICT-MRI," 29th Annual Meeting of ISMRM, 2021.

Full text is available in the document.

### How to use:

1. For deep learning based model fitting, most of the codes are in MATLAB. You can use the example scan (INN-104-RWB) to run the scripts. Run the following codes in order:

```
make_training_dataset_DL.m 
```
to generate synthetic data from diffusion models under their own biophysical ranges. Output is the 'database...mat'

```
train_MLP_fitting.py
```
to train using the generated synthetic data (.mat) using a simple 3-layer Multi Layer Perceptron (MLP). Once you have the trained model (.sav), you can apply it on patient data (raw DW-MRI data, dependable on protocol).

```
preprocessing.m
```
to preprocess the (registration and denoising/unring etc) raw patient DW-MRI scans

```post_process_DL.m
```
to obtain the ROI data for input to MLP training

```
apply_MLP_fitting.py
```
applying trained MLP to the input data (after being pre and post-processed)

```save_maps.m
```
to convert from regression prediction from MLP to parametric maps of difussion MRI model 



2. For Gleason score (GS) prediction

Install MONAI first -> https://monai.io/

Run:
```
GS_classification.py
```

We need all/some of the parametric maps obtained from Step 1 and extract the pre-defined lesion ROI, and the ground truth (Gleason score for each lesion). We classify the lesion to 5-point Gleason score using SE-ResNet50 in MONAI, as shown as in the paper above, it gives better accuracy than DenseNet and resNet.

