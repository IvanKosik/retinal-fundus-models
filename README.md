# retinal-fundus-models
Code to train neural network models for retinal fundus vessel segmentation.

## Environment Configuration
### To use updated last versions of libs
```
conda create -n retinal-fundus-models python=3.7
conda activate retinal-fundus-models
pip install tensorflow-gpu==1.14 "keras<2.4" "h5py<3" albumentations numpy opencv-python pandas scikit-image segmentation-models
```

### Or to use the same versions of libs
```
conda env create -f environment.yaml
```
