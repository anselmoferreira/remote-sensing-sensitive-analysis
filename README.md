# Eyes in the Skies: A Data-driven Fusion Approach to Identifying Drug Crops from Remote Sensing Images

## Authors: Anselmo Ferreira, Siovani C. Felipussi, Ramon Pires, Sandra Avila, Geise Santos, Jorge Lambert, Jiwu Huang and Anderson Rocha

This is the source code related to the solution we proposed in our paper published at the IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. In this paper, we built, in collaboration with the Brazilian Federal Police, the first dataset and the first data-driven approach for identifying drug crops (i.e., Marijuana crops)
from remote sensing images. 

The paper can be found at https://ieeexplore.ieee.org/document/8726132

## Environment and Dependencies

We used Python 2.7 and latest Keras at the time of the paper submisson.

### Dataset

The Dataset can be found at http://dx.doi.org/10.21227/H2WD42. 

#### Changes in the code

You just need to change the filepaths contained at fold-data files, pointing them to where the dataset is in your computer.

## How to run the code

Just run demo.py. The code will use individual proposed models and the ensembled model to classify testing data.

On train_the_models there is the code to train the individual and fused models

## NOTES

I am not using SMOTE as you can see in main.py. This happens because I lost the submodel's 3 weights for both bags (probably I did not save them or overwritted the model files after the two-fold cross validation) :-(

I am using in the code a previous version of the experiment, which applies the normalized fused feature vectors to an SVM classifier.

You can try SMOTE in the code, or you can try training submodel 3 from the folder train_the_models folder and also add SMOTE in the training data. The difference in terms of accuracy will not be too much (about 0.5/1% in accuracy)

Any doubts please contact me:

Anselmo Ferreira (anselmo.ferreira@gmail.com)
