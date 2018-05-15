# Eyes in the Skies: A Data-driven Fusion Approach to Identifying Drug Crops from Remote Sensing Images

## Authors: Anselmo Ferreira, Siovani C. Felipussi, Ramon Pires, Sandra Avila, Geise Santos, Jorge Lambert, Jiwu Huang and Anderson Rocha

> This is the source code from our submitted paper to IEEE Transactions in Image Processing. In this paper, we built, 
in collaboration with the Brazilian Federal Police, the first dataset and the first data-driven approach for identifying drug crops (i.e., Marijuana crops)
from remote sensing images. The paper is currently under review so, to protect the intellectual property of our research, we did not publish (for now)
the models of our data-driven models nor the dataset, which shall be published under the paper acceptance.

## Environment and Dependencies

We used Python 2.7 and latest Keras at the time of the paper submisson.

### Dataset and CNN models

The Dataset will be published under paper acceptance and can be found at http://dx.doi.org/10.21227/H2WD42. 

Models contained in the "models" folder are empty models. We are intending to replace them with the real ones once our paper is accepted for publication

#### Changes in the code

 You just need to change the filepaths contained at fold-data files, pointing them to where the dataset is in your computer. We are not intending 
 to let the code running in this platform, so, you need to download the dataset and run the code in your computer. 

## How to run the code

Just run demo.py. The code will use individual proposed models and the ensembled model to generate feature vectors for training and testing an SVM classifier.
