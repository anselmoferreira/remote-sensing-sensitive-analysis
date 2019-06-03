# Eyes in the Skies: A Data-driven Fusion Approach to Identifying Drug Crops from Remote Sensing Images

## Authors: Anselmo Ferreira, Siovani C. Felipussi, Ramon Pires, Sandra Avila, Geise Santos, Jorge Lambert, Jiwu Huang and Anderson Rocha

> This is the source code from our submitted paper to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. In this paper, we built, 
in collaboration with the Brazilian Federal Police, the first dataset and the first data-driven approach for identifying drug crops (i.e., Marijuana crops)
from remote sensing images.

## Environment and Dependencies

We used Python 2.7 and latest Keras at the time of the paper submisson.

### Dataset and CNN models

The Dataset will be published under paper acceptance and can be found at http://dx.doi.org/10.21227/H2WD42. 

Models contained in the "models" folder are empty models. We are intending to replace them with the real ones once our paper is accepted for publication

#### Changes in the code

 You just need to change the filepaths contained at fold-data files, pointing them to where the dataset is in your computer. We are not intending 
 to let the code running in this platform, so, you need to download the dataset and run the code in your computer/server. 

## How to run the code

Just run demo.py. The code will use individual proposed models and the ensembled model to classify testing data.

We will also add the code to train the individual and fused model very soon.

Any doubts please contact me:

Anselmo Ferreira (anselmo.ferreira@gmail.com)
