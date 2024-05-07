# iALP
iALP:Identification of allergenic proteins based on large language model and gate linear unit


## Introduction
In this paper, we develop a deep neural network model based on linear units, named iALP, for predicting allergenic proteins. Compared with existing methods, this work has the following advantages:  
(1) In terms of feature extraction, iALP uses a large language model for allergen protein information extraction, which can capture more and deeper protein sequence information.
(2) In terms of modeling, iALP uses gated linear units connected to convolutional neural networks, which not only extracts linearly correlated features but also captures the complex nonlinear features hidden in these proteins.
(3) By adding manual features, this model can improve the accuracy of recognizing allergenic proteins under 100 amino acids in length.
The framework of the iALP method for allergen protein prediction is described below:
<img src="Figure 1.png">






## Related Files

#### iALP

| FILE NAME                  | DESCRIPTION                                                                                        |
|:---------------------------|:---------------------------------------------------------------------------------------------------|
| main.py                    | the main file of iALP predictor                                                                    |
| new dataset.py             | the file of the iALP predictor on the independent test set                                         |
| 10-100 predictor.py        | the file of improved performance of the iALP predictor on ALP from 10 to 100 amino acids in length | 
| data                       | three datasets                                                                                     |
                                                                                    



## Installation
- Requirement
  
  OS：
  
  - `Windows` ：Windows10 or later
  
  - `Linux`：Ubuntu 16.04 LTS or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `iALP`to your computer

  ```bash
  git clone https://github.com/xialab-ahu/iALP.git
  ```

- open the dir and install `requirement.txt` with `pip`

  ```
  cd iALP
  pip install -r requirement.txt
  ```
  
## Training and test iALP model
```shell
cd "./iALP/iALP"
python main.py
```


## Contact
Please feel free to contact us if you need any help.

