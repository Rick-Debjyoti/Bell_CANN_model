# Bell_CANN_model
Combined actuarial neural network(CANN)  enhancement of the Bell regression model for count frequency data.



## Overview

In this project, we explored the application of neural networks to enhance the Bell regression model for analyzing insurance claim frequency data. Generalized linear models (GLMs) are commonly used for such analysis, but they have limitations in capturing non-linearity and interaction terms among features. To address this, we proposed the Combined Actuarial Neural Network (CANN) model , first introduced by Professor Mario V. Wüthrich, which integrates neural networks with GLMs to introduce non-linearity and interaction terms. The French Motor Third party Liability(freMTPL2freq) dataset was used for modelling where risk features and claim numbers were collected for 677,991 motor third-part liability policies (observed on a year). The dataset is available in the CASdatasets R library. 

## Key Features

* Utilization of feed-forward neural networks combined with GLMs for enhanced modeling.
* Incorporation of embedding layers to encode categorical variables.
* Custom loss function implementation using Keras backend for deviance loss, typically used in GLMs.
* Integration of Bell regression model to address over-dispersion in insurance claim count data.

## Getting Started

* Clone this repository:

```
git clone https://github.com/Rick-Debjyoti/Bell_CANN_model.git
```

* Install dependencies:
```
pip install -r requirements.txt
```

* Refer to the `src/` directory for code implementation of the Jupyter notebook or R script.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

I am deeply thankful to my supervisor, `Dr. Deepesh Bhati`, for his
invaluable guidance, unwavering support, and insightful feedback. 
I am also grateful to Ph.D. Scholar, `Dr. Girish Aradhye` , who provided huge support and
encouragement for learning and research.