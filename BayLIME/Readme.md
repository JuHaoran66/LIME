## Setup
Copy-paste the modified_sklearn_BayesianRidge.py file (in the lime/utils folder on this repo) into your local sklearn.linear_model folder. 

This is a modified and condensed version, cited from the original repository at https://github.com/x-y-zhao/BayLime,
updated to work with the latest sklearn packages. For more details, please refer to the original GitHub repository. 
It is recommended, if possible, to run this with Python 3.7.3, scikit-learn 0.22.1, and TensorFlow 2.0.0. 
And it is neceesary to import scikit-learn under 0.24.2 if you want to run the original repo.

To find out where the folder is, simply run:
```python
from sklearn import linear_model
print(linear_model.__file__)
```