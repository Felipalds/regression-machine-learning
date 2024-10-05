import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv('./data/qsar_fish_toxicity.csv', delimiter=';')

x_train, x_temp = train_test_split(
      data, test_size=0.5)

x_validation, x_test = train_test_split(
      x_temp, test_size=0.5
  )

CIC0 = data.iloc[:, 0].values
SM1_Dz = data.iloc[:, 1].values
GATS1i = data.iloc[:, 2].values
NdsCH = data.iloc[:, 3].values
NdssC = data.iloc[:, 4].values
MLOGP = data.iloc[:, 5].values
LC50 = data.iloc[:, 6].values


#analisando o relacionamento entre as variáveis
plt.figure(figsize=(8, 8))
sn.set(font_scale=1)
sn.heatmap(data.corr(),annot=True,vmin=-1, vmax=1,linewidth=.5,fmt=".2f");
plt.show()

# To each one, run 20 times and get:
    # - Mean Average Error
    # - Mean Squared Error
    # - Root Mean Squared Error

## KNR
    # - K
    # - Distance


## SVR
    # - Kernel
    # - C


## MLP
    # - Hidden Layer
    # - Activation Function
    # - Max Iterations
    # - Learning Rate


## RF
    # - N Estimators
    # - Max Depth
    # - Min Samples Split
    # - Min Samples Leaf
    # - Criterion

## GB
    # - N Estimators
    # - Learning Rate
    # - Loss
    # - Max Depth
    # - Min Samples Split
    # - Min Samples Leaf

## RLM
    # - None
