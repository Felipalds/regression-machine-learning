import pandas as pd

data = pd.read_csv('./data/qsar_fish_toxicity.csv')

print(data.head)

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
