import pandas as pd
from pandas.core.nanops import nankurt
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

data = pd.read_csv('./data/qsar_fish_toxicity.csv', delimiter=';')
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

X = data.iloc[:, :6]
Y = data[data.columns[-1]]

x_train, x_temp, y_train, y_temp = train_test_split(
      X, Y, test_size=0.5, random_state=42, shuffle=True)

x_validation, x_test, y_validation, y_test = train_test_split(
      x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
  )

# CIC0 = data.iloc[:, 0].values
# SM1_Dz = data.iloc[:, 1].values
# GATS1i = data.iloc[:, 2].values
# NdsCH = data.iloc[:, 3].values
# NdssC = data.iloc[:, 4].values
# MLOGP = data.iloc[:, 5].values
# LC50 = data.iloc[:, 6].values


# analisando o relacionamento entre as variáveis
plt.figure(figsize=(8, 8))
sn.set(font_scale=1)
sn.heatmap(data.corr(),annot=True,vmin=-1, vmax=1,linewidth=.5,fmt=".2f");
plt.show()



# To each one, run 20 times and get:
    # - Mean Average Error
    # - Mean Squared Error
    # - Root Mean Squared Error

k_values = np.random.randint(1, 21, 20)  # Randomly choose values of K between 1 and 20
distance_metrics = np.random.choice(['minkowski', 'euclidean', 'manhattan'], 20)  # Randomly pick distance metrics

# Store the results


knr_r = []
knr_
knr_mae_scores = []
knr_mse_scores = []
knr_rmse_scores = []

for i in range(20):
    # Get the current K and distance metric
    k = k_values[i]
    metric = distance_metrics[i]

    # Create and train the KNeighborsRegressor model
    knr = KNeighborsRegressor(n_neighbors=k, metric=metric)
    knr.fit(x_train, y_train)

    # Make predictions on the validation set
    Y_pred = knr.predict(x_validation)

    # Calculate errors
    mae = mean_absolute_error(y_validation, Y_pred)
    mse = mean_squared_error(y_validation, Y_pred)
    rmse = np.sqrt(mse)

    # Store the results
    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)

    # Print the current run's K, metric, and errors
    print(f"Run {i+1}: K = {k}, Distance = {metric}")
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}\n")

# Summarize results
print("Average MAE over 20 runs:", np.mean(mae_scores))
print("Average MSE over 20 runs:", np.mean(mse_scores))
print("Average RMSE over 20 runs:", np.mean(rmse_scores))


## SVR
    # - Kernel
    # - C
print ("SVR")
for i in range(1):
    for c in range(1, 200):
        # for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for kernel in ['linear']:
            svr = SVR(kernel=kernel, C=100)
            svr.fit(x_train, y_train)

            svr_score = svr.score(x_train, y_train)
            svr_test_score = svr.score(x_test, y_test)

            predict = svr.predict(x_test)
            mqe = mean_squared_error(y_test, predict)
            rmqe = root_mean_squared_error(y_test, predict)
            mae = mean_absolute_error(y_test, predict)

            plt.scatter(y_test,predict)
            plt.title("SVR")
            plt.xlabel("Valor Real") #variável explicativa
            plt.ylabel("Valor Estimado") #variável dependente
            plt.show()


## MLP
    # - Hidden Layer
    # - Activation Function
    # - Max Iterations
    # - Learning Ratenan
    #

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
