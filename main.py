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
from sklearn.neural_network import MLPRegressor


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
# plt.figure(figsize=(8, 8))
# sn.set(font_scale=1)
# sn.heatmap(data.corr(),annot=True,vmin=-1, vmax=1,linewidth=.5,fmt=".2f");
# plt.show()




# To each one, run 20 times and get:
    # - Mean Average Error
    # - Mean Squared Error
    # - Root Mean Squared Error

k_values = np.random.randint(1, 21, 20)
distance_metrics = np.random.choice(['minkowski', 'euclidean', 'manhattan'], 20)

knr_dict = {"": [], "K": [], "METRIC": [], "MAE": [], "MSE": [], "RMSE": []}

current_rmse = float('inf')
knr_best_model = None

for i in range(20):
    k = k_values[i]
    metric = distance_metrics[i]

    knr = KNeighborsRegressor(n_neighbors=k, metric=metric)
    knr.fit(x_train, y_train)

    Y_pred = knr.predict(x_validation)

    mae = mean_absolute_error(y_validation, Y_pred)
    mse = mean_squared_error(y_validation, Y_pred)
    rmse = np.sqrt(mse)

    if current_rmse > rmse:
        current_rmse = rmse
        knr_best_model = knr

    knr_dict["K"].append(k)
    knr_dict["METRIC"].append(metric)
    knr_dict["MAE"].append(mae)
    knr_dict["MSE"].append(mse)
    knr_dict["RMSE"].append(rmse)
    knr_dict[""].append(i+1)

    print(f"Run {i+1}: K = {k}, Distance = {metric}")
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}\n")

# Calculate means and stdevs
knr_dict["K"].append(np.mean(knr_dict["K"]))
knr_dict["METRIC"].append("")
knr_dict["K"].append(np.std(knr_dict["K"]))
knr_dict["METRIC"].append("")
knr_dict[""].append("Average")
knr_dict["MAE"].append(np.mean(knr_dict["MAE"]))
knr_dict["MSE"].append(np.mean(knr_dict["MSE"]))
knr_dict["RMSE"].append(np.mean(knr_dict["RMSE"]))
knr_dict[""].append("Standard Deviation")
knr_dict["MAE"].append(np.std(knr_dict["MAE"]))
knr_dict["MSE"].append(np.std(knr_dict["MSE"]))
knr_dict["RMSE"].append(np.std(knr_dict["RMSE"]))

knr_best_rmse = np.min(knr_dict["RMSE"])
knr_rmse_sum = np.sum(knr_dict["RMSE"])


knr_df = pd.DataFrame(knr_dict)
knr_df.to_csv("knr-results.csv")

## SVR
    # - Kernel
    # - C

svr_best_model: SVR = SVR()
svr_best_rmse = float('inf')
svr_rmqe_sum = 0
svr_dict = {"": [], "C": [], "KERNEL": [], "MAE": [], "MSE": [], "RMSE": []}

print ("SVR")
for i in range(1):
    for c in range(1, 10):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        # for kernel in ['linear']:
            svr = SVR(kernel=kernel, C=100)
            svr.fit(x_train, y_train)

            svr_score = svr.score(x_train, y_train)
            svr_test_score = svr.score(x_test, y_test)

            Y_pred = svr.predict(x_test)

            mae = mean_absolute_error(y_validation, Y_pred)
            mse = mean_squared_error(y_validation, Y_pred)
            rmse = np.sqrt(mse)

            svr_dict["C"].append(c)
            svr_dict["KERNEL"].append(kernel)
            svr_dict["MAE"].append(mae)
            svr_dict["MSE"].append(mse)
            svr_dict["RMSE"].append(rmse)
            svr_dict[""].append(i+1)

            svr_rmqe_sum += rmse

            if rmse < svr_best_rmse:
                svr_best_rmse = rmse
                svr_best_model = svr


        predict_validation = svr_best_model.predict(x_validation)
        # plt.scatter(y_test, predict_validation)
        # plt.title("SVR")
        # plt.xlabel("Valor Real") #variável explicativa
        # plt.ylabel("Valor Estimado") #variável dependente
        # plt.show()

# Calculate means
svr_dict["C"].append(np.mean(svr_dict["C"]))
svr_dict["KERNEL"].append("")
svr_dict[""].append("Average")
svr_dict["MAE"].append(np.mean(svr_dict["MAE"]))
svr_dict["MSE"].append(np.mean(svr_dict["MSE"]))
svr_dict["RMSE"].append(np.mean(svr_dict["RMSE"]))

svr_best_rmse = np.min(knr_dict["RMSE"])
svr_rmse_sum = np.sum(knr_dict["RMSE"])


svr_df = pd.DataFrame(svr_dict)
svr_df.to_csv("svr-results.csv")

## MLP
    # - Hidden Layer
    # - Activation Function
    # - Max Iterations
    # - Learning Ratenan
    #


mlp_best_model = MLPRegressor()
mlp_best_rmse = float('inf')
mlp_rmqe_sum = 0
mlp_dict = {"": [], "HIDDEN_LAYER": [], "ACTIVATION": [], "MAE": [], "MSE": [], "RMSE": []}

print("MLP")
for hidden_layer_sizes in [(50,), (100,), (50, 50), (100, 100)]:
    for activation in ['relu', 'tanh', 'logistic']:
        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=500)
        mlp.fit(x_train, y_train)

        Y_pred = mlp.predict(x_test)

        mae = mean_absolute_error(y_validation, Y_pred)
        mse = mean_squared_error(y_validation, Y_pred)
        rmse = np.sqrt(mse)

        mlp_dict["HIDDEN_LAYER"].append(hidden_layer_sizes)
        mlp_dict["ACTIVATION"].append(activation)
        mlp_dict["MAE"].append(mae)
        mlp_dict["MSE"].append(mse)
        mlp_dict["RMSE"].append(rmse)
        mlp_dict[""].append("MLP")

        mlp_rmqe_sum += rmse

        if rmse < mlp_best_rmse:
            mlp_best_rmse = rmse
            mlp_best_model = mlp

# Calculate means
mlp_dict["HIDDEN_LAYER"].append(np.nan)
mlp_dict["ACTIVATION"].append("")
mlp_dict[""].append("Average")
mlp_dict["MAE"].append(np.mean(mlp_dict["MAE"]))
mlp_dict["MSE"].append(np.mean(mlp_dict["MSE"]))
mlp_dict["RMSE"].append(np.mean(mlp_dict["RMSE"]))

mlp_df = pd.DataFrame(mlp_dict)
mlp_df.to_csv("mlp-results.csv")

## RF
    # - N Estimators
    # - Max Depth
    # - Min Samples Split
    # - Min Samples Leaf
    # - Criterion
    #
from sklearn.ensemble import RandomForestRegressor

rf_best_model = RandomForestRegressor()
rf_best_rmse = float('inf')
rf_dict = {"": [], "N_ESTIMATORS": [], "MAX_DEPTH": [], "MAE": [], "MSE": [], "RMSE": []}

print("RF")
for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, None]:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(x_train, y_train)

        Y_pred = rf.predict(x_test)

        mae = mean_absolute_error(y_validation, Y_pred)
        mse = mean_squared_error(y_validation, Y_pred)
        rmse = np.sqrt(mse)

        rf_dict["N_ESTIMATORS"].append(n_estimators)
        rf_dict["MAX_DEPTH"].append(max_depth)
        rf_dict["MAE"].append(mae)
        rf_dict["MSE"].append(mse)
        rf_dict["RMSE"].append(rmse)
        rf_dict[""].append("RF")

        if rmse < rf_best_rmse:
            rf_best_rmse = rmse
            rf_best_model = rf

# Calculate means
rf_dict["N_ESTIMATORS"].append(np.nan)
rf_dict["MAX_DEPTH"].append("")
rf_dict[""].append("Average")
rf_dict["MAE"].append(np.mean(rf_dict["MAE"]))
rf_dict["MSE"].append(np.mean(rf_dict["MSE"]))
rf_dict["RMSE"].append(np.mean(rf_dict["RMSE"]))

rf_df = pd.DataFrame(rf_dict)
rf_df.to_csv("rf-results.csv")


## GB
    # - N Estimators
    # - Learning Rate
    # - Loss
    # - Max Depth
    # - Min Samples Split
    # - Min Samples Leaf

from sklearn.ensemble import GradientBoostingRegressor

gb_best_model = GradientBoostingRegressor()
gb_best_rmse = float('inf')
gb_dict = {"": [], "N_ESTIMATORS": [], "LEARNING_RATE": [], "MAX_DEPTH": [], "MAE": [], "MSE": [], "RMSE": []}

print("GB")
for n_estimators in [50, 100, 200]:
    for learning_rate in [0.01, 0.1, 0.2]:
        for max_depth in [3, 5, 7]:
            gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            gb.fit(x_train, y_train)

            Y_pred = gb.predict(x_test)

            mae = mean_absolute_error(y_validation, Y_pred)
            mse = mean_squared_error(y_validation, Y_pred)
            rmse = np.sqrt(mse)

            gb_dict["N_ESTIMATORS"].append(n_estimators)
            gb_dict["LEARNING_RATE"].append(learning_rate)
            gb_dict["MAX_DEPTH"].append(max_depth)
            gb_dict["MAE"].append(mae)
            gb_dict["MSE"].append(mse)
            gb_dict["RMSE"].append(rmse)
            gb_dict[""].append("GB")

            if rmse < gb_best_rmse:
                gb_best_rmse = rmse
                gb_best_model = gb

# Calculate means
gb_dict["N_ESTIMATORS"].append(np.nan)
gb_dict["LEARNING_RATE"].append("")
gb_dict["MAX_DEPTH"].append("")
gb_dict[""].append("Average")
gb_dict["MAE"].append(np.mean(gb_dict["MAE"]))
gb_dict["MSE"].append(np.mean(gb_dict["MSE"]))
gb_dict["RMSE"].append(np.mean(gb_dict["RMSE"]))

gb_df = pd.DataFrame(gb_dict)
gb_df.to_csv("gb-results.csv")

## RLM
    # - None
from sklearn.linear_model import LinearRegression

rlm = LinearRegression()
rlm.fit(x_train, y_train)

Y_pred = rlm.predict(x_test)

mae = mean_absolute_error(y_validation, Y_pred)
mse = mean_squared_error(y_validation, Y_pred)
rmse = np.sqrt(mse)

print(f"RLM - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
