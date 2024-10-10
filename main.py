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

knr_dict = {"": [], "K": [], "WEIGHTS": [], "MAE": [], "MSE": [], "RMSE": []}
rf_dict = {"": [], "N_ESTIMATORS": [], "MAX_DEPTH": [], "MAE": [], "MSE": [], "RMSE": []}
gb_dict = {"": [], "N_ESTIMATORS": [], "LEARNING_RATE": [], "MAX_DEPTH": [], "MAE": [], "MSE": [], "RMSE": []}
svr_dict = {"": [], "C": [], "KERNEL": [], "MAE": [], "MSE": [], "RMSE": []}
mlp_dict = {"": [], "HIDDEN_LAYER": [], "ACTIVATION": [], "MAE": [], "MSE": [], "RMSE": []}

for i in range(20): #

    x_train, x_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.5, random_state=i, shuffle=True)

    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )


    current_rmse = float('inf')
    current_mse = float('inf')
    current_mae = float('inf')
    current_k = 0
    current_weights = '0'
    knr_best_model = KNeighborsRegressor()
    mae = None
    rmse = None
    mse = None
    for k in range(1, 20):
        for weights in ['uniform', 'distance']:
            knr = KNeighborsRegressor(n_neighbors=k, weights = "uniform" )
            knr.fit(x_train, y_train)

            Y_pred = knr.predict(x_validation)

            mae = mean_absolute_error(y_validation, Y_pred)
            mse = mean_squared_error(y_validation, Y_pred)
            rmse = np.sqrt(mse)

            if current_rmse > rmse:
                current_rmse = rmse
                knr_best_model = knr
                current_mse = mse
                current_mae = mae
                current_weights = weights
                current_k = k
            print(f"Run {i+1}: K = {k}, Weights = {weights}")
            print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}\n")

    Y_pred = knr_best_model.predict(x_test)

    mae = mean_absolute_error(y_validation, Y_pred)
    mse = mean_squared_error(y_validation, Y_pred)
    rmse = np.sqrt(mse)

    knr_dict[''].append(i)
    knr_dict['K'].append(current_k)
    knr_dict['WEIGHTS'].append(current_weights)
    knr_dict['MAE'].append(current_mae)
    knr_dict['MSE'].append(current_mse)
    knr_dict['RMSE'].append(current_rmse)



    ## SVR
    print ("SVR")
    svr_best_model: SVR = SVR()
    svr_best_rmse = float('inf')
    svr_best_c = 0
    svr_best_kernel = 0

    for c in range(1, 20):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        # for kernel in ['linear']:
            svr = SVR(kernel=kernel, C=c)
            svr.fit(x_train, y_train)

            svr_score = svr.score(x_train, y_train)
            svr_test_score = svr.score(x_test, y_test)

            Y_pred = svr.predict(x_test)

            mae = mean_absolute_error(y_validation, Y_pred)
            mse = mean_squared_error(y_validation, Y_pred)
            rmse = np.sqrt(mse)

            if rmse < svr_best_rmse:
                svr_best_rmse = rmse
                svr_best_model = svr
                svr_best_c = c
                svr_best_kernel = kernel


    svr_dict["C"].append(c)
    svr_dict["KERNEL"].append(svr_best_kernel)
    svr_dict["MAE"].append(mae)
    svr_dict["MSE"].append(mse)
    svr_dict["RMSE"].append(rmse)
    svr_dict[""].append(i+1)
    predict_validation = svr_best_model.predict(x_validation)


    ## MLP
        # - Hidden Layer
        # - Activation Function
        # - Max Iterations
        # - Learning Ratenan
        #
    mlp_best_model = MLPRegressor()
    mlp_best_rmse = float('inf')
    mlp_rmqe_sum = 0

    mlp_best_hidden_layer_sizes = 0
    mlp_best_activation = 0


    print("MLP")
    for hidden_layer_sizes in [(50,), (100,), (50, 50), (100, 100)]:
        for activation in ['relu', 'tanh', 'logistic']:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=500)
            mlp.fit(x_train, y_train)

            Y_pred = mlp.predict(x_test)

            mae = mean_absolute_error(y_validation, Y_pred)
            mse = mean_squared_error(y_validation, Y_pred)
            rmse = np.sqrt(mse)


            mlp_rmqe_sum += rmse

            if rmse < mlp_best_rmse:
                mlp_best_rmse = rmse
                mlp_best_model = mlp
                mlp_best_activation = activation
                mlp_best_hidden_layer_sizes = hidden_layer_sizes

    mlp_dict["HIDDEN_LAYER"].append(mlp_best_hidden_layer_sizes)
    mlp_dict["ACTIVATION"].append(mlp_best_activation)
    mlp_dict["MAE"].append(mae)
    mlp_dict["MSE"].append(mse)
    mlp_dict["RMSE"].append(rmse)
    mlp_dict[""].append("MLP")

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


    current_max_depth = 0
    current_n_estimators = 0
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
                current_rmse = rmse
                current_mse = mse
                current_mae = mae
                current_max_depth = max_depth
                current_n_estimators = n_estimators

    # Calculate means
    rf_dict[''].append(i)
    rf_dict['N_ESTIMATORS'].append(current_n_estimators)
    rf_dict['MAX_DEPTH'].append(current_max_depth)
    rf_dict['MAE'].append(current_mae)
    rf_dict['MSE'].append(current_mse)
    rf_dict['RMSE'].append(current_rmse)


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


    print("GB")
    current_learning_rate = 0
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
                    current_learning_rate = learning_rate
                    current_n_estimators = n_estimators
                    current_max_depth = max_depth

    gb_dict[''].append(i)
    gb_dict['N_ESTIMATORS'].append(current_n_estimators)
    gb_dict['MAX_DEPTH'].append(current_max_depth)
    gb_dict['LEARNING_RATE'].append(current_learning_rate)
    gb_dict['MAE'].append(current_mae)
    gb_dict['MSE'].append(current_mse)
    gb_dict['RMSE'].append(current_rmse)



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



## All CSVs
### KN Regressor
knr_dict["K"].append(np.mean(knr_dict["K"]))
knr_dict["WEIGHTS"].append("")
knr_dict["K"].append(np.std(knr_dict["K"]))
knr_dict["WEIGHTS"].append("")
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
svr_dict[""].append("Average")
svr_dict["MAE"].append(np.mean(svr_dict["MAE"]))
svr_dict["MSE"].append(np.mean(svr_dict["MSE"]))
svr_dict["RMSE"].append(np.mean(svr_dict["RMSE"]))
svr_dict[""].append("Standard Deviation")
svr_dict["MAE"].append(np.std(svr_dict["MAE"]))
svr_dict["MSE"].append(np.std(svr_dict["MSE"]))
svr_dict["RMSE"].append(np.std(svr_dict["RMSE"]))

svr_dict["C"].append(np.mean(svr_dict["C"]))
svr_dict["KERNEL"].append("")
svr_dict["C"].append(np.std(svr_dict["C"]))
svr_dict["KERNEL"].append("")

svr_best_rmse = np.min(svr_dict["RMSE"])
svr_rmse_sum = np.sum(svr_dict["RMSE"])


svr_df = pd.DataFrame(svr_dict)
svr_df.to_csv("svr-results.csv")

## MLP
mlp_dict["HIDDEN_LAYER"].append(np.nan)
mlp_dict["ACTIVATION"].append("")
mlp_dict["HIDDEN_LAYER"].append(np.nan)
mlp_dict["ACTIVATION"].append("")
mlp_dict[""].append("Average")
mlp_dict["MAE"].append(np.mean(mlp_dict["MAE"]))
mlp_dict["MSE"].append(np.mean(mlp_dict["MSE"]))
mlp_dict["RMSE"].append(np.mean(mlp_dict["RMSE"]))
mlp_dict[""].append("Standard Deviation")
mlp_dict["MAE"].append(np.std(mlp_dict["MAE"]))
mlp_dict["MSE"].append(np.std(mlp_dict["MSE"]))
mlp_dict["RMSE"].append(np.std(mlp_dict["RMSE"]))



mlp_df = pd.DataFrame(mlp_dict)
mlp_df.to_csv("mlp-results.csv")

## RF
rf_dict["N_ESTIMATORS"].append(np.nan)
rf_dict["MAX_DEPTH"].append("")
rf_dict["N_ESTIMATORS"].append(np.nan)
rf_dict["MAX_DEPTH"].append("")
rf_dict[""].append("Average")
rf_dict["MAE"].append(np.mean(rf_dict["MAE"]))
rf_dict["MSE"].append(np.mean(rf_dict["MSE"]))
rf_dict["RMSE"].append(np.mean(rf_dict["RMSE"]))
rf_dict[""].append("Standard Deviation")
rf_dict["MAE"].append(np.std(rf_dict["MAE"]))
rf_dict["MSE"].append(np.std(rf_dict["MSE"]))
rf_dict["RMSE"].append(np.std(rf_dict["RMSE"]))

rf_df = pd.DataFrame(rf_dict)
rf_df.to_csv("rf-results.csv")

## GB
gb_dict["N_ESTIMATORS"].append(np.nan)
gb_dict["LEARNING_RATE"].append("")
gb_dict["MAX_DEPTH"].append("")
gb_dict["N_ESTIMATORS"].append(np.nan)
gb_dict["LEARNING_RATE"].append("")
gb_dict["MAX_DEPTH"].append("")
gb_dict[""].append("Average")
gb_dict["MAE"].append(np.mean(gb_dict["MAE"]))
gb_dict["MSE"].append(np.mean(gb_dict["MSE"]))
gb_dict["RMSE"].append(np.mean(gb_dict["RMSE"]))
gb_dict[""].append("Standard Deviation")
gb_dict["MAE"].append(np.std(gb_dict["MAE"]))
gb_dict["MSE"].append(np.std(gb_dict["MSE"]))
gb_dict["RMSE"].append(np.std(gb_dict["RMSE"]))

gb_df = pd.DataFrame(gb_dict)
gb_df.to_csv("gb-results.csv")


## Análise estatística
from scipy.stats import kruskal

knr_rmse = knr_dict['RMSE'][:-2]
svr_rmse = svr_dict['RMSE'][:-2]
mlp_rmse = mlp_dict['RMSE'][:-2]
rf_rmse = rf_dict['RMSE'][:-2]
gb_rmse = gb_dict['RMSE'][:-2]

stat, p_value = kruskal(knr_rmse, svr_rmse, mlp_rmse, rf_rmse, gb_rmse)

print(f"Kruskal-Wallis H-statistic: {stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Diferença significante")
