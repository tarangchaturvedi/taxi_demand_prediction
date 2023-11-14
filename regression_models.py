import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def regression_models(df_train, df_test, tsne_train_output, tsne_test_output):
    lr_reg=LinearRegression().fit(df_train, tsne_train_output)
    y_pred = lr_reg.predict(df_test)
    lr_test_predictions = [round(value) for value in y_pred]
    y_pred = lr_reg.predict(df_train)
    lr_train_predictions = [round(value) for value in y_pred]

    # Training a hyper-parameter tuned random forest regressor on our train data
    regr1 = RandomForestRegressor(max_features='sqrt',min_samples_leaf=4,min_samples_split=3,n_estimators=40, n_jobs=-1)
    regr1.fit(df_train, tsne_train_output)

    # Predicting on test data using our trained random forest model
    y_pred = regr1.predict(df_test)
    rndf_test_predictions = [round(value) for value in y_pred]
    y_pred = regr1.predict(df_train)
    rndf_train_predictions = [round(value) for value in y_pred]

    # Training a hyper-parameter tuned Xg-Boost regressor on our train data
    x_model = xgb.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=3, min_child_weight=3, gamma=0, subsample=0.8, reg_alpha=200, reg_lambda=200, colsample_bytree=0.8,nthread=4)
    x_model.fit(df_train, tsne_train_output)

    #predicting with our trained Xg-Boost regressor
    y_pred = x_model.predict(df_test)
    xgb_test_predictions = [round(value) for value in y_pred]
    y_pred = x_model.predict(df_train)
    xgb_train_predictions = [round(value) for value in y_pred]

    train_mape=[]
    test_mape=[]

    train_mape.append((mean_absolute_error(tsne_train_output,df_train['ft_1'].values))/(sum(tsne_train_output)/len(tsne_train_output)))
    train_mape.append((mean_absolute_error(tsne_train_output,df_train['exp_avg'].values))/(sum(tsne_train_output)/len(tsne_train_output)))
    train_mape.append((mean_absolute_error(tsne_train_output,rndf_train_predictions))/(sum(tsne_train_output)/len(tsne_train_output)))
    train_mape.append((mean_absolute_error(tsne_train_output, xgb_train_predictions))/(sum(tsne_train_output)/len(tsne_train_output)))
    train_mape.append((mean_absolute_error(tsne_train_output, lr_train_predictions))/(sum(tsne_train_output)/len(tsne_train_output)))

    test_mape.append((mean_absolute_error(tsne_test_output, df_test['ft_1'].values))/(sum(tsne_test_output)/len(tsne_test_output)))
    test_mape.append((mean_absolute_error(tsne_test_output, df_test['exp_avg'].values))/(sum(tsne_test_output)/len(tsne_test_output)))
    test_mape.append((mean_absolute_error(tsne_test_output, rndf_test_predictions))/(sum(tsne_test_output)/len(tsne_test_output)))
    test_mape.append((mean_absolute_error(tsne_test_output, xgb_test_predictions))/(sum(tsne_test_output)/len(tsne_test_output)))
    test_mape.append((mean_absolute_error(tsne_test_output, lr_test_predictions))/(sum(tsne_test_output)/len(tsne_test_output)))

    print ("Error Metric Matrix (Tree Based Regression Methods) -  MAPE")
    print ("--------------------------------------------------------------------------------------------------------")
    print ("Baseline Model -                             Train: ",train_mape[0],"      Test: ",test_mape[0])
    print ("Exponential Averages Forecasting -           Train: ",train_mape[1],"      Test: ",test_mape[1])
    print ("Linear Regression -                         Train: ",train_mape[4],"      Test: ",test_mape[4])
    print ("Random Forest Regression -                   Train: ",train_mape[2],"     Test: ",test_mape[2])
    print ("XgBoost Regression -                         Train: ",train_mape[3],"      Test: ",test_mape[3])
    print ("--------------------------------------------------------------------------------------------------------")