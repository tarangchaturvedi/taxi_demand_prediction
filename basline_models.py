import math
######
WINDOW_SIZE = 3
TOTAL_ITERATIONS = 4464 * 40

def moving_average_predictions(ratios, prediction_type):
    if prediction_type == 'R':
        predicted_value = ratios['Ratios'].values[0]
    else:
        predicted_value = ratios['Prediction'].values[0]

    error = []
    predicted_values = []

    for i in range(TOTAL_ITERATIONS):
        if i % 4464 == 0:
            predicted_values.append(0)
            error.append(0)
            continue

        predicted_values.append(predicted_value)

        if prediction_type == 'R':
            error.append(abs(math.pow(int(ratios['Given'].values[i] * predicted_value) - ratios['Prediction'].values[i], 1)))
        else:
            error.append(abs(math.pow(predicted_value - ratios['Prediction'].values[i], 1)))

        if i + 1 >= WINDOW_SIZE:
            sum_values = sum([(j + 1) * ratios[prediction_type].values[i - WINDOW_SIZE + j] for j in range(WINDOW_SIZE)])
            sum_of_coeff = sum(range(WINDOW_SIZE, 0, -1))
            predicted_value = int(sum_values / sum_of_coeff)
        else:
            sum_values = sum([(j + 1) * ratios[prediction_type].values[j] for j in range(i + 1)])
            sum_of_coeff = sum(range(i + 1, 0, -1))
            predicted_value = int(sum_values / sum_of_coeff)

    ratios[f'MA_{prediction_type}_Predicted'] = predicted_values
    ratios[f'MA_{prediction_type}_Error'] = error
    mape_err = (sum(error) / len(error)) / (sum(ratios['Prediction'].values) / len(ratios['Prediction'].values))
    mse_err = sum([e ** 2 for e in error]) / len(error)
    
    return ratios, mape_err, mse_err

###############################################################################################

def exponential_average_predictions(ratios, prediction_type, alpha):
    predicted_value = ratios['Ratios'].values[0] if prediction_type == 'R' else ratios['Prediction'].values[0]

    error = []
    predicted_values = []

    for i in range(TOTAL_ITERATIONS):
        if i % 4464 == 0:
            predicted_values.append(0)
            error.append(0)
            continue

        predicted_values.append(predicted_value)

        if prediction_type == 'R':
            error.append(abs(math.pow(int(ratios['Given'].values[i] * predicted_value) - ratios['Prediction'].values[i], 1)))
        else:
            error.append(abs(math.pow(predicted_value - ratios['Prediction'].values[i], 1)))

        predicted_value = alpha * predicted_value + (1 - alpha) * ratios[prediction_type].values[i]

    ratios[f'EA_{prediction_type}_Predicted'] = predicted_values
    ratios[f'EA_{prediction_type}_Error'] = error
    mape_err = (sum(error) / len(error)) / (sum(ratios['Prediction'].values) / len(ratios['Prediction'].values))
    mse_err = sum([e ** 2 for e in error]) / len(error)
    
    return ratios, mape_err, mse_err
############################################################################################

def baseline_models():
    mean_err=[0]*5
    median_err=[0]*5
    ratios,mean_err[0],median_err[0]=moving_average_predictions(ratios,'R')
    ratios,mean_err[1],median_err[1]=moving_average_predictions(ratios,'P')
    ratios,mean_err[2],median_err[2]=exponential_average_predictions(ratios,'R',0.5)
    ratios,mean_err[3],median_err[3]=exponential_average_predictions(ratios,'P',0.5)

    ##########
    print ("Error Metric Matrix (Forecasting Methods) - MAPE & MSE")
    print ("--------------------------------------------------------------------------------------------------------")
    print ("Moving Averages (Ratios) -                             MAPE: ",mean_err[0],"      MSE: ",median_err[0])
    print ("Moving Averages (2016 Values) -                        MAPE: ",mean_err[1],"       MSE: ",median_err[1])
    print ("--------------------------------------------------------------------------------------------------------")
    print ("Exponential Moving Averages (Ratios) -              MAPE: ",mean_err[2],"      MSE: ",median_err[2])
    print ("Exponential Moving Averages (2016 Values) -         MAPE: ",mean_err[3],"      MSE: ",median_err[3])