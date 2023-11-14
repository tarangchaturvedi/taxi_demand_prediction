from utility import split_data
import numpy as np
import pandas as pd


def data_featurization(kmeans, regions_cum):
    # we take number of pickups that are happened in last 5 10min intravels
    number_of_time_stamps = 5

    # it is list of lists
    # it will contain number of pickups 13099 for each cluster
    output = []

    # tsne_lat will contain 13104-5=13099 times lattitude of cluster center for every cluster
    tsne_lat = []

    # tsne_lon will contain 13104-5=13099 times logitude of cluster center for every cluster
    tsne_lon = []
    # for every cluster we will be adding 13099 values, each value represent to which day of the week that pickup bin belongs to
    # it is list of lists
    tsne_weekday = []

    tsne_feature = []

    tsne_feature = [0]*number_of_time_stamps
    for i in range(0,40):
        tsne_lat.append([kmeans.cluster_centers_[i][0]]*13099)
        tsne_lon.append([kmeans.cluster_centers_[i][1]]*13099)
        # our prediction start from 5th 10min intravel since we need to have number of pickups that are happened in last 5 pickup bins
        tsne_weekday.append([int(((int(k/144))%7+4)%7) for k in range(5,4464+4176+4464)])
        
        tsne_feature = np.vstack((tsne_feature, [regions_cum[i][r:r+number_of_time_stamps] for r in range(0,len(regions_cum[i])-number_of_time_stamps)]))
        output.append(regions_cum[i][5:])
    tsne_feature = tsne_feature[1:]

    #################################################################################
    # Getting the predictions of exponential moving averages to be used as a feature in cumulative form
    
    alpha=0.3

    # it is a temporary array that store exponential weighted moving avarage for each 10min intravel, 
    predicted_values=[]

    # it is similar like tsne_lat
    predict_list = []
    tsne_flat_exp_avg = []
    for r in range(0,40):
        for i in range(0,13104):
            if i==0:
                predicted_value= regions_cum[r][0]
                predicted_values.append(0)
                continue
            predicted_values.append(predicted_value)
            predicted_value =int((alpha*predicted_value) + (1-alpha)*(regions_cum[r][i]))
        predict_list.append(predicted_values[5:])
        predicted_values=[]

    #################################################################################

    train_features, test_features = split_data(tsne_feature)
    #################################################################################

    # the above contains values in the form of list of lists (i.e. list of values of each region), here we make all of them in one list
    train_new_features = []
    for i in range(0,40):
        train_new_features.extend(train_features[i])
    test_new_features = []
    for i in range(0,40):
        test_new_features.extend(test_features[i])
    ################################################################################

    # Preparing the data frame for our train data
    columns = ['ft_5','ft_4','ft_3','ft_2','ft_1']
    df_train = pd.DataFrame(data=train_new_features, columns=columns) 
    df_train['lat'] = sum([i[:9169] for i in tsne_lat], [])
    df_train['lon'] = sum([i[:9169] for i in tsne_lon], [])
    df_train['weekday'] = sum([i[:9169] for i in tsne_weekday], [])
    df_train['exp_avg'] = sum([i[:9169] for i in predict_list], [])

    #print(df_train.shape)

    # Preparing the data frame for our train data
    df_test = pd.DataFrame(data=test_new_features, columns=columns) 
    df_test['lat'] = sum([i[9169:] for i in tsne_lat], [])
    df_test['lon'] = sum([i[9169:] for i in tsne_lon], [])
    df_test['weekday'] = sum([i[9169:] for i in tsne_weekday], [])
    df_test['exp_avg'] = sum([i[9169:] for i in predict_list], [])
    #print(df_test.shape)

    tsne_train_output = sum([i[:9169] for i in output], [])
    tsne_test_output = sum([i[9169:] for i in output], [])

    return df_train, df_test, tsne_train_output, tsne_test_output
    ##################################################################################