from data_featurization import data_featurization
import dask.dataframe as dd
from utility import datapreparation, return_unq_pickup_bins, initialize_kmeans, fill_missing, smoothing
from basline_models import baseline_models
import pandas as pd
from regression_models import regression_models

def main():
    #COLLECTING DATA
    path = "D:\\AppliedAI_Course\\[AppliedAI]\\Module 6_ Machine Learning Real-World Case Studies\\Case study 4_Taxi demand prediction in New York City\\Notes and Resources"    
    month_jan_2015 = dd.read_csv(path + "\\" + 'yellow_tripdata_2015-01.csv')
    print("jan 2015 data loaded")
    month_jan_2016 = dd.read_csv(path + "\\" + 'yellow_tripdata_2016-01.csv')
    month_feb_2016 = dd.read_csv(path + "\\" + 'yellow_tripdata_2016-02.csv')
    month_mar_2016 = dd.read_csv(path + "\\" + 'yellow_tripdata_2016-03.csv')

    #print(month_jan_2015.columns)

    kmeans = initialize_kmeans(month_jan_2015)
    #data preparation
    jan_2015_frame,jan_2015_groupby = datapreparation(month_jan_2015,kmeans,1,2015)    
    jan_2016_frame,jan_2016_groupby = datapreparation(month_jan_2016,kmeans,1,2016)
    feb_2016_frame,feb_2016_groupby = datapreparation(month_feb_2016,kmeans,2,2016)
    mar_2016_frame,mar_2016_groupby = datapreparation(month_mar_2016,kmeans,3,2016)

    # for every month we get all indices of 10min intravels in which atleast one pickup got happened

    #jan
    jan_2015_unique = return_unq_pickup_bins(jan_2015_frame)
    jan_2016_unique = return_unq_pickup_bins(jan_2016_frame)
    #feb
    feb_2016_unique = return_unq_pickup_bins(feb_2016_frame)
    #march
    mar_2016_unique = return_unq_pickup_bins(mar_2016_frame)

    # Jan-2015 data is smoothed, Jan,Feb & March 2016 data missing values are filled with zero
    jan_2015_fill = fill_missing(jan_2015_groupby['trip_distance'].values,jan_2015_unique)
    jan_2015_smooth = smoothing(jan_2015_groupby['trip_distance'].values,jan_2015_unique)
    jan_2016_smooth = fill_missing(jan_2016_groupby['trip_distance'].values,jan_2016_unique)
    feb_2016_smooth = fill_missing(feb_2016_groupby['trip_distance'].values,feb_2016_unique)
    mar_2016_smooth = fill_missing(mar_2016_groupby['trip_distance'].values,mar_2016_unique)

    # Making list of all the values of pickup data in every bin for a period of 3 months and storing them region-wise 
    regions_cum = []

    for i in range(0,40):
        regions_cum.append(jan_2016_smooth[4464*i:4464*(i+1)]+feb_2016_smooth[4176*i:4176*(i+1)]+mar_2016_smooth[4464*i:4464*(i+1)])

    #Preparing the Dataframe only with x(i) values as jan-2015 data and y(i) values as jan-2016
    ratios = pd.DataFrame()
    ratios['Given']=jan_2015_smooth
    ratios['Prediction']=jan_2016_smooth
    ratios['Ratios']=ratios['Prediction']*1.0/ratios['Given']*1.0

    ###########
    baseline_models(ratios)
    ###########

    #featurizing data for regression models
    df_train, df_test, tsne_train_output, tsne_test_output = data_featurization()

    regression_models(df_train, df_test, tsne_train_output, tsne_test_output)

if __name__ == "__main__":
    main()