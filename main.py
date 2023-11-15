from data_featurization import data_featurization
import dask.dataframe as dd
from utility import datapreparation, unique_pickup_bins, initialize_kmeans, frame_smoothing, data_loading, frame_preparation
from basline_models import baseline_models
import pandas as pd
from regression_models import regression_models

def main():
    months = ['jan_2015','jan_2016','feb_2016','mar_2016']
    #COLLECTING DATA
    path = "D:\\AppliedAI_Course\\[AppliedAI]\\Module 6_ Machine Learning Real-World Case Studies\\Case study 4_Taxi demand prediction in New York City\\Notes and Resources"
    initial_frames = {}
    data_loading(months, path, initial_frames)

    kmeans = initialize_kmeans(initial_frames['jan_2015'])
    #data processing
    prepared_frames = {}
    groupby_frames = {}
    frame_preparation(months,initial_frames, kmeans, prepared_frames, groupby_frames)

    # for every month we get all indices of 10min intravels in which atleast one pickup got happened
    unique_pickup_bins(months)

    # Jan-2015 data is smoothed, Jan,Feb & March 2016 data missing values are filled with zero
    smoothed_frames= {}
    frame_smoothing(months, smoothed_frames)

    baseline_models(smoothed_frames)

    #featurizing data for regression models
    df_train, df_test, tsne_train_output, tsne_test_output = data_featurization(kmeans, smoothed_frames)

    regression_models(df_train, df_test, tsne_train_output, tsne_test_output)

if __name__ == "__main__":
    main()
