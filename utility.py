import dask.dataframe as dd
import math
import datetime
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

def data_loading(months, path, initial_frames):
    for month in months:
        initial_frames[f'{month}'] = dd.read_csv(path + "\\" + f'yellow_tripdata_{month}.csv')

def convert_to_unix(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())

# Return a DataFrame with additional trip time-related columns
def return_with_trip_times(frame):
    duration = frame[['tpep_pickup_datetime','tpep_dropoff_datetime']].compute()
    #pickups and dropoffs to unix time
    duration_pickup = [convert_to_unix(x) for x in duration['tpep_pickup_datetime'].values]
    duration_drop = [convert_to_unix(x) for x in duration['tpep_dropoff_datetime'].values]
    #calculate duration of trips
    durations = (np.array(duration_drop) - np.array(duration_pickup))/float(60)

    #append durations of trips and speed in miles/hr to a new dataframe
    new_frame = frame[['passenger_count','trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','total_amount']].compute()
    
    new_frame['trip_times'] = durations
    new_frame['pickup_times'] = duration_pickup
    new_frame['Speed'] = 60*(new_frame['trip_distance']/new_frame['trip_times'])
    
    return new_frame


#return a dataframe with outliers removed
def remove_outliers(new_frame):
    speed_limit = 45.31
    # Applying all filters
    new_frame = new_frame[
        ((new_frame.dropoff_longitude.between(-74.15, -73.7004)) &
         (new_frame.dropoff_latitude.between(40.5774, 40.9176))) &
        ((new_frame.pickup_longitude.between(-74.15, -73.7004)) &
         (new_frame.pickup_latitude.between(40.5774, 40.9176)))
    ]
    new_frame = new_frame[(new_frame.trip_times > 0) & (new_frame.trip_times < 720)]
    new_frame = new_frame[(new_frame.trip_distance > 0) & (new_frame.trip_distance < 23)]
    new_frame = new_frame[(new_frame.Speed < speed_limit) & (new_frame.Speed > 0)]
    new_frame = new_frame[(new_frame.total_amount < 1000) & (new_frame.total_amount > 0)]

    return new_frame

# Constants for Unix times
UNIX_TIMES_2015 = [1420070400, 1422748800, 1425168000, 1427846400, 1430438400, 1433116800]
UNIX_TIMES_2016 = [1451606400, 1454284800, 1456790400, 1459468800, 1462060800, 1464739200]

def convert_to_est(unix_time, start_unix_time):
    # Convert Unix time to EST
    return int((unix_time - start_unix_time) / 600) + 33

def add_pickup_bins(frame, month, year):
    unix_pickup_times = frame['pickup_times'].values
    unix_times = [UNIX_TIMES_2015, UNIX_TIMES_2016]
    
    start_pickup_unix = unix_times[year - 2015][month - 1]
    
    # Convert pickup times to 10-minute bins in EST
    ten_minute_bins = [convert_to_est(unix_time, start_pickup_unix) for unix_time in unix_pickup_times]
    
    frame['pickup_bins'] = np.array(ten_minute_bins)
    
    return frame


# Data Preparation for the months of Jan,Feb and March 2016
def frame_preparation(months, initial_frames, kmeans, prepared_frames, groupby_frames):
    for month in months:
        if 'jan' in month:
            mnth = 1
        elif 'feb' in month:
            mnth = 2
        else:
            mnth = 3
        if '2015' in month:
            year = 2015
        else:
            year = 2016
        prepared_frames[f'{month}'], groupby_frames[f'{month}'] = datapreparation(initial_frames[f'{month}'],kmeans,mnth,year)
        
def datapreparation(frame,kmeans,month_no,year_no):
    
    print ("Return with trip times..")

    frame_with_durations = return_with_trip_times(frame)
    
    print ("Remove outliers..")
    frame_with_durations_outliers_removed = remove_outliers(frame_with_durations)

    
    print ("Estimating clusters..")
    frame_with_durations_outliers_removed['pickup_cluster'] = kmeans.predict(frame_with_durations_outliers_removed[['pickup_latitude', 'pickup_longitude']])
    #frame_with_durations_outliers_removed_2016['pickup_cluster'] = kmeans.predict(frame_with_durations_outliers_removed_2016[['pickup_latitude', 'pickup_longitude']])

    print ("Final groupbying..")
    final_updated_frame = add_pickup_bins(frame_with_durations_outliers_removed,month_no,year_no)
    final_groupby_frame = final_updated_frame[['pickup_cluster','pickup_bins','trip_distance']].groupby(['pickup_cluster','pickup_bins']).count()
    
    return final_updated_frame,final_groupby_frame

###########################################
def unique_pickup_bins(months):
    for month in months:
        globals()[f'{month}_unique'] = return_unq_pickup_bins(globals()[f'{month}_frame'])

# Gets the unique bins where pickup values are present for each each reigion
def return_unq_pickup_bins(frame):
    values = []
    for i in range(0,40):
        new = frame[frame['pickup_cluster'] == i]
        list_unq = list(set(new['pickup_bins']))
        list_unq.sort()
        values.append(list_unq)
    return values

#####################################################################################
def fill_missing(count_values,values):
    smoothed_regions=[]
    ind=0
    for r in range(0,40):
        smoothed_bins=[]
        for i in range(4464):
            if i in values[r]:
                smoothed_bins.append(count_values[ind])
                ind+=1
            else:
                smoothed_bins.append(0)
        smoothed_regions.extend(smoothed_bins)
    return smoothed_regions


def smoothing(count_values,values):
    smoothed_regions=[] # stores list of final smoothed values of each reigion
    ind=0
    repeat=0 
    smoothed_value=0
    for r in range(0,40):
        smoothed_bins=[] #stores the final smoothed values
        repeat=0
        for i in range(4464):
            if repeat!=0: # prevents iteration for a value which is already visited/resolved
                repeat-=1
                continue
            if i in values[r]: #checks if the pickup-bin exists 
                smoothed_bins.append(count_values[ind]) # appends the value of the pickup bin if it exists
            else:
                if i!=0:
                    right_hand_limit=0
                    for j in range(i,4464):
                        if  j not in values[r]: #searches for the left-limit or the pickup-bin value which has a pickup value
                            continue
                        else:
                            right_hand_limit=j
                            break
                    if right_hand_limit==0:
                    #Case 1: When we have the last/last few values are found to be missing,hence we have no right-limit here
                        smoothed_value=count_values[ind-1]*1.0/((4463-i)+2)*1.0                               
                        for j in range(i,4464):                              
                            smoothed_bins.append(math.ceil(smoothed_value))
                        smoothed_bins[i-1] = math.ceil(smoothed_value)
                        repeat=(4463-i)
                        ind-=1
                    else:
                    #Case 2: When we have the missing values between two known values
                        smoothed_value=(count_values[ind-1]+count_values[ind])*1.0/((right_hand_limit-i)+2)*1.0             
                        for j in range(i,right_hand_limit+1):
                            smoothed_bins.append(math.ceil(smoothed_value))
                        smoothed_bins[i-1] = math.ceil(smoothed_value)
                        repeat=(right_hand_limit-i)
                else:
                    #Case 3: When we have the first/first few values are found to be missing,hence we have no left-limit here
                    right_hand_limit=0
                    for j in range(i,4464):
                        if  j not in values[r]:
                            continue
                        else:
                            right_hand_limit=j
                            break
                    smoothed_value=count_values[ind]*1.0/((right_hand_limit-i)+1)*1.0
                    for j in range(i,right_hand_limit+1):
                            smoothed_bins.append(math.ceil(smoothed_value))
                    repeat=(right_hand_limit-i)
            ind+=1
        smoothed_regions.extend(smoothed_bins)
    return smoothed_regions

###############################################################################################

def initialize_kmeans(frame):
    print("initializing kmeans")
    frame = return_with_trip_times(frame)
    frame = remove_outliers(frame)
    coords = frame[['pickup_latitude', 'pickup_longitude']].values
    kmeans = MiniBatchKMeans(n_clusters=40, batch_size=10000,random_state=0).fit(coords)
    return kmeans
###################################

def split_data(tsne_feature):
    total_timestamps = 13099
    train_size = int(total_timestamps * 0.7)
    test_size = int(total_timestamps * 0.3)

    print("Size of train data:", train_size)
    print("Size of test data:", test_size)

    # Extracting first 70% of timestamp values for training data
    train_features = [tsne_feature[i * total_timestamps : (total_timestamps*i + train_size)] for i in range(40)]

    # Extracting remaining 30% of timestamp values for testing data
    test_features = [tsne_feature[i * total_timestamps + train_size : (i + 1) * total_timestamps] for i in range(40)]
    return train_features, test_features
#########################################

def frame_smoothing(months, smoothed_frames):
    for month in months:
        if month == 'jan_2015':
            globals()[f'{month}_fill'] = fill_missing(globals()[f'{month}_groupby']['trip_distance'].values.compute(), globals()[f'{month}_unique'])
            smoothed_frames[f'{month}'] = smoothing(globals()[f'{month}_groupby']['trip_distance'].values.compute(), globals()[f'{month}_unique'])
        else:
            smoothed_frames[f'{month}'] = fill_missing(globals()[f'{month}_groupby']['trip_distance'].values.compute(), globals()[f'{month}_unique'])
