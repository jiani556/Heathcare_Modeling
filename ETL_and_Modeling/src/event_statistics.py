import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    Read the events.csv and mortality_events.csv files.
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    Event count is defined as the number of events recorded for a given patient.
    '''
    avg_dead_event_count = events[events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id').count().mean()[0]
    max_dead_event_count = events[events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id').count().max()[0]
    min_dead_event_count = events[events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id').count().min()[0]
    avg_alive_event_count = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id').count().mean()[0]
    max_alive_event_count = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id').count().max()[0]
    min_alive_event_count = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id').count().min()[0]

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count
def encounter_count_metrics(events, mortality):
    '''
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU.
    '''

    avg_dead_encounter_count = events[events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].nunique().mean()
    max_dead_encounter_count = events[events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].nunique().max()
    min_dead_encounter_count = events[events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].nunique().min()
    avg_alive_encounter_count = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].nunique().mean()
    max_alive_encounter_count = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].nunique().max()
    min_alive_encounter_count = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].nunique().min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Record length is the duration between the first event and the last event for a given patient.
    '''
    uniq_ts = events.groupby('patient_id')['timestamp'].unique().reset_index()
    uniq_ts['length'] = uniq_ts.apply(lambda row: (pd.to_datetime(row['timestamp']).max() - pd.to_datetime(row['timestamp']).min()).days, axis=1)
    avg_dead_rec_len = uniq_ts[uniq_ts['patient_id'].isin(mortality['patient_id'])]['length'].mean()
    max_dead_rec_len = uniq_ts[uniq_ts['patient_id'].isin(mortality['patient_id'])]['length'].max()
    min_dead_rec_len = uniq_ts[uniq_ts['patient_id'].isin(mortality['patient_id'])]['length'].min()
    avg_alive_rec_len = uniq_ts[~uniq_ts['patient_id'].isin(mortality['patient_id'])]['length'].mean()
    max_alive_rec_len = uniq_ts[~uniq_ts['patient_id'].isin(mortality['patient_id'])]['length'].max()
    min_alive_rec_len = uniq_ts[~uniq_ts['patient_id'].isin(mortality['patient_id'])]['length'].min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():

    train_path = '../data/train/'

    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)

if __name__ == "__main__":
    main()
