import utils

def read_csv(filepath):

    '''
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = utils.pd.read_csv(filepath + 'events.csv')

    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = utils.pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = utils.pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):

    '''
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv.
    Use the global variable deliverables_path while specifying the filepath.
    Return indx_date
    '''

    alive_events = events[~events['patient_id'].isin(mortality['patient_id'])].groupby('patient_id')['timestamp'].unique().reset_index()
    alive_events['timestamp'] = alive_events.apply(lambda row: utils.pd.to_datetime(row['timestamp']).max(), axis=1)
    mortality['timestamp'] = mortality['timestamp'].apply(lambda x: utils.date_convert(x))
    mortality['timestamp'] = mortality['timestamp'] - utils.pd.Timedelta(days=30)

    indx_date = utils.pd.concat([mortality, alive_events]).reset_index()
    indx_date = indx_date.sort_values(by=['patient_id']).reset_index()
    indx_date = indx_date[['patient_id', 'timestamp']].rename(columns={'timestamp': 'indx_date'})
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    '''
    Use the global variable deliverables_path while specifying the filepath.
    Each row is of the form patient_id, event_id, value.
    Return filtered_events
    '''

    events['timestamp'] = events.loc[:,'timestamp'].apply(lambda x : utils.date_convert(x))
    events = events.join(indx_date.set_index('patient_id'), on='patient_id')
    events['window'] = events['indx_date'] - events['timestamp']

    filtered_events = events[(events['window'] >= utils.pd.Timedelta(days = 0)) & (events['window'] <= utils.pd.Timedelta(days = 2000))]
    filtered_events = filtered_events.sort_values(by=['patient_id', 'event_id']).reset_index()
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df, feature_map_df, deliverables_path):
    '''
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv.
    Use the global variable deliverables_path while specifying the filepath.
    Return filtered_events
    '''
    filtered_events_df = filtered_events_df[filtered_events_df['value'].notna()]
    # sum values for diagnostics and medication events
    dia_med = filtered_events_df.loc[filtered_events_df['event_id'].str.startswith(('DIAG', 'DRUG'), na=False)].groupby(['patient_id', 'event_id'], as_index=False)['value'].sum()
    #count occurences for lab events
    lab = filtered_events_df.loc[filtered_events_df['event_id'].str.startswith('LAB', na=False)].groupby(['patient_id', 'event_id'], as_index=False)['value'].count()
    agg_events = utils.pd.concat([dia_med, lab])
    agg_events = utils.pd.merge(agg_events, feature_map_df, on='event_id')

    # Normalize
    agg_events['feature_value']  = agg_events.groupby('idx')['value'].transform(lambda x: x / x.max())

    agg_events = agg_events.sort_values(by=['patient_id', 'idx']).reset_index()

    agg_events = agg_events[['patient_id', 'idx', 'feature_value']].rename(columns={'idx': 'feature_id'})
    agg_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', index=False)

    return agg_events


def create_features(events, mortality, feature_map):
    deliverables_path = '../deliverables/'

    # Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    # Filter events in the observation window
    filtered_events = filter_events(events, indx_date, deliverables_path)

    # Aggregate the event values for each patient
    agg_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features = {}
    mortality_dict = {}

    patient_ids = agg_events['patient_id'].unique()
    mortality_ids = mortality['patient_id'].unique()
    for i in patient_ids:
        patient_features[i] = [(agg_events['feature_id'][j], agg_events['feature_value'][j]) for j in agg_events[agg_events['patient_id']==i].index]
        mortality_dict[i] = 1 if i in mortality_ids else 0

    return patient_features, mortality_dict


def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    '''
    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...

    Note: features are ordered in ascending order, and patients are stored in ascending order as well.
    '''
    sorted_patient_features = {}
    for key in sorted(patient_features.keys()):
        sorted_patient_features[key] = sorted(patient_features[key], key=lambda x: x[0])

    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    for key in sorted(sorted_patient_features.keys()):
        d1 = str(int(mortality[key])) + ' ' + str(utils.bag_to_svmlight(sorted_patient_features[key]))
        d2 = str(int(key)) + ' ' + str(mortality[key]) + ' ' + str(utils.bag_to_svmlight(sorted_patient_features[key]))
        deliverable1.write(bytes((f"{d1} \n"), 'UTF-8'));  # Use 'UTF-8'
        deliverable2.write(bytes((f"{d2} \n"), 'UTF-8'));

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train',
                 '../deliverables/features.train')


if __name__ == "__main__":
    main()
