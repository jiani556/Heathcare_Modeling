-- ***************************************************************************
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
events = LOAD '../../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
mortality = LOAD '../../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
eventswithmort = JOIN events BY patientid LEFT OUTER, mortality BY patientid;
eventswithmort = FOREACH eventswithmort GENERATE events::patientid AS patientid, events::eventid AS eventid, events::value AS value, events::etimestamp as etimestamp, mortality::mtimestamp as mtimestamp, (mortality::label IS NULL ? 0:1) AS label;
-- perform join of events and mortality by patientid;

deadevents = FILTER eventswithmort BY (label == 1);
deadevents = FOREACH deadevents GENERATE patientid AS patientid, eventid AS eventid, value AS value, label AS label, DaysBetween(SubtractDuration(mtimestamp,'P30D'), etimestamp) AS time_difference;
-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

aliveevents = FILTER eventswithmort BY (label != 1);
aliveindextime = GROUP aliveevents BY patientid;
aliveindextime = FOREACH aliveindextime GENERATE group AS patientid, MAX(aliveevents.etimestamp) AS indexdate;
aliveevents = JOIN aliveevents BY patientid, aliveindextime BY patientid;
aliveevents = FOREACH aliveevents GENERATE aliveevents::patientid AS patientid, aliveevents::eventid AS eventid, aliveevents::value AS value, 0 AS label, DaysBetween(aliveindextime::indexdate, aliveevents::etimestamp) AS time_difference;
-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
STORE deadevents INTO 'deadevents' USING PigStorage(',');

-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ensure the number of rows of your output feature is 3618, otherwise the following sections will be highly impacted and make you lose all the relevant credits.
-- ***************************************************************************
allevents = UNION aliveevents, deadevents;
filtered = FILTER allevents BY (value IS NOT NULL) AND (time_difference >= 0L) AND (time_difference <= 2000L);
-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)

--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
STORE filtered INTO 'filtered' USING PigStorage(',');

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
featureswithid = GROUP filtered BY (patientid,eventid);
featureswithid = FOREACH featureswithid GENERATE group.patientid AS patientid, group.eventid AS eventid, COUNT(filtered.eventid) AS featurevalue;
-- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)

--TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
all_features = FOREACH featureswithid GENERATE eventid;
all_features = DISTINCT all_features;
all_features = ORDER all_features BY eventid ASC;
all_features = RANK all_features;
all_features = FOREACH all_features GENERATE ($0-1) AS idx, eventid;
-- compute the set of distinct eventids obtained from previous step, sort them by eventid and then rank these features by eventid to create (idx, eventid). Rank should start from 0.

-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');

features = JOIN featureswithid BY eventid, all_features BY eventid;
features = FOREACH features GENERATE featureswithid::patientid AS patientid, all_features::idx AS idx, featureswithid::featurevalue AS featurevalue;
-- perform join of featureswithid and all_features by eventid and replace eventid with idx. It is of the form (patientid, idx, featurevalue)

--TEST-4
features = ORDER features BY patientid, idx;
STORE features INTO 'features_map' USING PigStorage(',');

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- Use DOUBLE precision
-- ***************************************************************************
maxvalues = GROUP features BY idx;
maxvalues = FOREACH maxvalues GENERATE group AS idx, MAX(features.featurevalue) AS maxvalue;
-- group events by idx and compute the maximum feature value in each group. I t is of the form (idx, maxvalue)

normalized = JOIN features BY idx, maxvalues BY idx;
-- join features and maxvalues by idx

features = FOREACH normalized GENERATE features::patientid AS patientid, features::idx AS idx, ((double)features::featurevalue/(double)maxvalues::maxvalue) AS normalizedfeaturevalue;
-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)

--TEST-5
features = ORDER features BY patientid, idx;
STORE features INTO 'features_normalized' USING PigStorage(',');

-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	2,0
--      3,1
-- ***************************************************************************

labels = FOREACH filtered GENERATE patientid, label;
labels = DISTINCT labels;
-- create it of the form (patientid, label) for dead and alive patients

--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
STORE samples INTO 'samples' USING PigStorage(' ');

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('6505');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');
