# Big data tool for ETL and modeling

## Data
- we will be working with the MIMIC database. MIMIC, although de-identified, still contains detailed information regarding the clinical care of patients, and must be treated with appropriate care and respect. In order to obtain access, it is necessary to finish the MIMIC CITI program training provided by MIT and get the certificate.
- https://www.citiprogram.org/index.cfm?pageID=154&icat=0&ac=0.

## Descriptive Statistics
- hive/event_statistics.hql will create Descriptive Statistics for Event Count, Encounter Count, Record Length

## Feature construction
- convert the raw data to standardized format using Pig
- pig/etl.pig file will implement the necessary python functions:1)Compute the index date 2)Filter events 3)Aggregate events 4)Save in SVMLight format

## Predictive Modeling
- lr/lrsgd.py  SGD Logistic Regression
- mapper reducer:
- hadoop jar /usr/lib/hadoop -mapreduce/hadoop -streaming.jar -D mapreduce.job.reduces=5 -files lr -mapper "python lr/mapper.py -n 5 -r 0.4" -reducer "python lr/reducer.py -f <number of features>"  -input /training -output /models

- generate the ROC curve:
- cat pig/testing/* | python lr/testensemble.py -m models
