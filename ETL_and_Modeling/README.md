# ETL and modeling

## Data
- we will be working with the MIMIC database. MIMIC, although de-identified, still contains detailed information regarding the clinical care of patients, and must be treated with appropriate care and respect. In order to obtain access, it is necessary to finish the MIMIC CITI program training provided by MIT and get the certificate.
- https://www.citiprogram.org/index.cfm?pageID=154&icat=0&ac=0.

## Descriptive Statistics
- src/event statistics.py will create Descriptive Statistics for Event Count, Encounter Count, Record Length

## Feature construction
- convert raw data into a standard data format before running real machine learning models
- src/etl.py file will implement the necessary python functions:1)Compute the index date 2)Filter events 3)Aggregate events 4)Save in SVMLight format

## Predictive Modeling
-Logistic Regression, SVM and Decision Tree to perform Mortality Prediction
