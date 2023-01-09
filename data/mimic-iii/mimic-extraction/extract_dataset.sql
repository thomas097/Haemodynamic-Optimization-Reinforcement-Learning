SET SEARCH_PATH TO public,mimiciii;

-- Extract materizalized views for all features of interest
BEGIN;
\i create_cohort_table.sql
\i extract_urine_output_all.sql
\i extract_gcs_all.sql
\i extract_lab_results_all.sql
\i extract_vitals_all.sql
\i extract_cohort.sql
\i extract_vasopressors_metavision.sql
\i extract_inputevents_metavision.sql
\i extract_ventilator.sql
\i extract_lab_results.sql
\i extract_vitals.sql
\i extract_pao2.sql
\i extract_demographics.sql
\i extract_urine_output.sql
\i extract_fio2.sql
\i extract_gcs.sql

--
-- Change paths from '.../Master Thesis Project VU' to location of repository
--

\copy rl_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/cohort.csv' CSV HEADER;
\copy vasopressors_mv2 TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/vasopressors_mv_cohort.csv' CSV HEADER;
\copy inputevents_mv2 TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/inputevents_mv_cohort.csv' CSV HEADER;
\copy vent_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/vent_cohort.csv' CSV HEADER;
\copy labs_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/labs_cohort.csv' CSV HEADER;
\copy vitals_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/vitals_cohort.csv' CSV HEADER;
\copy demographics_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/demographics_cohort.csv' CSV HEADER;
\copy urineoutput_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/urineoutput_cohort.csv' CSV HEADER;
\copy fio2_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/fio2_cohort.csv' CSV HEADER;
\copy gcs_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/gcs_cohort.csv' CSV HEADER; -- NEW: used for SOFA