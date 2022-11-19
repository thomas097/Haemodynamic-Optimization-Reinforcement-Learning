-- As the script is generating many tables, it may take some time.
-- We assume the database and the search path are set correctly.
-- You can set the search path as follows:
SET SEARCH_PATH TO public,mimiciii;
-- This will create tables on public and read tables from mimiciii

BEGIN;
-- Generate views for RL dataset
\i create_cohort_table.sql
\i create_UrineOutput_view.sql
\i gcs_all.sql
\i labs_all_rl.sql
\i vitals_all_rl.sql
\i get_cohort.sql
\i get_vasopressor_cv.sql
\i get_vasopressor_mv.sql
\i get_inputevents_cv.sql
\i get_inputevents_mv.sql
\i get_labs_cohort.sql
\i get_vitals_cohort.sql
\i get_demographics_cohort.sql
\i get_urineoutput_cohort.sql
\i get_FiO2_cohort.sql
\i get_gcs_cohort.sql

-- save to file
\copy rl_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/cohort.csv' CSV HEADER;
\copy vasopressors_cv TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/vasopressors_cv_cohort.csv' CSV HEADER;
\copy vasopressors_mv TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/vasopressors_mv_cohort.csv' CSV HEADER;
\copy inputevents_cv2 TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/inputevents_cv_cohort.csv' CSV HEADER;
\copy inputevents_mv2 TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/inputevents_mv_cohort.csv' CSV HEADER;
\copy labs_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/labs_cohort.csv' CSV HEADER;
\copy vitals_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/vitals_cohort.csv' CSV HEADER;
\copy demographics_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/demographics_cohort.csv' CSV HEADER;
\copy urineoutput_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/urineoutput_cohort.csv' CSV HEADER;
\copy fio2_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/fio2_cohort.csv' CSV HEADER;
\copy gcs_cohort TO 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/final/gcs_cohort.csv' CSV HEADER; -- NEW: used for SOFA