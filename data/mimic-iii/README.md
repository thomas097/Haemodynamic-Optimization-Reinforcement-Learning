In this folder the source code is provided for the extraction of patient trajectories from MIMIC-III

## Instructions
To extract the MIMIC-III dataset as used in our paper, please follow the following steps:

0. First, request access to the MIMIC-III v1.4 database via PhysioNet: 
    - Follow the instructions at: https://physionet.org/content/mimiciii/1.4/

1. Install PostgreSQL 
    -  Download installer and follow instructions at: http://www.postgresql.org/download
    -  Set (and remember) password, e.g. `postgrespass0123!`

2. Launch the *psql* client
    - Leave everything blank except when asked to provide the password
    
3. Download and place MIMIC-Code concept-extraction files into folder `mimic-code/mimic-code`
    - Files can be found at: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts

4. Execute the following commands to build a MIMIC-III database:
   - `$ DROP DATABASE IF EXISTS mimic;`
   - `$ CREATE DATABASE mimic OWNER postgres;`
   - `$ \c mimic;`
   - `$ CREATE SCHEMA mimiciii;`
   - `$ set search_path to mimiciii;`
   - `$ \i '<PATH_TO_REPO>/data/mimic-iii/buildmimic/postgres_create_tables.sql';`
   - `$ \set ON_ERROR_STOP 1`
   - `$ \set mimic_data_dir '<PATH_TO_DATABASE_SOURCE_FILES>/mimic-iii-clinical-database-1.4'`
   - `$ \i '<PATH_TO_REPO>/data/mimic-iii/buildmimic/postgres_load_data.sql';`
   - `$ \i '<PATH_TO_REPO>/data/mimic-iii/buildmimic/postgres_add_indexes.sql';`

5. Run MIMIC-Code commands to derive concepts (additional tables and views) from MIMIC-III:
   - `$ \cd '<PATH_TO_REPO>/data/mimic-iii/mimic-code/query'`
   - `$ \i make-tables.sql`
   - `$ \dt`

Check whether a table `patient_cohort` is included in the list. If so, everything was succesful!

6. Go to `<PATH_TO_REPO>/data/mimic-iii/mimic-extraction/extract_dataset.sql` and change paths to repo

7. Extract materialized views of tables to obtain dataset in CSV format
   - `$ \cd '<PATH_TO_REPO>/data/mimic-iii/mimic-extraction'`
   - `$ \i extract_dataset.sql`
   
8. Go to `<PATH_TO_REPO>/preprocessing` and follow the steps in `DataPreprocessing_Aggregated.ipynb`

The result should be a triplet of datasets `train.csv`, `test.csv` and `valid.csv` along with some metadata files stored in a folder `<PATH_TO_REPO>/preprocessing/datasets/mimic-iii/aggregated_full_cohort_*h`.
    
    
