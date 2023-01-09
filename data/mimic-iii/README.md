
## Instructions
To extract the MIMIC-III dataset as used in our paper, please follow the following steps:

0. First, request access to the MIMIC-III v1.4 database via PhysioNet: 
    - Follow the instructions at: https://physionet.org/content/mimiciii/1.4/

1. Install PostgreSQL from http://www.postgresql.org/download/windows/
    -  Set (and remember) password, e.g. `postgrespass0123!`

2. Launch the *psql* client
    - Leave everything blank except when asked to provide the password

3. Execute the following commands to build a MIMIC-III database:
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

4. Run MIMIC-Code commands to derive concepts (additional tables and views) from MIMIC-III:
   - `$ \cd '<PATH_TO_REPO>/data/mimic-iii/mimic-code/query'`
   - `$ \i make-tables.sql`
   - `$ \dt`

Check whether a table `patient_cohort` is included in the list. If so, everything was succesful!

5. Extract materialized views of tables to obtain dataset in CSV format
   - `$ \cd '<PATH_TO_REPO>/data/mimic-iii/mimic-concepts'`
   - `$ \i extract_dataset.sql`
   
6. Go to `<PATH_TO_REPO>/preprocessing` and follow the steps in `DataPreprocessing_Aggregated.ipynb`
    
    
