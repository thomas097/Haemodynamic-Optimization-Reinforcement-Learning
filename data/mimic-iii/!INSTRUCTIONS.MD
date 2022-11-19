0. Install PostgreSQL from http://www.postgresql.org/download/windows/
    -  Set password, e.g. postgrespass0123!

1. Launch psql client
    - Leave everything blank except for the password

2. Execute buold commands to construct MIMIC-III database:
   - $ DROP DATABASE IF EXISTS mimic;
   - $ CREATE DATABASE mimic OWNER postgres;
   - $ \c mimic;
   - $ CREATE SCHEMA mimiciii;
   - $ set search_path to mimiciii;
   - $ \i '<PATH_TO_REPO>/data/mimic-iii/buildmimic/postgres_create_tables.sql';
   - $ \set ON_ERROR_STOP 1
   - $ \set mimic_data_dir '<PATH_TO_DATASET>/mimic-iii-clinical-database-1.4'
   - $ \i '<PATH_TO_REPO>/data/mimic-iii/buildmimic/postgres_load_data.sql';
   - $ \i '<PATH_TO_REPO>/data/mimic-iii/buildmimic/postgres_add_indexes.sql';

3. Run MIMIC-Code commands to infer concepts (extra tables) from MIMIC-III:
   - $ \cd '<PATH_TO_REPO>/data/mimic-iii/mimic-code/query'
   - $ \i make-tables.sql
   - $ \dt  

If there is a `patient_cohort` table in the list everything was succesful :)

4. Extract materialized RL-views to obtain CSV dataset
   - $ \cd '<PATH_TO_REPO>/data/mimic-iii/mimic-thomas'
   - $ \i extract_dataset.sql
    
    
