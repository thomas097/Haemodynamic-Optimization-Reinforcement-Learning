0. Install PostgreSQL from http://www.postgresql.org/download/windows/
    -  Password, e.g., postgrespass01!

1. Launch psql client (keep pressing ENTER except for password)

2. Execute buold commands to construct MIMIC-III database:
    $ DROP DATABASE IF EXISTS mimic;
    $ CREATE DATABASE mimic OWNER postgres;
    $ \c mimic;
    $ CREATE SCHEMA mimiciii;
    $ set search_path to mimiciii;
    $ \i 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/buildmimic/postgres_create_tables.sql';
    $ \set ON_ERROR_STOP 1
    $ \set mimic_data_dir 'D:/mimic-iii-clinical-database-1.4'
    $ \i 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/buildmimic/postgres_load_data.sql';
    $ \i 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/buildmimic/postgres_add_indexes.sql';

3. Run MIMIC-Code commands to infer concepts (extra tables) from MIMIC-III:
    $ \cd 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/mimic-code/query'
    $ \i make-tables.sql
    $ \dt

4. Extract materialized RL-views to obtain CSV dataset
    $ \cd 'C:/Users/Uw naam/Desktop/Master Thesis Project VU/data/mimic-iii/mimic-luca'
    $ \i RL_views.sql

5. Save databases as csv files.
    $ \i databases_to_csv.sql
    
6. In new terminal navigate to `data/mimic-iii` and run convert_cv_to_mv.ipynb to convert CareVue tables to the simpler MetaVision format
    $ cd data/mimic-iii
    $ jupyter-lab
    (Run cells in notebook)
    