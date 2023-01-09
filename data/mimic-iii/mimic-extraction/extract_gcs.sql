DROP TABLE IF EXISTS gcs_cohort CASCADE;
CREATE TABLE gcs_cohort AS
(
SELECT * FROM mimiciii.gcs
)