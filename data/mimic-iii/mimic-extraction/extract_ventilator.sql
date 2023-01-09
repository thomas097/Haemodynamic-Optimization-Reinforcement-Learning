DROP TABLE IF EXISTS vent_cohort CASCADE;
CREATE TABLE vent_cohort AS
(
SELECT icustay_id, starttime, endtime FROM public.ventdurations
)