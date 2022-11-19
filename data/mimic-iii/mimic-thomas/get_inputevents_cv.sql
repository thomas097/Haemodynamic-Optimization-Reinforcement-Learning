DROP TABLE IF EXISTS inputevents_cv2 CASCADE;
CREATE TABLE inputevents_cv2 AS
(
SELECT input.subject_id, input.hadm_id, 
input.icustay_id, input.charttime, 
input.itemid, input.amount, 
input.amountuom, input.rate, 
input.rateuom, input.storetime, input.orderid, input.linkorderid -- new
FROM inputevents_cv input 
INNER JOIN public.rl_cohort on input.subject_id = public.rl_cohort.subject_id 
ORDER BY subject_id ASC
)