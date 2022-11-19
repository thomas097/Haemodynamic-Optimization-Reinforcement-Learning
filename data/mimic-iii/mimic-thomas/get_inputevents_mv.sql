DROP TABLE IF EXISTS inputevents_mv2 CASCADE;
CREATE TABLE inputevents_mv2 AS
(
SELECT input.subject_id, input.hadm_id, 
	input.icustay_id, input.starttime, 
	input.endtime, input.itemid, input.amount, 
	input.amountuom, input.rate, input.rateuom, 
	input.storetime, input.orderid, linkorderid,
	input.ordercategoryname, input.secondaryordercategoryname, 
	input.ordercategorydescription, input.patientweight, 
	input.totalamount, input.totalamountuom, 
	input.statusdescription
FROM mimiciii.inputevents_mv input 
INNER JOIN public.rl_cohort on input.subject_id = public.rl_cohort.subject_id
ORDER BY subject_id ASC
)