
cohort.csv
- icustay_id
- window_start
- window_end
- hospital_expire_flag (0 = survived; 1 = deceased)

vassopressors_mv_cohort.csv
- icustay_id
- mcgkgmin

inputevents_mv_cohort.csv
- icustay_id
- itemid (always -1 to prevent conflict with MIMIC-III)
- starttime
- endtime
- amount
- amountuom
- ordercategoryname (always 'IV fluid' to prevent conflict with MIMIC-III)

vitals_cohort.csv
- icustay_id
- charttime
- vital_id: HeartRate, SpO2, TempC, DiasBP, MeanBP, SysBP, Glucose, RespRate
- valuenum

labs_cohort.csv
- icustay_id
- charttime
- lab_id: ALAT, ANOIN GAP, ASAT, BICARBONATE, BILIRUBIN, BUN, CALCIUM, CHLORIDE, CREATININE, GLUCOSE, HEMOGLOBIN, MAGNESIUM, PLATELET, POTASSIUM, SODIUM, WBC, PT, PTT, BaseExcess, LACTATE, PACO2, PAO2, PH, ALBUMIN, BANDS, ION_CALCIUM	
- valuenum

urineoutput_cohort.csv
- icustay_id
- charttime
- value
