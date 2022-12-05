
cohort.csv
- icustay_id
- window_start
- window_end
- hospital_expire_flag (0 = discharged; 1 = deceased in-hospital)

vassopressors_mv_cohort.csv
- icustay_id
- starttime
- endtime
- mcgkgmin

inputevents_mv_cohort.csv
- icustay_id
- itemid (always -1 to prevent conflict with MIMIC-III)
- starttime
- endtime
- amount
- amountuom
- ordercategoryname (e.g. 'infuus - Crystalloid')

vitals_cohort.csv
- icustay_id
- charttime
- vital_id: HeartRate, SpO2, TempC, DiasBP, MeanBP, SysBP, Glucose, RespRate, SvO2, CVP
- valuenum

vitals_cohort.csv
- icustay_id
- starttime
- endtime
- sepsis_antibiotics (are antibiotics prescribed commonly used to combat sepsis?)
- profyl_antibiotics (profylactic administration of antibiotics)
- profyl_other (anticoagulatants?)

labs_cohort.csv
- icustay_id
- charttime
- lab_id: ALAT, ANION GAP, ASAT, BICARBONATE, BILIRUBIN, BUN, CALCIUM, CHLORIDE, CREATININE, GLUCOSE, HEMOGLOBIN, MAGNESIUM, PLATELET, POTASSIUM, SODIUM, WBC, PT, PTT, BaseExcess, LACTATE, PACO2, PAO2, PH, ALBUMIN, BANDS, ION_CALCIUM, HEMATOCRIT, D-Dimer
- valuenum

urineoutput_cohort.csv
- icustay_id
- charttime
- value

fio2_cohort.csv
- icustay_id
- charttime
- fio2

demographics_cohort.csv
- icustay_id
- age
- is_male
- height
- weight
- vent
- sofa (on admission)
- sirs (on admission)
- is_sepsis (admitted with sepsis diagnosis)
- is_elective (elective admissions after surgery)
