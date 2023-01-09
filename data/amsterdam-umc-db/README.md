In this folder the source code is provided for the extraction of patient trajectories from the AmsterdamUMCdb

## Instructions

0. Request access to the AmsterdamUMCdb database via https://amsterdammedicaldatascience.nl/amsterdamumcdb/

1. Run `DataExtraction_AmsterdamUMCdb_AllAdmissions.ipynb`
   - Make sure to change the path `OUT_DIR` to specify the desired output location and modify the paths of the data files to the directories where the source files are located

**Warning:** the AmsterdamUMCdb source files contain many gigabytes of data. While the data extraction has been designed to be memory-friendly, it is recommended to have at least 120GB of harddisk space available and have at least 8 GB of working memory

## Tables Generated
After running the data extraction pipeline, the following tables with the following columns is created

`cohort.csv`
- icustay_id
- window_start
- window_end
- hospital_expire_flag (0 = discharged; 1 = deceased in-hospital)

`vassopressors_mv_cohort.csv`
- icustay_id
- starttime
- endtime
- mcgkgmin

`inputevents_mv_cohort.csv`
- icustay_id
- itemid (always -1 to prevent conflict with MIMIC-III)
- starttime
- endtime
- amount
- amountuom
- ordercategoryname (e.g. 'infuus - Crystalloid')

`vitals_cohort.csv`
- icustay_id
- charttime
- vital_id: HeartRate, SpO2, TempC, DiasBP, MeanBP, SysBP, Glucose, RespRate, SvO2, CVP
- valuenum

`vitals_cohort.csv`
- icustay_id
- starttime
- endtime
- sepsis_antibiotics (are antibiotics prescribed commonly used to combat sepsis?)
- profyl_antibiotics (profylactic administration of antibiotics)
- profyl_other (anticoagulatants?)

`labs_cohort.csv`
- icustay_id
- charttime
- lab_id: ALAT, ANION GAP, ASAT, BICARBONATE, BILIRUBIN, BUN, CALCIUM, CHLORIDE, CREATININE, GLUCOSE, HEMOGLOBIN, MAGNESIUM, PLATELET, POTASSIUM, SODIUM, WBC, PT, PTT, BaseExcess, LACTATE, PACO2, PAO2, PH, ALBUMIN, BANDS, ION_CALCIUM, HEMATOCRIT, D-Dimer
- valuenum

`urineoutput_cohort.csv`
- icustay_id
- charttime
- value

`fio2_cohort.csv`
- icustay_id
- charttime
- fio2

`demographics_cohort.csv`
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
