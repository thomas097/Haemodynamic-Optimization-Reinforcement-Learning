DROP MATERIALIZED VIEW IF EXISTS rl_cohort CASCADE;
CREATE MATERIALIZED VIEW rl_cohort as

SELECT DISTINCT subject_id, 
                temp.hadm_id, 
                temp.icustay_id, 
		intime, 
                outtime, 
                suspected_infection_time_poe,
		-- unknown infection time makes window_start undetermined;
                -- we can use intime as proxy and leave infection time as NULL
		CASE WHEN suspected_infection_time_poe_days is NULL THEN intime
    		     ELSE suspected_infection_time_poe - interval '1 day'
		END AS window_start,
                -- when infection time unknown, window_end is 3 days after the intime of the admission
                CASE WHEN suspected_infection_time_poe_days is NULL THEN intime + interval '3 day'
    		     ELSE suspected_infection_time_poe + interval '2 day'
		END AS window_end,
                hospital_expire_flag
            	FROM (
    SELECT pc.hadm_id, pc.icustay_id, 
            pc.suspected_infection_time_poe,
            pc.suspected_infection_time_poe_days, sofa.sofa, 
            pc.intime, pc.outtime, pc.excluded,
            fpc.exclusion_secondarystay,
            fpc.exclusion_nonadult,
            fpc.exclusion_csurg,
            fpc.exclusion_carevue,
            fpc.exclusion_early_suspicion,
            fpc.exclusion_late_suspicion,
            fpc.exclusion_bad_data
    FROM patient_cohort as pc
    INNER JOIN sofa 
    	ON pc.hadm_id = sofa.hadm_id
    INNER JOIN full_cohort as fpc
    	on fpc.icustay_id = pc.icustay_id
    WHERE exclusion_nonadult = 0
	-- AND pc.suspected_infection_time_poe_days is NOT NULL AND sofa.sofa >= 2 -- we do not want to exclude any patients on sepsis!
    	-- AND excluded = 0
    	AND exclusion_bad_data = 0
    	AND exclusion_secondarystay = 0
    )
   as temp
INNER JOIN mimiciii.admissions
ON temp.hadm_id = admissions.hadm_id;

COMMIT;