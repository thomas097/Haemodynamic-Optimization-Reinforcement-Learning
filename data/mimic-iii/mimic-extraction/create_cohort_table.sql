DROP MATERIALIZED VIEW IF EXISTS rl_cohort CASCADE;
CREATE MATERIALIZED VIEW rl_cohort as

SELECT DISTINCT subject_id, 
                temp.hadm_id, 
                temp.icustay_id, 
		intime, 
                outtime, 
                suspected_infection_time_poe,
		intime AS window_start,
                intime + interval '7 day' AS window_end, -- we cap extremely long trajectories to prevent degenerate OPE
		suspected_infection_time_poe IS NOT NULL AS is_sepsis,
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
    	-- AND excluded = 0
    	AND exclusion_bad_data = 0
    	AND exclusion_secondarystay = 0
    )
   as temp
INNER JOIN mimiciii.admissions
ON temp.hadm_id = admissions.hadm_id;

COMMIT;