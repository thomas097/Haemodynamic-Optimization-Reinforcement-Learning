DROP TABLE IF EXISTS vasopressors_mv2 CASCADE;
CREATE TABLE vasopressors_mv2 AS
(
select mimiciii.inputevents_mv.icustay_id, starttime, endtime, rate, rateuom, patientweight, orderid, linkorderid,
	case 
		when itemid in (30120,221906,30047) and rateuom='mcg/kg/min' then round(cast(rate as numeric),4)  -- norad
		when itemid in (30120,221906,30047) and rateuom='mcg/min' then round(cast(rate/patientweight as numeric),4)  -- norad
		when itemid in (30119,221289) and rateuom='mcg/kg/min' then round(cast(rate as numeric),4) -- epi
		when itemid in (30119,221289) and rateuom='mcg/min' then round(cast(rate/patientweight as numeric),4) -- epi
		when itemid in (30051,222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),4) -- vasopressin, in U/h
		when itemid in (30051,222315) and rateuom='units/min' then round(cast(rate*5 as numeric),4) -- vasopressin
		when itemid in (30051,222315) and rateuom='units/hour' then round(cast(rate*5/60 as numeric),4) -- vasopressin
		when itemid in (30128,221749,30127) and rateuom='mcg/kg/min' then round(cast(rate*0.45 as numeric),4) -- phenyl
		when itemid in (30128,221749,30127) and rateuom='mcg/min' then round(cast(rate*0.45 / patientweight as numeric),4) -- phenyl
		when itemid in (221662,30043,30307) and rateuom='mcg/kg/min' then round(cast(rate*0.01 as numeric),4)  -- dopa
		when itemid in (221662,30043,30307) and rateuom='mcg/min' then round(cast(rate*0.01/ patientweight as numeric),4) -- dopa
        	else null 
	end as mcgkgmin,
	case
		when itemid in (30120,221906,30047) then 'norepinephrine'
		when itemid in (30119,221289) then 'epinephrine'
		when itemid in (30051,222315) then 'vasopressin'
		when itemid in (30128,221749,30127) then 'phenylephrine'
		when itemid in (221662,30043,30307) then 'dopamine'
        	else null 
	end as vasodrug

	from mimiciii.inputevents_mv
  INNER JOIN public.rl_cohort ON mimiciii.inputevents_mv.icustay_id = public.rl_cohort.icustay_id
  where itemid in
  (30128,30120,30051,221749,221906,30119,30047,30127,221289,222315,221662,30043,30307)
  and statusdescription != 'Rewritten' -- only valid orders
  Order by Icustay_id, StartTime
)