SUMMARY OF FINDINGS: 
1. WHAT WE SET OUT TO PROVE: Aerotech needs a reliable altitude control for a Navy demo. We built a baseline if-else heuristic controller along with a PID controller, then compared performance on the same 30s trajectory from .5 to 1.5m. 

PHASE 1-2 TAKEAWAYS:
We observed that sonar is clean and stable in the 0-2m range, with minimal drift. We saw that barometer measurements had a large offset and slow drift, which isn't strong for altitude control in the lower range. Thus we chose sonar as the reference height for control and evaluation in future phases. We chose the sonar based on the phase-2 fit for calibration of altitude changes having lower R^2 and RMSE compared to barometer: delta_height_sonar_cal = 0.352 * delta_sonar_m + 0.0241

PHASE 3-4 TAKEAWAYS:
The heuristic controller worked by taking the location with respect to the band to command the drone. Above the band -> climb, inside band -> hold, below band -> descend. This provided a fast rise time and simple control loop, however there was high overshoot and higher steady state error. 

The PID controller we implemented was a standard PID controller with Kp, Ki and Kd gains. We used a PID automated tuning framework.

Metric	If-Else	PID	Winner
Settling time [s]	∞	19.06	PID
Steady-state error [m]	0.153	0.041	PID
Rise time [s]	0.94	14.00	Heur.
Max overshoot [%]	35.3	1.3	PID
Control chattering [switches/min]	0.0	0.0	Tie
RMS error [m]	0.240	0.287	Heur.

Our key findings were that PID won 3/6 metrics. Most critically, it reduced overshoot from 35.3% to 1.3% and reduced steady state error 73%, from .153 to 0.041m. Heuristic shows a faster intial rise time and slightly lower full run RMS due to this, but at the cost of large overshoot and never settling on the correct value (one of the most critical performance attributes). For mission criteria nad safety (no oscillations and precise tracking), PID control is the correct choice. 

WHY PID? 
Selecting PID gives us a competitive edge. PID allows for low overshoot, finite settling at a target altitude and small steady state error, aligned with our requirements for smooth and precise flight. It is robust to drift and bias. Additionally, furhter PID tuning (increasing Kp and maybe Kd) could reduce RMS and shorten settling time without sacrificing stability. 
 