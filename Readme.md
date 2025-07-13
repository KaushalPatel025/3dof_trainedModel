# Seismic Response Prediction of 3DOF System

-The 3DOF system is trained with the help of synthetic data obtained from the OpenSees. 
-Total 22 gound motion database was considered from PEERNGA West. 
-The time histories are further scaled to obtain Max PGA of 0.2, 0.4, 0.6, 0.8, and 1.0 (g).
-The model is trained with the help of CatBoost ML model.
-It can be used for educational purpose to understand various concepts related to seismic engineering.

## How to Run
- Input the mass (m1, m2, m3) in kg and stiffness (k1, k2, k3) in N/m in a given jupyterlab file and run the code to obtain the roof displacement (in m) and maximum interstorey drift ratio.

## Requirements 
Numpy, Scipy, Pandas, Matplotlib, Tensorflow packages. 