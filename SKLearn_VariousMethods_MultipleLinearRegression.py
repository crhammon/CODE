import numpy as np 
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
import sklearn.metrics as metrics
import math as m
import statsmodels
from regression_results import error_stats

#---------------------------------------------------------------
## Nguyen DATA
#---------------------------------------------------------------

pH = np.array([8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,7.65,7.65,7.65,7.65,7.65,7.65,7.65,7.65,7.65,7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,]) # S.U.
Si = np.array([33,53,73,33,53,73,33,53,73,33,53,73,33,53,73,33,53,73,33,53,73,33,53,73,33,53,73]) # Silica as mg/L as SiO2
V = np.array([21,41,61,41,61,21,61,21,41,41,61,21,61,21,41,21,41,61,61,21,41,21,41,61,41,61,21]) # Vanadium, ug/L as V
P = np.array([55,55,55,155,155,155,105,105,105,105,105,105,55,55,55,155,155,155,155,155,155,105,105,105,55,55,55]) # ug/L as P
As = np.array([15,15,15,35,35,35,55,55,55,15,15,15,35,35,35,55,55,55,15,15,15,35,35,35,55,55,55]) # ug/L as As
BV10_GFH = np.log(np.array([30000,16900,12500,13300,9200,9000,10200,6600,5300,51000,25200,23000,26300,20700,16000,19700,11100,8300,96000,88000,55000,65000,40600,29900,42800,29200,27100]))
BV10_E33 = np.log(np.array([30100,17100,13900,7200,5500,5100,8500,6000,4000,31500,20100,17500,19300,16000,15100,10100,7500,7200,33500,31200,30900,21900,18300,16100,22600,16500,14100]))
BV10_MET = np.log(np.array([13500,8800,4600,5000,2300,2000,4300,1500,1400,22100,14900,7200,13900,8400,5800,7700,5000,3100,38900,36800,29000,25900,18600,12000,16600,15000,11700]))

Xp = np.array([pH,Si,V,P,As])
#print('Xp:'+ str(Xp.shape))
X = Xp.T
#print('X:'+ str(X.shape))

#---------------------------------------------------------------
## Design Criteria!
#---------------------------------------------------------------
pH_m = 7.7 # S.U.
Si_m = 17 # mg/L as SiO2
V_m = 3.4 # ug/L as V
P_m = 80*30.974/(4*15.999+30.974) # ug/L as P (coversion shown here)
As_m = 23 # ug/L as As

#---------------------------------------------------------------
## GFH
#---------------------------------------------------------------
print("For ordinary least squares regression on log-transformed BV10s: ")
print()
print("For GFH:")
lm = linear_model.LinearRegression()
model = lm.fit(X,BV10_GFH)
PredictionsORD_GFH = lm.predict(X)
#print(PredictionsORD_GFH)
print()
print("Data:")
print(np.exp(BV10_GFH))
print()
print("Predictions:")
print(np.around(np.exp(PredictionsORD_GFH)))
print()
error_stats(np.exp(BV10_GFH), np.exp(PredictionsORD_GFH))

Coef = lm.coef_

BV10_m = lm.intercept_ + Coef[0]*pH_m + Coef[1]*Si_m + Coef[2]*V_m + Coef[3]*P_m+Coef[4]*As_m	

print("Bed volumes to breakthrough for GFH: " +str(np.around(np.exp(BV10_m))))

#---------------------------------------------------------------
## E33
#---------------------------------------------------------------
print()
print("For E33:")
lm = linear_model.LinearRegression()
model = lm.fit(X,BV10_E33)
PredictionsORD_E33 = lm.predict(X)
#print(PredictionsORD_GFH)
print()
print("Data:")
print(np.exp(BV10_E33))
print()
print("Predictions:")
print(np.around(np.exp(PredictionsORD_E33)))
print()
error_stats(np.exp(BV10_E33), np.exp(PredictionsORD_E33))

Coef = lm.coef_

BV10_m_E33 = lm.intercept_ + Coef[0]*pH_m + Coef[1]*Si_m + Coef[2]*V_m + Coef[3]*P_m+Coef[4]*As_m	

print("Bed volumes to breakthrough for E33: " +str(np.around(np.exp(BV10_m_E33))))


#---------------------------------------------------------------
## METSORB
#---------------------------------------------------------------
print()
print("For METSORB:")
lm = linear_model.LinearRegression()
model = lm.fit(X,BV10_MET)
PredictionsORD_MET = lm.predict(X)
#print(PredictionsORD_GFH)
print()
print("Data:")
print(np.exp(BV10_MET))
print()
print("Predictions:")
print(np.around(np.exp(PredictionsORD_MET)))
print()
error_stats(np.exp(BV10_MET), np.exp(PredictionsORD_MET))

Coef = lm.coef_

BV10_m_MET = lm.intercept_ + Coef[0]*pH_m + Coef[1]*Si_m + Coef[2]*V_m + Coef[3]*P_m+Coef[4]*As_m	

print("Bed volumes to breakthrough for METSORB: " +str(np.around(np.exp(BV10_m_MET))))
print()
print("SUMMARY")
print("Bed volumes to breakthrough for GFH: " +str(np.around(np.exp(BV10_m))))
print("Bed volumes to breakthrough for E33: " +str(np.around(np.exp(BV10_m_E33))))
print("Bed volumes to breakthrough for METSORB: " +str(np.around(np.exp(BV10_m_MET))))

#---------------------------------------------------------------
## COST CALCULATION - Data from Table 2-10 in EPA document (2011)
#---------------------------------------------------------------

<<<<<<< Updated upstream
# Currently in google sheets
=======
Cost_GFH = np.around(240*35.315) # $/m3
Cost_E33 = np.around(156*35.315) # $/m3
Cost_MET = np.around(3900/200*1000) # $/m3

BV_GFH = 10320/1000 # m3/BV, bed volume of the large scale reactor for GFH
BV_E33 = 10320/1000 # m3/BV
BV_MET = 9840/1000  # m3/BV

Q = 900*60*24*365/264.172 # m3/year

BV_Treated_GFH = Q/BV_GFH # BV/year
BV_Treated_E33 = Q/BV_E33 # BV/year
BV_Treated_MET = Q/BV_MET # BV/year

Time_to_repl_GFH = np.around(BV10_m/BV_Treated_GFH,2) # Time to replacement for media, years
Time_to_repl_GFH_c = np.around(BV10_m_c/BV_Treated_GFH,2)

Time_to_repl_E33 = np.around(BV10_m/BV_Treated_E33,2) # Time to replacement for media, years
Time_to_repl_E33_c = np.around(BV10_m_c/BV_Treated_E33,2)

Time_to_repl_MET = np.around(BV10_m/BV_Treated_MET,2) # Time to replacement for media, years
Time_to_repl_MET_c = np.around(BV10_m_c/BV_Treated_MET,2)

Cost_per_year_GFH = np.around(BV_GFH*Cost_GFH/Time_to_repl_GFH) # $/year
Cost_per_year_GFH_c = np.around(BV_GFH*Cost_GFH/Time_to_repl_GFH_c) # $/year

Cost_per_year_E33 = np.around(BV_E33*Cost_E33/Time_to_repl_E33) # $/year
Cost_per_year_E33_c = np.around(BV_E33*Cost_E33/Time_to_repl_E33_c) # $/year

Cost_per_year_MET = np.around(BV_MET*Cost_MET/Time_to_repl_MET) # $/year
Cost_per_year_MET_c = np.around(BV_MET*Cost_MET/Time_to_repl_MET_c) # $/year


    
x = PrettyTable()

x.field_names = ["Parameter", "GFH", "E33", "MetSorb"]

x.add_row(["R2", r2_GFH, r2_E33, r2_MET])
x.add_row(["MAE", MAE_GFH, MAE_E33, MAE_MET])
x.add_row(["BV to breakthrough, extrapolated", BV10_m, BV10_m_E33, BV10_m_MET])
x.add_row(["BV to breakthrough, conservative", BV10_m_c, BV10_m_E33_c, BV10_m_MET_c])
x.add_row(["Unit Cost ($/m3)", Cost_GFH, Cost_E33, Cost_MET])
x.add_row(["Time to replacement, extrapolated (months)", Time_to_repl_GFH, Time_to_repl_E33, Time_to_repl_MET])
x.add_row(["Time to replacement, conservative (months)", Time_to_repl_GFH_c, Time_to_repl_E33_c, Time_to_repl_MET_c])
x.add_row(["Media Cost, extrapolated ($/year)", Cost_per_year_GFH, Cost_per_year_E33, Cost_per_year_MET])
x.add_row(["Media Cost, conservative ($/year)", Cost_per_year_GFH_c, Cost_per_year_E33_c, Cost_per_year_MET_c])


print(x)
>>>>>>> Stashed changes

################################################################
################ OTHER METHODS BELOW, NOT USED #################
################################################################


## Which models were testedd determined from: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html


# #---------------------------------------------------------------
# ## Ridge
# #---------------------------------------------------------------

# ridge = linear_model.Ridge(alpha = 0.5)
# ridge_fit = ridge.fit(X,BV10)
# #print("For ridge regression: ")
# Predictions = ridge.predict(X)
# #print(Predictions)
# #error_stats(BV10, Predictions)

# #---------------------------------------------------------------
# ## Lasso
# #---------------------------------------------------------------

# lasso = linear_model.Lasso(alpha = 1)
# lasso_fit = lasso.fit(X,BV10)
# #print("For lasso regression: ")
# Predictions_lasso = lasso.predict(X)
# #print(Predictions_lasso)
# #error_stats(BV10, Predictions_lasso)

# #---------------------------------------------------------------
# ## Elastic Net
# #---------------------------------------------------------------

# EN = linear_model.ElasticNet(alpha = .1)
# EN_fit = EN.fit(X,BV10)
# #print("For Elastic Net regression: ")
# Predictions_EN = EN.predict(X)
# #print(Predictions_lasso)
# #error_stats(BV10, Predictions_EN)

# #---------------------------------------------------------------
# ## Stochastic Gradient Descent
# #---------------------------------------------------------------

<<<<<<< Updated upstream

=======
# SGD = SGDRegressor(loss="huber", penalty="l2", max_iter=500000)
# SGD.fit(X,BV10)
# #print("For Stochastic Gradient Descent: ")
# Predictions_SGD = SGD.predict(X)
# #print(Predictions_lasso)
# #error_stats(BV10, Predictions_SGD)
>>>>>>> Stashed changes
