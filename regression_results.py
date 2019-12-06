def error_stats(y_true, y_pred):
	import sklearn.metrics as metrics
	import numpy as np
	# Regression metrics
	explained_variance=metrics.explained_variance_score(y_true, y_pred)
	mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
	mse=metrics.mean_squared_error(y_true, y_pred) 
	median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
	r2=metrics.r2_score(y_true, y_pred)

	print('explained_variance: ' + str(round(explained_variance,4)))    
	print('r2: ', round(r2,4))
	print('MAE: ', round(mean_absolute_error,4))
	print('MSE: ', round(mse,4))
	print('RMSE: ', round(np.sqrt(mse),4))
	print()
	print()