def create_prior_matrices(dimensions,genres):
	prior_mat_p = np.zeros(dimensions);
	prior_mat_q = np.zeros(dimensions);
	prior_mat_p[np.arange(dimensions)] = genres;
	prior_mat_q[np.arange(dimensions)] = genres;
	return (prior_mat_p,prior_mat_q);

def calculate_recall_at_t(predictions,is_predicted_filter,t);
	is_top_t_prediction = np.zeros(predictions.shape,dtype=bool);
	is_top_t_prediction[predictions.argsort()[::-1][:,t]] = True;
	return np.average(np.dot(is_top_t_prediction,is_top_t_prediction));

def calibrate_recs(recs,recs_filter,lamb):
	prior_p,prior_q = create_prior_matrices(dimensions,genres);
	for i in range(recs.shape[0]):
		ret_recs = np.dot((1-lamb),np.multiply(recs_filter,recs)) + lamb * (prior_p * np.log(np.sum(prior_q)));	#closed form solution of the optimization function.
def calculate_calibration_at_t(recs,recs_filter,t,lamb):
#	print(top_k_indices.shape);
#	print(normalized_adjusted_similarities.shape);
	recs = calibrate_recs(recs,recs_filter,lamb);
	return np.sum(prior_mat_p[u][g] * log(prior_mat_p[u][g]/prior_mat_q_tilde[u][g]));
