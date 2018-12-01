def predict_userbased(test_user,train_mat,variance_adjusted_train_mat,item_no,k = 20):
#	filter out rows from train_mat that have 0 at item_no column.
#	print(train_mat[(train_mat[:,item_no] > 0), :].shape);
	test_user_l2_norm = np.sqrt(np.sum(np.multiply(test_user,test_user),axis=1))[0];
	test_user_stacked = np.repeat(test_user,train_mat.shape[0],axis=0);
	train_l2_norms = np.sqrt(np.sum(np.multiply(variance_adjusted_train_mat,variance_adjusted_train_mat),axis=1,keepdims=True));
	cosine_sims = np.divide(np.sum(np.multiply(test_user_stacked,variance_adjusted_train_mat),axis=1,keepdims=True),train_l2_norms)/test_user_l2_norm;
	corrated = np.sum(((test_user_stacked != 0) == (train_mat != 0)),axis=1,keepdims=True);
	adjusted_similarities = np.multiply(cosine_sims,corrated);
	normalized_adjusted_similarities = adjusted_similarities / (np.sum(adjusted_similarities,axis=0)[0]);
	top_k_indices = np.argpartition(-1 * normalized_adjusted_similarities,k,axis=0)[:k,0];
	sum_normalized_adjusted_similarities = np.sum(normalized_adjusted_similarities[top_k_indices],axis=0)[0];
#	print(top_k_indices.shape);
#	print(normalized_adjusted_similarities.shape);
#	print(normalized_adjusted_similarities[top_k_indices]);
	pred =  np.sum(np.multiply(train_mat[top_k_indices,item_no],normalized_adjusted_similarities[top_k_indices]),axis = 0)[0]/sum_normalized_adjusted_similarities;
	if pred == 0:
		return 2;
	return pred;

def give_predictions_on_test_user_userbased(train_mat,test_user,variance_adjusted_train_mat,top_k,num_predictions):
	is_predicted_filter = test_user[test_user >= 3.9];
	is_predicted_filter = keep_random_k_trues(is_predicted_filter,num_predictions);
	new_test_user = np.copy(test_user);
	new_test_user[is_predicted_filter] = 0;
	predictions = np.zeros(is_predicted_filter.shape,dtype=float);

	for i in range(test_user.shape[1]):
		if is_predicted_filter[0,i] == True:
			prediction = predict_userbased(new_test_user,train_mat,variance_adjusted_train_mat,i,top_k);
			predictions[0,i] = prediction;

#	print(top_k_indices.shape);
#	print(top_k_indices.shape);
	return (predictions,is_predicted_filter);
