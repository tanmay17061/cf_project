from dataset import generate_matrix;
import numpy as np;
import matplotlib,os;
matplotlib.use('Agg');
import matplotlib.pyplot as plt;
from plot import plot_histograms_from_main;

def keep_random_k_trues(mat,k):
	ret_mat = np.zeros(mat.shape,dtype=bool);
	trues_kept = 0;
	for i in range(ret_mat.shape[0]):
		if np.random.rand() > 0.5:
			ret_mat[i,:] = True;
			trues_kept += 1;
		if trues_kept >= k:
			break;
	return ret_mat;

def give_stats_on_split(test_mat,train_mat,top_k):
	ctr = 0;
	count_preds = 0;
	var_vector = np.var(train_mat,axis=1,keepdims=True)
	max_var = np.max(var_vector);
	min_var = np.min(var_vector);
	variance_adjusted_train_mat = (train_mat - min_var) / (max_var - min_var);	#used only for user-based.
	for lamb in [0,0.2,0.5,0.8,0.9,0.95,0.99,0.999]:
		print('lambda =',lamb);
		sum_recall_at_10,sum_recall_at_50,sum_calibration_at_10,sum_calibration_at_50 = 0,0,0,0;
		for test_user in test_mat:
			test_user = test_user.reshape(1,-1);
			(test_user_predictions,test_user_is_predicted_filter) = give_predictions_on_test_user_userbased(train_mat,test_user,variance_adjusted_train_mat,top_k,100);
			test_user_recall = calculate_recall_at_t(test_user_predictions,test_user_is_predicted_filter,t);
			(recs,recs_filter) = calibrate_recommendations(test_user_predictions,test_user_is_predicted_filter,lamb,)
			sum_recall_at_10 += calculate_recall_at_t(recs,recs_filter,10);
			sum_recall_at_50 += calculate_recall_at_t(recs,recs_filter,50);
			#	print(top_k_indices.shape);
			#	print(normalized_adjusted_similarities.shape);
			sum_calibration_at_10 += calculate_calibration_at_t(recs,recs_filter,10,lamb);
			sum_calibration_at_50 += calculate_calibration_at_t(recs,recs_filter,50,lamb);
		(avg_recall_at_10,avg_recall_at_50,avg_calibration_at_10,avg_calibration_at_50) = (sum_recall_at_10/test_mat.shape[0],sum_recall_at_50/test_mat.shape[0],sum_calibration_at_10/test_mat.shape[0],sum_calibration_at_50/test_mat.shape[0]);
		print(avg_recall_at_10,avg_recall_at_50,avg_calibration_at_10,avg_calibration_at_50);
def give_stats_on_folds(mat,num_folds,top_k,is_user_based = True):
	block_size = mat.shape[0] // num_folds;
	preds,aes = np.zeros((num_folds,)),np.zeros((num_folds,));
	for fold_num in range(num_folds):
		test_mat = mat[fold_num * block_size : min((fold_num + 1) * block_size,mat.shape[0])];
		train_mat = np.vstack((mat[:fold_num * block_size],mat[min((fold_num + 1) * block_size,mat.shape[0]):]));
		print('fold_num = ',fold_num);
		ae = give_stats_on_split(test_mat,train_mat,top_k);
		aes[fold_num] = ae[0];
		preds[fold_num] = ae[1];
	return np.sum(np.divide(aes,preds)) / num_folds;

def give_stats_for_top_k(k,mat,folds,is_user_based = True):
	return give_stats_on_folds(mat,folds,k,is_user_based);


if __name__ == '__main__':
#	mat = generate_matrix('../data/ml-1m/u.data');
	stats_baseline = list();
	stats_calibrated = list();
	top_ks = [10,20,30,40,50];
#	for top_k in top_ks:
#		stats_baseline.append(give_stats_for_top_k(top_k,mat,5,True));
#		stats_calibrated.append(give_stats_for_top_k(top_k,mat,5,False));
#	print(top_ks);
#	print(stats_baseline);
#	print(stats_calibrated);
	plot_histograms_from_main({'stats_baseline':stats_baseline,'stats_calibrated':stats_calibrated,'top_ks':top_ks});
