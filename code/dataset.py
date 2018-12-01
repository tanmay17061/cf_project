import numpy as np;

def find_max_in_file_column(ratings_file_dir,col_num):
	title_passed = False;
	max_val = -999999;
	with open(ratings_file_dir,'rt') as f_i:
		for line in f_i:
			if not title_passed:
				title_passed = True;
				continue;
			data = line.split('\t');
			max_val = max(int(data[col_num]),max_val);
	return max_val-1;	#0-based indexing.

def generate_matrix(ratings_file_dir):
	max_user_id = find_max_in_file_column(ratings_file_dir,0);
	max_movie_id = find_max_in_file_column(ratings_file_dir,1);

	ret_matrix = np.zeros((max_user_id+1,max_movie_id+1),dtype=float);
	title_passed = False;
	with open(ratings_file_dir,'rt') as fi:
		for line in fi:
			if not title_passed:
				title_passed = True;
				continue;
			data = line.split('\t');
			user_id,movie_id,rating = int(data[0])-1,int(data[1])-1,float(data[2]);
			ret_matrix[user_id,movie_id] = rating;
	np.random.shuffle(ret_matrix)
#	ret_matrix[ret_matrix < 4] = 0;
#	ret_matrix[ret_matrix >= 4] = 1;
	return ret_matrix;

if __name__ == '__main__':
	mat = generate_matrix('../data/ml-1m/u.data');
	print(mat.shape);
	print(np.max(mat));
	print(np.min(mat));
