import json;
import matplotlib,os;
matplotlib.use('Agg');
import matplotlib.pyplot as plt;

def plot_histograms(lambdas,densitys,c_kl):
	plt.clf();
	cols = ['r','b','g','black','y'][:len(lambdas)];
	for lamb,density,col in zip(lambdas,densitys,cols):
		density = density + [0] * (len(c_kl) - len(density));
		plt.plot(c_kl,density,c=col,label='λ = {}'.format(lamb));
	plt.xlabel('calibration metric C_KL');
	plt.ylabel('density');
	plt.legend();
#	print(top_k_indices.shape);
#	print(normalized_adjusted_similarities.shape);
	plt.savefig('../data/plots/histogram.png',format='png');
	plt.clf();

def plot_histograms_from_main(dic):
	with open('../data/jsons/lambdas.json','rt') as f_i:
		lambdas = json.loads(f_i.read());
	with open('../data/jsons/densitys.json','rt') as f_i:
		densitys = [];
		for line in f_i:
			densitys.append(json.loads(line));
	c_kl = [0.0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.5];
	plot_histograms(lambdas,densitys,c_kl);
