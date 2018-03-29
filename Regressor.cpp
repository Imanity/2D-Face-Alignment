#include "Regressor.h"

using namespace std;
using namespace cv;

Regressor::Regressor(string config_file) {
	configFile = config_file;
}

Regressor::~Regressor() { }

void Regressor::train(Dataset &dataset, vector<vector<RandomForest>> &forests) {
	cout << endl << "======================== Training ========================" << endl << endl;

	// Get config
	int stage_num = 0, feature_point_num = 0;
	int tree_num_per_forest = forests[0][0].trees.size(), leaf_num_per_tree = forests[0][0].trees[0]->depth;
	fstream config(configFile, ios::in);
	if (!config) {
		cerr << "Error: Config not exist." << endl;
		return;
	}
	string line;
	while (getline(config, line)) {
		if (line.find("stage_num") != string::npos) {
			stage_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("feature_point_num") != string::npos) {
			feature_point_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
	}

	// Calculate S_0
	vector<Point2D> S;
	dataset.generate_S_0(S, configFile);

	// showImgWithLandmarks(dataset.data[0].imagePath, S);
	
	for (int stage_id = 0; stage_id < stage_num; ++stage_id) {
		// Output info
		cout << "Training stage: " << stage_id << endl;
		time_t start_time, end_time;
		time(&start_time);

		// Prepare params for regressor
		struct feature_node **binary_features;
		binary_features = new struct feature_node*[dataset.data.size()];
		for (int i = 0; i < dataset.data.size(); ++i) {
			binary_features[i] = new feature_node[tree_num_per_forest * feature_point_num + 1];
		}
		int index = 0, f_id = 0;
		for (int i = 0; i < dataset.data.size(); ++i) {
			cout << "Image " << i << endl;
			cv::Mat_<uchar> image = imread(dataset.data[i].imagePath, 0);
			f_id = 0;
			for (int j = 0; j < feature_point_num; ++j) {
				for (int k = 0; k < tree_num_per_forest; ++k) {
					int leaf_id = 0;
					DecisionTree tree = forests[j][stage_id].trees[k];
					leaf_id = classifyImgByTree(tree, image, S[j]);
					binary_features[i][f_id].index = index + k * leaf_num_per_tree + leaf_id;
					binary_features[i][f_id].value = 1.0;
					f_id++;
				}
				index += leaf_num_per_tree * tree_num_per_forest;
			}
			binary_features[i][f_id].index = -1;
			binary_features[i][f_id].value = -1.0;
		}
		cout << "Read finished." << endl;

#pragma omp parallel for
		for (int feature_id = 0; feature_id < feature_point_num; ++feature_id) {
			// Calculate delta S
			vector<Point2D> deltaS;
			for (int i = 0; i < dataset.data.size(); ++i) {
				deltaS.push_back(Point2D(dataset.data[i].groundTruth[feature_id].x / (double)dataset.data[i].img_w - S[feature_id].x, 
					dataset.data[i].groundTruth[feature_id].y / (double)dataset.data[i].img_h - S[feature_id].y));
			}
			double *target_x = new double[dataset.data.size()];
			double *target_y = new double[dataset.data.size()];
			for (int i = 0; i < dataset.data.size(); ++i) {
				target_x[i] = deltaS[i].x;
				target_y[i] = deltaS[i].y;
			}

			// Get Problem
			struct problem * prob = new struct problem;
			prob->l = dataset.data.size();
			prob->n = tree_num_per_forest * leaf_num_per_tree * feature_point_num;
			prob->x = binary_features;
			prob->y = target_x;
			prob->bias = -1.0;

			// Get params
			struct parameter* params = new struct parameter;
			params->solver_type = L2R_L2LOSS_SVR_DUAL;
			params->C = 1.0 / dataset.data.size();
			params->p = 0;

			// Linear Regression

			// TODO

			// Release
			delete[] target_x;
			delete[] target_y;
		}
		// Release
		for (int i = 0; i < dataset.data.size(); i++) {
			delete[] binary_features[i];
		}
		delete[] binary_features;

		time(&end_time);
		cout << "Training stage " << stage_id << " finished. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl << endl;
	}
}

int Regressor::classifyImgByTree(DecisionTree tree, cv::Mat_<uchar> &image, Point2D now_landmark_pos) {
	int leaf_id = 0;
	TreeNode *p = tree;
	while (p->depth > 1) {
		Point2D p_1 = p->feature.first;
		Point2D p_2 = p->feature.second;
		double p_1_x = now_landmark_pos.x + p_1.x;
		double p_1_y = now_landmark_pos.y + p_1.y;
		double p_2_x = now_landmark_pos.x + p_2.x;
		double p_2_y = now_landmark_pos.y + p_2.y;
		int pos_1_x = p_1_x * (double)image.cols;
		int pos_1_y = p_1_y * (double)image.rows;
		int pos_2_x = p_2_x * (double)image.cols;
		int pos_2_y = p_2_y * (double)image.rows;
		pos_1_x = pos_1_x < 0 ? 0 : pos_1_x;
		pos_1_x = pos_1_x >= image.cols ? (image.cols - 1) : pos_1_x;
		pos_1_y = pos_1_y < 0 ? 0 : pos_1_y;
		pos_1_y = pos_1_y >= image.rows ? (image.rows - 1) : pos_1_y;
		pos_2_x = pos_2_x < 0 ? 0 : pos_2_x;
		pos_2_x = pos_2_x >= image.cols ? (image.cols - 1) : pos_2_x;
		pos_2_y = pos_2_y < 0 ? 0 : pos_2_y;
		pos_2_y = pos_2_y >= image.rows ? (image.rows - 1) : pos_2_y;
		int sub_val = (int)image(pos_1_y, pos_1_x) - (int)image(pos_2_y, pos_2_x);
		if (sub_val < p->threshold) {
			p = p->left_child;
		}
		else {
			leaf_id += pow(2.0, p->depth - 2);
			p = p->right_child;
		}
	}
	return leaf_id;
}
