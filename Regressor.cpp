#include "Regressor.h"

using namespace std;
using namespace cv;

Regressor::Regressor(Params_ params_) {
	params = params_;
}

Regressor::~Regressor() { }

void print_null(const char *s) {}

void Regressor::Train(Dataset &dataset) {
	cout << endl << "======================== Training ========================" << endl << endl;
	set_print_string_function(print_null);

	// Initialize model
	for (int i = 0; i < params.stage_num; ++i) {
		vector<struct model*> model_x_;
		vector<struct model*> model_y_;
		vector<RandomForest> forests_;
		model_x_.resize(params.landmark_num);
		model_y_.resize(params.landmark_num);
		forests_.resize(params.landmark_num);
		linear_model_x.push_back(model_x_);
		linear_model_y.push_back(model_y_);
		forests.push_back(forests_);
	}

	// Calculate S_0
	vector<Point2D> S_0;
	S_0 = dataset.S_0;
	vector<vector<Point2D>> S;
	for (int i = 0; i < dataset.data.size(); ++i) {
		S.push_back(S_0);
	}
	fstream fout("model/S_0.mdl", ios::out);
	for (int i = 0; i < S_0.size(); ++i) {
		fout << S_0[i].x << " " << S_0[i].y << endl;
	}
	fout.close();
	
	for (int stage_id = 0; stage_id < params.stage_num; ++stage_id) {
		// Output info
		cout << "Training stage: " << stage_id << endl;
		cout << "---------------------------------------------------------" << endl;
		time_t start_time, end_time;
		time(&start_time);

		// Get random features
		time_t currTime = time(0);
		cv::RNG rd(currTime);
		vector<pair<Point2D, Point2D>> features;
		for (int i = 0; i < params.feature_num; ++i) {
			double x1, y1, x2, y2;
			double r = params.local_region_size[stage_id];
			do {
				x1 = rd.uniform(-r, r);
				y1 = rd.uniform(-r, r);
			} while (x1 * x1 + y1 * y1 > r * r);
			do {
				x2 = rd.uniform(-r, r);
				y2 = rd.uniform(-r, r);
			} while (x2 * x2 + y2 * y2 > r * r);
			Point2D p1(x1, y1), p2(x2, y2);
			features.push_back(make_pair(p1, p2));
		}
		vector<vector<vector<int>>> img_vals;
		dataset.shapeIndexFeature(S, S_0, params.special_point_id, features, img_vals);

		cout << "\tGenerating Random Forests." << endl;

		// Train random forests
		for (int landmark_id = 0; landmark_id < params.landmark_num; ++landmark_id) {
			forests[stage_id][landmark_id].generateFromDataset(dataset, params, features, img_vals, landmark_id);
			stringstream filename;
			filename << "model/random_forest/stage_" << stage_id << "_landmark_" << landmark_id << ".forest";
			forests[stage_id][landmark_id].outputToFile(filename.str());
		}
		cout << "\tGenerate Finished." << endl;

		// Extract binary features
		struct feature_node **binary_features;
		binary_features = new struct feature_node*[dataset.data.size()];
		for (int i = 0; i < dataset.data.size(); ++i) {
			binary_features[i] = new feature_node[params.tree_num_per_forest * params.landmark_num + 1];
		}
		int leaf_num_per_tree = pow(2.0, params.tree_depth - 1);
		int index = 1, f_id = 0;
		for (int i = 0; i < dataset.data.size(); ++i) {
			f_id = 0;
			index = 1;
			for (int j = 0; j < params.landmark_num; ++j) {
				for (int k = 0; k < params.tree_num_per_forest; ++k) {
					int leaf_id = 0;
					leaf_id = classifyImgByTree(forests[stage_id][j].trees[k], img_vals[i][j]);
					binary_features[i][f_id].index = index + k * leaf_num_per_tree + leaf_id;
					binary_features[i][f_id].value = 1.0;
					f_id++;
				}
				index += leaf_num_per_tree * params.tree_num_per_forest;
			}
			binary_features[i][params.tree_num_per_forest * params.landmark_num].index = -1;
			binary_features[i][params.tree_num_per_forest * params.landmark_num].value = -1.0;
		}

		double *targets_x = new double[dataset.data.size()];
		double *targets_y = new double[dataset.data.size()];

		cout << "\tTraining Linear Models." << endl;

		for (int landmark_id = 0; landmark_id < params.landmark_num; ++landmark_id) {
			// Calculate delta S
			for (int i = 0; i < dataset.data.size(); ++i) {
				Transform_2D tran = getSimilarityTransform(S_0, S[i], params.special_point_id);
				Point2D delta_S = transformPoint(tran, Point2D(dataset.data[i].groundTruth[landmark_id].x - S[i][landmark_id].x, dataset.data[i].groundTruth[landmark_id].y - S[i][landmark_id].y));
				targets_x[i] = delta_S.x;
				targets_y[i] = delta_S.y;
			}

			// Get Problem
			struct problem * prob = new struct problem;
			prob->l = dataset.data.size();
			prob->n = params.tree_num_per_forest * leaf_num_per_tree * params.landmark_num;
			prob->x = binary_features;
			prob->y = targets_x;
			prob->bias = -1;

			// Get params
			struct parameter* param = new struct parameter;
			param->solver_type = L2R_L2LOSS_SVR_DUAL;
			param->C = 1.0 / dataset.data.size();
			param->p = 0;

			// Linear Regression

			check_parameter(prob, param);
			struct model* regression_model_x = train(prob, param);

			prob->y = targets_y;
			check_parameter(prob, param);
			struct model* regression_model_y = train(prob, param);

			// Save models
			stringstream sstream_x, sstream_y;
			sstream_x << "model/regressor/stage_" << stage_id << "_landmark_" << landmark_id << "_x.mdl";
			sstream_y << "model/regressor/stage_" << stage_id << "_landmark_" << landmark_id << "_y.mdl";
			save_model(sstream_x.str().c_str(), regression_model_x);
			save_model(sstream_y.str().c_str(), regression_model_y);
			linear_model_x[stage_id][landmark_id] = regression_model_x;
			linear_model_y[stage_id][landmark_id] = regression_model_y;

			if (landmark_id % 5 == 0) {
				cout << "\t\tStage " << stage_id << ": " << landmark_id << "/" << params.landmark_num << " finished." << endl;
			}
		}
		cout << endl << "\tTraining Finished." << endl;

		// Update S
		for (int i = 0; i < dataset.data.size(); ++i) {
			for (int j = 0; j < params.landmark_num; ++j) {
				Transform_2D tran = getSimilarityTransform(S[i], S_0, params.special_point_id);
				double deltaS_x = predict(linear_model_x[stage_id][j], binary_features[i]);
				double deltaS_y = predict(linear_model_y[stage_id][j], binary_features[i]);
				Point2D deltaS = transformPoint(tran, Point2D(deltaS_x, deltaS_y));
				S[i][j].x += deltaS.x;
				S[i][j].y += deltaS.y;
			}
		}

		// Release
		for (int i = 0; i < dataset.data.size(); ++i) {
			delete[] binary_features[i];
		}
		delete[] binary_features;
		delete[] targets_x;
		delete[] targets_y;

		time(&end_time);
		cout << "Training stage " << stage_id << " finished. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl;
		cout << "---------------------------------------------------------" << endl << endl;
	}
}

void Regressor::predictImage(string img_path) {
	cv::Mat_<uchar> image = imread(img_path, 0);

	// Get boundingbox
	CascadeClassifier haar_cascade;
	haar_cascade.load("model/haarcascade_frontalface_alt2.xml");
	std::vector<cv::Rect> faces;
	haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));

	int max_area_bbox_id = 0, max_area = 0;
	for (int i = 0; i < faces.size(); ++i) {
		if (faces[i].area() > max_area) {
			max_area = faces[i].area();
			max_area_bbox_id = i;
		}
	}
	cv::Rect bbox = faces[max_area_bbox_id];

	// Get S_0
	vector<Point2D> S_0, S;
	fstream fin("model/S_0.mdl", ios::in);
	if (!fin) {
		cerr << "S_0 model not exist!" << endl;
		return;
	}
	for (int i = 0; i < params.landmark_num; ++i) {
		double x, y;
		fin >> x >> y;
		S_0.push_back(Point2D(x, y));
	}
	S = S_0;

	// Cascade regression
	int leaf_num_per_tree = pow(2.0, params.tree_depth - 1);
	for (int stage_id = 0; stage_id < params.stage_num; ++stage_id) {
		// Get Image Feature
		struct feature_node *binary_feature;
		binary_feature = new feature_node[params.tree_num_per_forest * params.landmark_num + 1];
		int index = 1, f_id = 0;
		Transform_2D tran = getSimilarityTransform(S, S_0, params.special_point_id);
		for (int j = 0; j < params.landmark_num; ++j) {
			for (int k = 0; k < params.tree_num_per_forest; ++k) {
				int leaf_id = 0;
				DecisionTree tree = forests[stage_id][j].trees[k];
				leaf_id = classifyImgByTree(tree, image, bbox, S[j], tran);
				binary_feature[f_id].index = index + k * leaf_num_per_tree + leaf_id;
				binary_feature[f_id].value = 1.0;
				f_id++;
			}
			index += leaf_num_per_tree * params.tree_num_per_forest;
		}
		binary_feature[params.tree_num_per_forest * params.landmark_num].index = -1;
		binary_feature[params.tree_num_per_forest * params.landmark_num].value = -1.0;

		// Predict according to feature
		for (int j = 0; j < params.landmark_num; ++j) {
			double deltaS_x = predict(linear_model_x[stage_id][j], binary_feature);
			double deltaS_y = predict(linear_model_y[stage_id][j], binary_feature);
			Point2D deltaS = transformPoint(tran, Point2D(deltaS_x, deltaS_y));
			S[j].x += deltaS.x;
			S[j].y += deltaS.y;
		}
	}

	// Show image
	showImgWithLandmarks(img_path, S, faces, max_area_bbox_id);
}

void Regressor::predictImage(string img_path, cv::Rect bbox) {
	cv::Mat_<uchar> image = imread(img_path, 0);

	// Get S_0
	vector<Point2D> S_0, S;
	fstream fin("model/S_0.mdl", ios::in);
	if (!fin) {
		cerr << "S_0 model not exist!" << endl;
		return;
	}
	for (int i = 0; i < params.landmark_num; ++i) {
		double x, y;
		fin >> x >> y;
		S_0.push_back(Point2D(x, y));
	}
	S = S_0;

	// Cascade regression
	int leaf_num_per_tree = pow(2.0, params.tree_depth - 1);
	for (int stage_id = 0; stage_id < params.stage_num; ++stage_id) {
		// Get Image Feature
		struct feature_node *binary_feature;
		binary_feature = new feature_node[params.tree_num_per_forest * params.landmark_num + 1];
		int index = 1, f_id = 0;
		Transform_2D tran = getSimilarityTransform(S, S_0, params.special_point_id);
		for (int j = 0; j < params.landmark_num; ++j) {
			for (int k = 0; k < params.tree_num_per_forest; ++k) {
				int leaf_id = 0;
				DecisionTree tree = forests[stage_id][j].trees[k];
				leaf_id = classifyImgByTree(tree, image, bbox, S[j], tran);
				binary_feature[f_id].index = index + k * leaf_num_per_tree + leaf_id;
				binary_feature[f_id].value = 1.0;
				f_id++;
			}
			index += leaf_num_per_tree * params.tree_num_per_forest;
		}
		binary_feature[params.tree_num_per_forest * params.landmark_num].index = -1;
		binary_feature[params.tree_num_per_forest * params.landmark_num].value = -1.0;

		// Predict according to feature
		for (int j = 0; j < params.landmark_num; ++j) {
			double deltaS_x = predict(linear_model_x[stage_id][j], binary_feature);
			double deltaS_y = predict(linear_model_y[stage_id][j], binary_feature);
			Point2D deltaS = transformPoint(tran, Point2D(deltaS_x, deltaS_y));
			S[j].x += deltaS.x;
			S[j].y += deltaS.y;
		}
	}

	// Show image
	vector<Rect> faces;
	faces.push_back(bbox);
	showImgWithLandmarks(img_path, S, faces, 0);
}

void Regressor::readModels() {
	cout << "Reading Model..." << endl;
	time_t start_time, end_time;
	time(&start_time);

	forests.clear();
	linear_model_x.clear();
	linear_model_y.clear();

	// Read forests
	for (int i = 0; i < params.stage_num; ++i) {
		vector<RandomForest> rf;
		for (int j = 0; j < params.landmark_num; ++j) {
			stringstream sstream;
			sstream << "model/random_forest/stage_" << i << "_landmark_" << j << ".forest";
			RandomForest forest;
			forest.generateFromFile(sstream.str());
			rf.push_back(forest);
		}
		forests.push_back(rf);
	}

	// Read models
	for (int i = 0; i < params.stage_num; ++i) {
		vector<struct model*> model_x_;
		vector<struct model*> model_y_;
		for (int j = 0; j < params.landmark_num; ++j) {
			stringstream sstream_x, sstream_y;
			sstream_x << "model/regressor/stage_" << i << "_landmark_" << j << "_x.mdl";
			sstream_y << "model/regressor/stage_" << i << "_landmark_" << j << "_y.mdl";
			model_x_.push_back(load_model(sstream_x.str().c_str()));
			model_y_.push_back(load_model(sstream_y.str().c_str()));
		}
		linear_model_x.push_back(model_x_);
		linear_model_y.push_back(model_y_);
	}

	time(&end_time);
	cout << "Reading finished. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl << endl;
}

int Regressor::classifyImgByTree(DecisionTree tree, std::vector<int> &img_vals) {
	int leaf_id = 0;
	TreeNode *p = tree;
	while (p != NULL && p->depth > 1) {
		if (img_vals[p->feature_id] < p->threshold) {
			p = p->left_child;
		}
		else {
			leaf_id += pow(2.0, p->depth - 2);
			p = p->right_child;
		}
	}
	return leaf_id;
}

int Regressor::classifyImgByTree(DecisionTree tree, cv::Mat_<uchar> &image, cv::Rect &bbox, Point2D &landmark_pos, Transform_2D &t) {
	int leaf_id = 0;
	TreeNode *p = tree;
	while (p != NULL && p->depth > 1) {
		Point2D delta_S_1 = p->feature.first;
		Point2D delta_S_2 = p->feature.second;

		Point2D d1 = transformPoint(t, delta_S_1);
		Point2D d2 = transformPoint(t, delta_S_2);

		int x_1 = (d1.x + landmark_pos.x) * (double)bbox.width / 2.0 + (double)bbox.x + (double)bbox.width / 2.0;
		int y_1 = (d1.y + landmark_pos.y) * (double)bbox.height / 2.0 + (double)bbox.y + (double)bbox.height / 2.0;
		x_1 = x_1 < 0 ? 0 : x_1;
		y_1 = y_1 < 0 ? 0 : y_1;
		x_1 = x_1 >= image.cols ? (image.cols - 1) : x_1;
		y_1 = y_1 >= image.rows ? (image.rows - 1) : y_1;

		int x_2 = (d2.x + landmark_pos.x) * (double)bbox.width / 2.0 + (double)bbox.x + (double)bbox.width / 2.0;
		int y_2 = (d2.y + landmark_pos.y) * (double)bbox.height / 2.0 + (double)bbox.y + (double)bbox.height / 2.0;
		x_2 = x_2 < 0 ? 0 : x_2;
		y_2 = y_2 < 0 ? 0 : y_2;
		x_2 = x_2 >= image.cols ? (image.cols - 1) : x_2;
		y_2 = y_2 >= image.rows ? (image.rows - 1) : y_2;

		int sub_val = (int)image(y_1, x_1) - (int)image(y_2, x_2);
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
