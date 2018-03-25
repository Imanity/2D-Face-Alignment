#include "RandomForest.h"

using namespace std;
using namespace cv;

RandomForest::RandomForest() {
}

RandomForest::~RandomForest() {}

bool RandomForest::generateFromDataset(Dataset &dataset, string configFile, int stage_id, int feature_point_id) {
	cout << "Constructing Random Forest..." << endl;
	time_t start_time, end_time;
	time(&start_time);
	// Get config
	int feature_num, tree_depth, tree_num_per_forest;
	double local_region_size, forest_overlap;
	fstream config(configFile, ios::in);
	if (!config) {
		cerr << "Error: Config not exist." << endl;
		return false;
	}
	string line;
	while (getline(config, line)) {
		if (line.find("feature_num") != string::npos) {
			feature_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("tree_depth") != string::npos) {
			tree_depth = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("tree_num_per_forest") != string::npos) {
			tree_num_per_forest = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("local_region_size") != string::npos) {
			int pos1 = line.find("{ ") + 2, pos2 = line.find(", ");
			for (int i = 0; i < stage_id; ++i) {
				pos1 = pos2 + 2;
				pos2 = line.find(", ", pos1 + 1);
			}
			if (pos2 == string::npos) {
				pos2 = line.size();
			}
			local_region_size = atof(line.substr(pos1, pos2 - pos1).c_str());
		}
		if (line.find("forest_overlap") != string::npos) {
			forest_overlap = atof(line.substr(line.find("= ") + 2).c_str());
		}
	}
	// cout << feature_num << "\t" << tree_depth << "\t" << tree_num_per_forest << "\t" << local_region_size << "\t" << forest_overlap << endl;
	config.close();

	// Get random features
	time_t currTime = time(0);
	cv::RNG rd(currTime);
	vector<pair<Point2D, Point2D>> features;
	for (int i = 0; i < feature_num; ++i) {
		double x1, y1, x2, y2;
		do {
			x1 = rd.uniform(-local_region_size, local_region_size);
			y1 = rd.uniform(-local_region_size, local_region_size);
		} while (x1 * x1 + y1 * y1 > local_region_size * local_region_size);
		do {
			x2 = rd.uniform(-local_region_size, local_region_size);
			y2 = rd.uniform(-local_region_size, local_region_size);
		} while (x2 * x2 + y2 * y2 > local_region_size * local_region_size);
		Point2D p1(x1, y1), p2(x2, y2);
		features.push_back(make_pair(p1, p2));
	}
	vector<vector<int>> vals;
	dataset.shapeIndexFeature(features, vals, feature_point_id);

	// Build Decision Trees
	int step = floor(((double)dataset.size()) * forest_overlap / (tree_num_per_forest - 1));
	for (int i = 0; i < tree_num_per_forest; ++i) {
		vector<int> imgIds;
		int start_index = i * step;
		int end_index = dataset.size() - (tree_num_per_forest - i - 1) * step;
		for (int j = start_index; j < end_index; ++j) {
			imgIds.push_back(j);
		}
		DecisionTree t = generateDecisionTree(dataset, features, vals, imgIds, feature_point_id, tree_depth);
		trees.push_back(t);
	}

	time(&end_time);
	cout << "Constructing finished. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl;
	return true;
}

DecisionTree RandomForest::generateDecisionTree(Dataset &dataset, vector<pair<Point2D, Point2D>> &features, vector<vector<int>> &vals, vector<int> &imgIds, int feature_point_id, int depth) {
	if (depth < 1 || !imgIds.size()) {
		return NULL;
	}
	DecisionTree tree = new TreeNode(depth);
	vector<int> leftIds, rightIds;
	vector<int> tmpLeftIds, tmpRightIds;

	time_t current_time;
	current_time = time(0);
	cv::RNG rd(current_time);

	// Select feature to split
	double var = -DBL_MAX;
	for (int i = 0; i < features.size(); ++i) {
		tmpLeftIds.clear();
		tmpRightIds.clear();

		// Get min and max value of feature i
		int minVal = INT_MAX, maxVal = -INT_MAX;
		for (int j = 0; j < imgIds.size(); ++j) {
			if (vals[imgIds[j]][i] < minVal) {
				minVal = vals[imgIds[j]][i];
			}
			if (vals[imgIds[j]][i] > maxVal) {
				maxVal = vals[imgIds[j]][i];
			}
		}

		// Randomly get threshold between min and max
		int split_threshold = minVal + (int)((0.5 + 0.8 * (rd.uniform(0.0, 1.0) - 0.5)) * (double)(maxVal - minVal));
		for (int j = 0; j < imgIds.size(); ++j) {
			if (vals[imgIds[j]][i] < split_threshold) {
				tmpLeftIds.push_back(imgIds[j]);
			} else {
				tmpRightIds.push_back(imgIds[j]);
			}
		}

		// Calculate variance
		double var_l = tmpLeftIds.size() ? dataset.calculateVariance(tmpLeftIds, feature_point_id) : 0.0;
		double var_r = tmpRightIds.size() ? dataset.calculateVariance(tmpRightIds, feature_point_id) : 0.0;
		double var_tmp = -(double)tmpLeftIds.size() * var_l - (double)tmpRightIds.size() * var_r;

		// Select feature
		if (var_tmp > var) {
			var = var_tmp;
			leftIds = tmpLeftIds;
			rightIds = tmpRightIds;
			tree->feature = features[i];
			tree->threshold = split_threshold;
		}
	}
	tree->left_child = generateDecisionTree(dataset, features, vals, leftIds, feature_point_id, depth - 1);
	tree->right_child = generateDecisionTree(dataset, features, vals, rightIds, feature_point_id, depth - 1);
	return tree;
}

void RandomForest::outputToFile(std::string filename) {
	if (!trees.size()) {
		return;
	}
	fstream outFile(filename, ios::out);
	outFile << "Forest Depth: " << trees[0]->depth << endl;
	for (int i = 0; i < trees.size(); ++i) {
		vector<NodeData> traverse;

		// Traverse the tree hierarchically
		queue<DecisionTree> q;
		q.push(trees[i]);
		while (traverse.size() < (int)pow(2, trees[i]->depth) - 1) {
			DecisionTree t = q.front();
			q.pop();
			if (t != NULL) {
				traverse.push_back(NodeData(Point2D(t->feature.first.x, t->feature.first.y), Point2D(t->feature.second.x, t->feature.second.y), t->threshold));
				q.push(t->left_child);
				q.push(t->right_child);
			} else {
				traverse.push_back(NodeData(Point2D(-1, -1), Point2D(-1, -1), 0));
				q.push(NULL);
				q.push(NULL);
			}
		}

		// Output to file
		outFile << "Tree " << i << endl;
		for (int j = 0; j < traverse.size(); ++j) {
			outFile << traverse[j].p1.x << " " << traverse[j].p1.y << " " << traverse[j].p2.x << " " << traverse[j].p2.y << " " << traverse[j].threshold << endl;
		}
	}
	outFile.close();
}

TreeNode::TreeNode(int curr_depth) {
	depth = curr_depth;
}

TreeNode::~TreeNode() {}
