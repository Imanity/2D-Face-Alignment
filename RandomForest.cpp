#include "RandomForest.h"

using namespace std;
using namespace cv;

RandomForest::RandomForest() {
}

RandomForest::~RandomForest() {}

bool RandomForest::generateFromDataset(Dataset &dataset, Params_ &params, vector<pair<Point2D, Point2D>> &features, vector<vector<vector<int>>> &img_vals, int landmark_id) {
	// Build Decision Trees
	int step = floor(((double)dataset.data.size()) * params.forest_overlap / (params.tree_num_per_forest - 1));
	for (int i = 0; i < params.tree_num_per_forest; ++i) {
		vector<int> imgIds;
		int start_index = i * step;
		int end_index = dataset.data.size() - (params.tree_num_per_forest - i - 1) * step;
		for (int j = start_index; j < end_index; ++j) {
			imgIds.push_back(j);
		}
		DecisionTree t = generateDecisionTree(dataset, params, features, imgIds, img_vals, landmark_id, params.tree_depth);
		trees.push_back(t);
	}

	return true;
}

bool RandomForest::generateFromFile(std::string filename) {
	fstream inFile(filename, ios::in);
	if (!inFile) {
		cerr << "Error: Forest file " << filename << " not exist." << endl;
		return false;
	}

	int depth;
	string line;

	// Get forest depth
	getline(inFile, line);
	depth = atoi(line.substr(line.find(": ") + 2).c_str());

	while (getline(inFile, line)) {
		// Get tree id
		int treeId = atoi(line.substr(line.find(" ") + 1).c_str());
		vector<vector<double>> node_data;
		for (int i = 0; i < (int)pow(2, depth) - 1; ++i) {
			// Get node data
			getline(inFile, line);
			vector<double> single_node;
			for (int j = 0; j < 4; ++j) {
				single_node.push_back(atof(line.substr(0, line.find(" ")).c_str()));
				line = line.substr(line.find(" ") + 1);
			}
			single_node.push_back(atof(line.c_str()));
			node_data.push_back(single_node);
		}
		trees.push_back(generateDecisionTreeFromVector(node_data, depth, 0));
	}
	inFile.close();

	return true;
}

DecisionTree RandomForest::generateDecisionTree(Dataset &dataset, Params_ &params, vector<pair<Point2D, Point2D>> &features, vector<int> &imgIds, vector<vector<vector<int>>> &img_vals, int landmark_id, int depth) {
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
			if (img_vals[imgIds[j]][landmark_id][i] < minVal) {
				minVal = img_vals[imgIds[j]][landmark_id][i];
			}
			if (img_vals[imgIds[j]][landmark_id][i] > maxVal) {
				maxVal = img_vals[imgIds[j]][landmark_id][i];
			}
		}

		// Randomly get threshold between min and max
		int split_threshold = minVal + (int)((0.5 + 0.8 * (rd.uniform(0.0, 1.0) - 0.5)) * (double)(maxVal - minVal));
		for (int j = 0; j < imgIds.size(); ++j) {
			if (img_vals[imgIds[j]][landmark_id][i] < split_threshold) {
				tmpLeftIds.push_back(imgIds[j]);
			} else {
				tmpRightIds.push_back(imgIds[j]);
			}
		}

		// Calculate variance
		double var_l = tmpLeftIds.size() ? dataset.calculateVariance(tmpLeftIds, landmark_id) : 0.0;
		double var_r = tmpRightIds.size() ? dataset.calculateVariance(tmpRightIds, landmark_id) : 0.0;
		double var_tmp = -(double)tmpLeftIds.size() * var_l - (double)tmpRightIds.size() * var_r;

		// Select feature
		if (var_tmp > var) {
			var = var_tmp;
			leftIds = tmpLeftIds;
			rightIds = tmpRightIds;
			tree->feature = features[i];
			tree->feature_id = i;
			tree->threshold = split_threshold;
		}
	}
	tree->left_child = generateDecisionTree(dataset, params, features, leftIds, img_vals, landmark_id, depth - 1);
	tree->right_child = generateDecisionTree(dataset, params, features, rightIds, img_vals, landmark_id, depth - 1);
	return tree;
}

DecisionTree RandomForest::generateDecisionTreeFromVector(std::vector<std::vector<double>> &node_data, int depth, int root_pos) {
	if (depth <= 0) {
		return NULL;
	}
	DecisionTree tree = new TreeNode(depth);
	tree->feature = make_pair(Point2D(node_data[root_pos][0], node_data[root_pos][1]), Point2D(node_data[root_pos][2], node_data[root_pos][3]));
	tree->threshold = node_data[root_pos][4];
	tree->left_child = generateDecisionTreeFromVector(node_data, depth - 1, 2 * root_pos + 1);
	tree->right_child = generateDecisionTreeFromVector(node_data, depth - 1, 2 * root_pos + 2);
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
