#pragma once

#include "Dataset.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

struct NodeData {
	Point2D p1, p2;
	int threshold;
	NodeData(Point2D p_1, Point2D p_2, int t_) : p1(p_1), p2(p_2), threshold(t_) {}
};

class TreeNode {
public:
	TreeNode *left_child;
	TreeNode *right_child;
	int threshold;
	int feature_id;
	std::pair<Point2D, Point2D> feature;
	int depth;
	
public:
	TreeNode(int curr_depth);
	~TreeNode();
};

typedef TreeNode* DecisionTree;

class RandomForest {
public:
	RandomForest();
	~RandomForest();

	bool generateFromDataset(Dataset &dataset, Params_ &params, std::vector<std::pair<Point2D, Point2D>> &features, std::vector<std::vector<std::vector<int>>> &img_vals, int landmark_id);
	bool generateFromFile(std::string filename);

	DecisionTree generateDecisionTree(Dataset &dataset, Params_ &params, std::vector<std::pair<Point2D, Point2D>> &features, std::vector<int> &imgIds, std::vector<std::vector<std::vector<int>>> &img_vals, int landmark_id, int depth);
	DecisionTree generateDecisionTreeFromVector(std::vector<std::vector<double>> &node_data, int depth, int root_pos);

	void outputToFile(std::string filename);

public:
	std::vector<DecisionTree> trees;
};
