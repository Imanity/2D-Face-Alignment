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

	bool generateFromDataset(Dataset &dataset, std::string configFile, int stage_id, int feature_point_id);
	bool generateFromFile(std::string filename);

	DecisionTree generateDecisionTree(Dataset &dataset, std::vector<std::pair<Point2D, Point2D>> &features, std::vector<std::vector<int>> &vals, std::vector<int> &imgIds, int feature_point_id, int depth);
	DecisionTree generateDecisionTreeFromVector(std::vector<std::vector<double>> &node_data, int depth, int root_pos);

	void outputToFile(std::string filename);

public:
	std::vector<DecisionTree> trees;
};
