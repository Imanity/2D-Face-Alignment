#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "RandomForest.h"
#include "Dataset.h"
#include "Utils.h"
#include "liblinear/linear.h"

class Regressor {
public:
	Regressor(std::string config_file);
	~Regressor();

	void train(Dataset &dataset, std::vector<std::vector<RandomForest>> &forests);

private:
	std::string configFile;

	std::vector<std::vector<struct model*>>	linear_model_x;
	std::vector<std::vector<struct model*>>	linear_model_y;

private:
	int classifyImgByTree(DecisionTree tree, cv::Mat_<uchar> &image, Point2D now_landmark_pos);
};
