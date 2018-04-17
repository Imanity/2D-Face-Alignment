#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "RandomForest.h"
#include "Dataset.h"
#include "Utils.h"
#include "liblinear/linear.h"

class Regressor {
public:
	Regressor(Params_ params_);
	~Regressor();

	void Train(Dataset &dataset);
	void readModels();
	void predictImage(std::string img_path);
	void predictImage(std::string img_path, cv::Rect bbox);

private:
	Params_ params;

	std::vector<std::vector<struct model*>>	linear_model_x;
	std::vector<std::vector<struct model*>>	linear_model_y;
	std::vector<std::vector<RandomForest>> forests;

private:
	int classifyImgByTree(DecisionTree tree, std::vector<int> &img_vals);
	int classifyImgByTree(DecisionTree tree, cv::Mat_<uchar> &image, cv::Rect &bbox, Point2D &landmark_pos, Transform_2D &t);
};
