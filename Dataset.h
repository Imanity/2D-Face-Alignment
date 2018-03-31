#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>

#include "Utils.h"

class SingleData {
public:
	SingleData();
	~SingleData();

	bool readFromFile(std::string filename, cv::CascadeClassifier &haar_cascade, bool BBOX_FLAG);
	void shapeIndexFeature(std::vector<Point2D> &S, std::vector<Point2D> &S_0, std::vector<int> &special_point_id, std::vector<std::pair<Point2D, Point2D>> &features, std::vector<std::vector<int>> &img_val);

public:
	std::string imagePath;
	std::vector<Point2D> groundTruth;
	std::vector<Point2D> landmarks;
	int img_w, img_h;
	cv::Rect bbox;
	bool isValid;
};

class Dataset {
public:
	Dataset();
	~Dataset();

	bool readFromFile(std::string configFile, bool BBOX_FLAG);
	void shapeIndexFeature(std::vector<std::vector<Point2D>> &S, std::vector<Point2D> &S_0, std::vector<int> &special_point_id, std::vector<std::pair<Point2D, Point2D>> &features, std::vector<std::vector<std::vector<int>>> &img_vals);
	double calculateVariance(std::vector<int> &ids, int landmark_id);
	void generate_S_0();

public:
	std::vector<SingleData> data;
	std::vector<Point2D> S_0;
};

bool isShapeInRect(std::vector<Point2D> &shape, cv::Rect &rect);
