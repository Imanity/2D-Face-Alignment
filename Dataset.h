#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>

struct Point2D {
	double x, y;
	Point2D(double x_, double y_) : x(x_), y(y_) {}
	Point2D() {
		x = y = 0.0;
	}
};

class SingleData {
public:
	SingleData();
	~SingleData();

	bool readFromFile(std::string filename);
	void shapeIndexFeature(std::vector<std::pair<Point2D, Point2D>> &features, std::vector<int> &vals, int id);
	Point2D getFeaturePoint(int id);

private:
	std::string imagePath;
	std::vector<Point2D> groundTruth;
};

class Dataset {
public:
	Dataset();
	~Dataset();

	bool readFromFile(std::string configFile);
	int size();
	void shapeIndexFeature(std::vector<std::pair<Point2D, Point2D>> &features, std::vector<std::vector<int>> &vals, int id);
	double calculateVariance(std::vector<int> &ids, int feature_point_id);

private:
	std::vector<SingleData> data;
};
