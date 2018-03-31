#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Point2D {
	double x, y;
	Point2D(double x_, double y_) : x(x_), y(y_) {}
	Point2D() {
		x = y = 0.0;
	}
};

struct Transform_2D {
	Point2D x_to, y_to;
	Point2D x_from, y_from;
};

extern void showImgWithLandmarks(std::string imagePath, std::vector<Point2D> &landmarks, std::vector<cv::Rect> &bbox, int max_area_bbox_id);
void setColor(cv::Mat &src, int x, int y, uchar r, uchar g, uchar b);

extern Transform_2D getSimilarityTransform(std::vector<Point2D> &shape_to, std::vector<Point2D> &shape_from, std::vector<int> &special_point_id);
extern Point2D transformPoint(Transform_2D &t, Point2D p);

class Params_ {
public:
	int feature_num;
	int landmark_num;
	int stage_num;
	int tree_depth;
	int tree_num_per_forest;
	double forest_overlap;
	std::vector<double> local_region_size;
	std::vector<int> special_point_id;

public:
	Params_();
	Params_(std::string configFile);
	~Params_();
	
	void output();
};
