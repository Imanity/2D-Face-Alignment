#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Dataset.h"

extern void showImgWithLandmarks(std::string imagePath, std::vector<Point2D> &landmarks);
void setColor(cv::Mat &src, int x, int y, uchar r, uchar g, uchar b);
