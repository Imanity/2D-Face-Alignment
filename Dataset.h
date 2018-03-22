#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

class Dataset {
public:
	Dataset();
	~Dataset();

	bool readFromFile(string configFile);

private:
	vector<SingleData> data;
};

class SingleData {
public:
	SingleData();
	~SingleData();

private:
	cv::Mat_<uchar> image;
	vector<double
};
