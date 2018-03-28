#include "Regressor.h"

using namespace std;
using namespace cv;

Regressor::Regressor(string config_file) {
	configFile = config_file;
}

Regressor::~Regressor() { }

void Regressor::train(Dataset &dataset, vector<vector<RandomForest>> &forests) {
	vector<Point2D> S;
	dataset.generate_S_0(S, configFile);
	showImgWithLandmarks(dataset.data[0].imagePath, S);
}
