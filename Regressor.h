#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "RandomForest.h"
#include "Dataset.h"
#include "Utils.h"

class Regressor {
public:
	Regressor(std::string config_file);
	~Regressor();

	void train(Dataset &dataset, std::vector<std::vector<RandomForest>> &forests);

private:
	std::string configFile;
};
