#include <iostream>

#include "Dataset.h"
#include "RandomForest.h"

using namespace std;

void trainForest(Dataset &dataset);
void readForest(vector<vector<RandomForest>> &forests);

int main() {
	Dataset helen;
	helen.readFromFile("config/helen.cfg");

	vector<vector<RandomForest>> forests;
	readForest(forests);

	return 0;
}

void trainForest(Dataset &dataset) {
	// Read config
	string configFile = "config/train.cfg";
	fstream config(configFile, ios::in);
	if (!config) {
		cerr << "Error: Config not exist." << endl;
		return;
	}
	int feature_point_num = 0, stage_num = 0;
	string line;
	while (getline(config, line)) {
		if (line.find("feature_point_num") != string::npos) {
			feature_point_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("stage_num") != string::npos) {
			stage_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
	}
	config.close();

	// Train
	for (int feature = 0; feature < feature_point_num; ++feature) {
		cout << "============================== Feature " << feature << "==============================" << endl;
		for (int stage = 0; stage < stage_num; ++stage) {
			cout << "------------------ Stage" << stage << "------------------" << endl;
			stringstream sstream;
			sstream << "model/random_forest/feature_" << feature << "_stage_" << stage << ".mdl";
			RandomForest forest;
			forest.generateFromDataset(dataset, configFile, stage, feature);
			forest.outputToFile(sstream.str());
		}
		cout << endl;
	}
}

void readForest(vector<vector<RandomForest>> &forests) {
	// Read config
	string configFile = "config/train.cfg";
	fstream config(configFile, ios::in);
	if (!config) {
		cerr << "Error: Config not exist." << endl;
		return;
	}
	int feature_point_num = 0, stage_num = 0;
	string line;
	while (getline(config, line)) {
		if (line.find("feature_point_num") != string::npos) {
			feature_point_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("stage_num") != string::npos) {
			stage_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
	}
	config.close();
	
	cout << "Reading Model..." << endl;
	time_t start_time, end_time;
	time(&start_time);

	// Read Forest
	for (int feature = 0; feature < feature_point_num; ++feature) {
		vector<RandomForest> forestPerFeature;
		for (int stage = 0; stage < stage_num; ++stage) {
			RandomForest forest;
			stringstream sstream;
			sstream << "model/random_forest/feature_" << feature << "_stage_" << stage << ".mdl";
			forest.generateFromFile(sstream.str());
			forestPerFeature.push_back(forest);
		}
		forests.push_back(forestPerFeature);
	}

	time(&end_time);
	cout << "Reading finished. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl;
}
