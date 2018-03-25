#include "Dataset.h"

using namespace std;
using namespace cv;

Dataset::Dataset() {
}

Dataset::~Dataset() {}

bool Dataset::readFromFile(string configFile) {
	cout << "Reading dataset..." << endl;
	time_t start_time, end_time;
	time(&start_time);
	// Resolve config
	string ImgFolderPath, ImgNameFile;
	fstream config(configFile, ios::in);
	if (!config) {
		cerr << "Error: Config not exist." << endl;
		return false;
	}
	string line;
	while (getline(config, line)) {
		if (line.find("ImgFolderPath") != string::npos) {
			ImgFolderPath = line.substr(line.find_first_of("\"") + 1, line.find_last_of("\"") - line.find_first_of("\"") - 1);
		}
		if (line.find("ImgNameFile") != string::npos) {
			ImgNameFile = line.substr(line.find_first_of("\"") + 1, line.find_last_of("\"") - line.find_first_of("\"") - 1);
		}
	}
	config.close();

	// Read filenames from ImgNameFile
	vector<string> imgNames;
	fstream filenames(ImgNameFile, ios::in);
	if (!filenames) {
		cerr << "Error: Config error, ImgNameFile not exist." << endl;
		return false;
	}
	while (!filenames.eof()) {
		filenames >> line;
		if (line.size() <= 1) {
			continue;
		}
		if (filenames.eof()) {
			break;
		}
		imgNames.push_back(line);
	}
	filenames.close();

	// Read images
	for (int i = 0; i < imgNames.size(); ++i) {
		SingleData singleData;
		if (!singleData.readFromFile(ImgFolderPath + imgNames[i])) {
			cerr << "Error: Image " << imgNames[i] << " read failed." << endl;
			continue;
		}
		data.push_back(singleData);
	}

	time(&end_time);
	cout << "Reading finished. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl;
	return true;
}

int Dataset::size() {
	return data.size();
}

void Dataset::shapeIndexFeature(vector<pair<Point2D, Point2D>> &features, vector<vector<int>> &vals, int id) {
	vals.clear();
	for (int i = 0; i < data.size(); ++i) {
		vector<int> v;
		data[i].shapeIndexFeature(features, v, id);
		vals.push_back(v);
	}
}

double Dataset::calculateVariance(std::vector<int> &ids, int feature_point_id) {
	double Ex_2 = 0, Ey_2 = 0, Ex = 0, Ey = 0;
	for (int i = 0; i < ids.size(); ++i) {
		int id = ids[i];
		double x = data[id].getFeaturePoint(feature_point_id).x;
		double y = data[id].getFeaturePoint(feature_point_id).y;
		Ex += x;
		Ey += y;
		Ex_2 += x * x;
		Ey_2 += y * y;
	}
	Ex /= (double)ids.size();
	Ey /= (double)ids.size();
	Ex_2 /= (double)ids.size();
	Ey_2 /= (double)ids.size();
	return Ex_2 - Ex * Ex + Ey_2 - Ey * Ey;
}

SingleData::SingleData() {
}

SingleData::~SingleData() {}

bool SingleData::readFromFile(string filename) {
	string imgFilename = filename + ".jpg";
	string ptsFilename = filename + ".pts";

	// Read image
	imagePath = imgFilename;

	// Read pts
	fstream ptsFile(ptsFilename, ios::in);
	if (!ptsFile) {
		return false;
	}
	string version_str, n_points_str, bracketCh;
	int version, n_points;
	ptsFile >> version_str >> version >> n_points_str >> n_points >> bracketCh;
	for (int i = 0; i < n_points; ++i) {
		double x, y;
		ptsFile >> x >> y;
		groundTruth.push_back(Point2D(x, y));
	}
	ptsFile.close();
	return true;
}

void SingleData::shapeIndexFeature(vector<pair<Point2D, Point2D>> &features, vector<int> &vals, int id) {
	vals.clear();
	cv::Mat_<uchar> image = imread(imagePath, 0);
	for (int i = 0; i < features.size(); ++i) {
		double x1 = features[i].first.x;
		double y1 = features[i].first.y;
		double x2 = features[i].second.x;
		double y2 = features[i].second.y;
		int x_1 = x1 * (double)image.cols + groundTruth[id].x;
		int y_1 = y1 * (double)image.rows + groundTruth[id].y;
		x_1 = x_1 < 0 ? 0 : x_1;
		y_1 = y_1 < 0 ? 0 : y_1;
		x_1 = x_1 >= image.cols ? image.cols - 1 : x_1;
		y_1 = y_1 >= image.rows ? image.rows - 1 : y_1;
		int x_2 = x2 * (double)image.cols + groundTruth[id].x;
		int y_2 = y2 * (double)image.rows + groundTruth[id].y;
		x_2 = x_2 < 0 ? 0 : x_2;
		y_2 = y_2 < 0 ? 0 : y_2;
		x_2 = x_2 >= image.cols ? image.cols - 1 : x_2;
		y_2 = y_2 >= image.rows ? image.rows - 1 : y_2;
		vals.push_back((int)image(y_1, x_1) - (int)image(y_2, x_2));
	}
}

Point2D SingleData::getFeaturePoint(int id) {
	return Point2D(groundTruth[id]);
}
