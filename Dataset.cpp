#include "Dataset.h"

using namespace std;
using namespace cv;

Dataset::Dataset() {
}

Dataset::~Dataset() {}

bool Dataset::readFromFile(string configFile, bool BBOX_FLAG) {
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
	CascadeClassifier haar_cascade;
	haar_cascade.load("model/haarcascade_frontalface_alt2.xml");
	for (int i = 0; i < imgNames.size(); ++i) {
		SingleData singleData;
		if (!singleData.readFromFile(ImgFolderPath + imgNames[i], haar_cascade, BBOX_FLAG)) {
			cerr << "Error: Image " << imgNames[i] << " read failed." << endl;
			continue;
		}
		if (singleData.isValid) {
			data.push_back(singleData);
		}
	}

	// Generate S0
	generate_S_0();

	time(&end_time);
	cout << data.size() << " images read. Time Ellapse: " << difftime(end_time, start_time) << "s" << endl << endl;
	return true;
}

void Dataset::shapeIndexFeature(std::vector<std::vector<Point2D>> &S, std::vector<Point2D> &S_0, vector<int> &special_point_id, vector<pair<Point2D, Point2D>> &features, vector<vector<vector<int>>> &img_vals) {
	img_vals.clear();
	for (int i = 0; i < data.size(); ++i) {
		vector<vector<int>> v;
		data[i].shapeIndexFeature(S[i], S_0, special_point_id, features, v);
		img_vals.push_back(v);
	}
}

double Dataset::calculateVariance(std::vector<int> &ids, int landmark_id) {
	double Ex_2 = 0, Ey_2 = 0, Ex = 0, Ey = 0;
	for (int i = 0; i < ids.size(); ++i) {
		int id = ids[i];
		double x = data[id].groundTruth[landmark_id].x;
		double y = data[id].groundTruth[landmark_id].y;
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

void Dataset::generate_S_0() {
	S_0.clear();

	int landmark_num = data[0].groundTruth.size();

	// Get average landmark pos
	for (int i = 0; i < landmark_num; ++i) {
		double x = 0.0, y = 0.0;
		for (int j = 0; j < data.size(); ++j) {
			x += data[j].groundTruth[i].x;
			y += data[j].groundTruth[i].y;
		}
		x /= (double)data.size();
		y /= (double)data.size();
		S_0.push_back(Point2D(x, y));
	}
}

SingleData::SingleData() {}

SingleData::~SingleData() {}

bool SingleData::readFromFile(string filename, CascadeClassifier &haar_cascade, bool BBOX_FLAG) {
	string imgFilename = filename + ".jpg";
	string ptsFilename = filename + ".pts";

	// Read image
	imagePath = imgFilename;
	cv::Mat_<uchar> image = imread(imagePath, 0);
	img_w = image.cols;
	img_h = image.rows;

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
		landmarks.push_back(Point2D(x, y));
	}
	ptsFile.close();

	// Get boundingbox
	if (BBOX_FLAG) {
		vector<Rect> faces;
		haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, Size(30, 30));
		
		if (!faces.size()) {
			isValid = false;
		}
		else {
			isValid = false;
			for (int i = 0; i < faces.size(); ++i) {
				Rect faceRec = faces[i];
				if (isShapeInRect(landmarks, faceRec)) {
					isValid = true;
					bbox = faceRec;
					break;
				}
			}
		}

		stringstream bbox_filename;
		bbox_filename << "model/boundingbox/" << filename.substr(filename.find_last_of("\\") + 1) << ".bbox";
		fstream fout(bbox_filename.str(), ios::out);
		if (isValid) {
			fout << 1 << endl << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << endl;
		}
		else {
			fout << 0 << endl;
		}
		fout.close();
	}
	else {
		stringstream bbox_filename;
		bbox_filename << "model/boundingbox/" << filename.substr(filename.find_last_of("\\") + 1) << ".bbox";
		fstream fin(bbox_filename.str(), ios::in);
		if (!fin) {
			cerr << "Boundingbox " << bbox_filename.str() << " not exist!" << endl;
			return false;
		}
		int isV = 0;
		fin >> isV;
		if (isV == 1) {
			isValid = true;
			int x = 0, y = 0, w = 0, h = 0;
			fin >> x >> y >> w >> h;
			bbox.x = x;
			bbox.y = y;
			bbox.width = w;
			bbox.height = h;
		}
		else {
			isValid = false;
			fin.close();
			return true;
		}
		fin.close();
	}

	// Get ground truth
	for (int i = 0; i < n_points; ++i) {
		double x = (landmarks[i].x - (double)bbox.x - (double)bbox.width / 2.0) / ((double)bbox.width / 2.0);
		double y = (landmarks[i].y - (double)bbox.y - (double)bbox.height / 2.0) / ((double)bbox.height / 2.0);
		groundTruth.push_back(Point2D(x, y));
	}
	
	return true;
}

void SingleData::shapeIndexFeature(std::vector<Point2D> &S, std::vector<Point2D> &S_0, vector<int> &special_point_id, vector<pair<Point2D, Point2D>> &features, vector<vector<int>> &vals) {
	vals.clear();
	cv::Mat_<uchar> image = imread(imagePath, 0);
	Transform_2D tran = getSimilarityTransform(S, S_0, special_point_id);

	for (int i = 0; i < groundTruth.size(); ++i) {
		vector<int> v;
		for (int j = 0; j < features.size(); ++j) {
			double x1 = features[j].first.x, y1 = features[j].first.y;
			double x2 = features[j].second.x, y2 = features[j].second.y;

			Point2D d1 = transformPoint(tran, Point2D(x1, y1));
			Point2D d2 = transformPoint(tran, Point2D(x2, y2));

			int x_1 = (d1.x + S[i].x) * (double)bbox.width / 2.0 + (double)bbox.x + (double)bbox.width / 2.0;
			int y_1 = (d1.y + S[i].y) * (double)bbox.height / 2.0 + (double)bbox.y + (double)bbox.height / 2.0;
			x_1 = x_1 < 0 ? 0 : x_1;
			y_1 = y_1 < 0 ? 0 : y_1;
			x_1 = x_1 >= image.cols ? (image.cols - 1) : x_1;
			y_1 = y_1 >= image.rows ? (image.rows - 1) : y_1;

			int x_2 = (d2.x + S[i].x) * (double)bbox.width / 2.0 + (double)bbox.x + (double)bbox.width / 2.0;
			int y_2 = (d2.y + S[i].y) * (double)bbox.height / 2.0 + (double)bbox.y + (double)bbox.height / 2.0;
			x_2 = x_2 < 0 ? 0 : x_2;
			y_2 = y_2 < 0 ? 0 : y_2;
			x_2 = x_2 >= image.cols ? (image.cols - 1) : x_2;
			y_2 = y_2 >= image.rows ? (image.rows - 1) : y_2;

			v.push_back((int)image(y_1, x_1) - (int)image(y_2, x_2));
		}
		vals.push_back(v);
	}
}

bool isShapeInRect(vector<Point2D> &shape, cv::Rect &rect) {
	double sum_x = 0.0, sum_y = 0.0;
	double max_x = -DBL_MAX, min_x = DBL_MAX, max_y = -DBL_MAX, min_y = DBL_MAX;
	for (int i = 0; i < shape.size(); i++) {
		if (shape[i].x > max_x)
			max_x = shape[i].x;
		if (shape[i].x < min_x)
			min_x = shape[i].x;
		if (shape[i].y > max_y)
			max_y = shape[i].y;
		if (shape[i].y < min_y)
			min_y = shape[i].y;

		sum_x += shape[i].x;
		sum_y += shape[i].y;
	}
	sum_x /= shape.size();
	sum_y /= shape.size();

	if ((max_x - min_x) > rect.width * 1.5)
		return false;
	if ((max_y - min_y) > rect.height * 1.5)
		return false;
	if (std::abs(sum_x - (rect.x + rect.width / 2.0)) > rect.width / 2.0)
		return false;
	if (std::abs(sum_y - (rect.y + rect.height / 2.0)) > rect.height / 2.0)
		return false;

	return true;
}
