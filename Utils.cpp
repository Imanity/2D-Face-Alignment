#include "Utils.h"

using namespace std;
using namespace cv;

Params_::Params_() {}

Params_::Params_(std::string configFile) {
	// Get config
	fstream config(configFile, ios::in);
	if (!config) {
		cerr << "Error: Config not exist." << endl;
		return;
	}
	string line;
	while (getline(config, line)) {
		if (line.find("feature_num") != string::npos) {
			feature_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("landmark_num") != string::npos) {
			landmark_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("stage_num") != string::npos) {
			stage_num = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("tree_depth") != string::npos) {
			tree_depth = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("tree_num_per_forest") != string::npos) {
			tree_num_per_forest = atoi(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("forest_overlap") != string::npos) {
			forest_overlap = atof(line.substr(line.find("= ") + 2).c_str());
		}
		if (line.find("local_region_size") != string::npos) {
			int pos1 = line.find("{ ") + 2, pos2 = line.find(", ");
			while (pos2 != string::npos) {
				local_region_size.push_back(atof(line.substr(pos1, pos2 - pos1).c_str()));
				pos1 = pos2 + 2;
				pos2 = line.find(", ", pos1 + 1);
			}
			local_region_size.push_back(atof(line.substr(pos1).c_str()));
		}
		if (line.find("special_point_id") != string::npos) {
			int pos1 = line.find("{ ") + 2, pos2 = line.find(", ");
			for (int i = 0; i < 3; ++i) {
				special_point_id.push_back(atoi(line.substr(pos1, pos2 - pos1).c_str()));
				pos1 = pos2 + 2;
				pos2 = line.find(", ", pos1 + 1);
			}
			special_point_id.push_back(atoi(line.substr(pos1).c_str()));
		}
	}
	config.close();
}

Params_::~Params_() {}

void Params_::output() {
	cout << "========================= Params =========================" << endl;
	cout << "feature_num: " << feature_num << "\tlandmark_num: " << landmark_num << endl;
	cout << "stage_num: " << stage_num << "\ttree_depth: " << tree_depth << endl;
	cout << "tree_num_per_forest: " << tree_num_per_forest << "\tforest_overlap: " << forest_overlap << endl;
	cout << "local_region_size: " << endl;
	for (int i = 0; i < local_region_size.size(); ++i) {
		cout << local_region_size[i] << " ";
	}
	cout << endl << "special_point_id: " << endl;
	for (int i = 0; i < special_point_id.size(); ++i) {
		cout << special_point_id[i] << " ";
	}
	cout << endl << "==========================================================" << endl << endl;
}

void showImgWithLandmarks(string imagePath, vector<Point2D> &landmarks, vector<cv::Rect> &bbox, int max_area_bbox_id) {
	Mat imageSrc = imread(imagePath), image;
	resize(imageSrc, image, Size(500, 500 * imageSrc.rows / imageSrc.cols));
	double r = 500.0 / (double)imageSrc.cols;

	// Draw boundingboxes
	for (int i = 0; i < bbox.size(); ++i) {
		for (int x = bbox[i].x; x <= bbox[i].x + bbox[i].width; ++x) {
			if (i == max_area_bbox_id) {
				setColor(image, (double)bbox[i].y * r, (double)x * r, 0, 0, 255);
				setColor(image, (double)(bbox[i].y + bbox[i].height) * r, (double)x * r, 0, 0, 255);
			}
		}
		for (int y = bbox[i].y; y <= bbox[i].y + bbox[i].height; ++y) {
			if (i == max_area_bbox_id) {
				setColor(image, (double)y * r, (double)bbox[i].x * r, 0, 0, 255);
				setColor(image, (double)y * r, (double)(bbox[i].x + bbox[i].width) * r, 0, 0, 255);
			}
		}
	}

	// Draw landmarks
	for (int i = 0; i < landmarks.size(); ++i) {
		double x, y;
		x = landmarks[i].x * (double)bbox[max_area_bbox_id].width / 2.0 + (double)bbox[max_area_bbox_id].x + (double)bbox[max_area_bbox_id].width / 2.0;
		y = landmarks[i].y * (double)bbox[max_area_bbox_id].height / 2.0 + (double)bbox[max_area_bbox_id].y + (double)bbox[max_area_bbox_id].height / 2.0;
		x = x * r;
		y = y * r;
		setColor(image, y, x, 255, 0, 0);
		setColor(image, y - 1, x, 255, 0, 0);
		setColor(image, y + 1, x, 255, 0, 0);
		setColor(image, y, x - 1, 255, 0, 0);
		setColor(image, y, x + 1, 255, 0, 0);
	}
	imshow("Result", image);
	waitKey(10000);
}

void setColor(Mat &src, int x, int y, uchar r, uchar g, uchar b) {
	src.at<cv::Vec3b>(x, y)[0] = b;
	src.at<cv::Vec3b>(x, y)[1] = g;
	src.at<cv::Vec3b>(x, y)[2] = r;
}

Transform_2D getSimilarityTransform(vector<Point2D> &shape_to, vector<Point2D> &shape_from, vector<int> &special_point_id) {
	Transform_2D t;

	t.x_to = Point2D(shape_to[special_point_id[3]].x - shape_to[special_point_id[2]].x, shape_to[special_point_id[3]].y - shape_to[special_point_id[2]].y);
	t.y_to = Point2D(shape_to[special_point_id[1]].x - shape_to[special_point_id[0]].x, shape_to[special_point_id[1]].y - shape_to[special_point_id[0]].y);

	t.x_from = Point2D(shape_from[special_point_id[3]].x - shape_from[special_point_id[2]].x, shape_from[special_point_id[3]].y - shape_from[special_point_id[2]].y);
	t.y_from = Point2D(shape_from[special_point_id[1]].x - shape_from[special_point_id[0]].x, shape_from[special_point_id[1]].y - shape_from[special_point_id[0]].y);

	return t;
}

Point2D transformPoint(Transform_2D &t, Point2D p) {
	double x = p.x, y = p.y;
	double a = (x * t.y_from.y - y * t.y_from.x) / (t.x_from.x * t.y_from.y - t.x_from.y * t.y_from.x);
	double b = (y * t.x_from.x - x * t.x_from.y) / (t.x_from.x * t.y_from.y - t.x_from.y * t.y_from.x);
	double x_ = a * t.x_to.x + b * t.y_to.x;
	double y_ = a * t.x_to.y + b * t.y_to.y;
	return Point2D(x_, y_);
}
