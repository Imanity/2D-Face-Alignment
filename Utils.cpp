#include "Utils.h"

using namespace std;
using namespace cv;

void showImgWithLandmarks(string imagePath, vector<Point2D> &landmarks) {
	Mat imageSrc = imread(imagePath), image;
	resize(imageSrc, image, Size(500, 500 * imageSrc.rows / imageSrc.cols));
	double c = image.cols, r = image.rows;
	for (int i = 0; i < landmarks.size(); ++i) {
		double x, y;
		x = c * landmarks[i].x;
		y = r * landmarks[i].y;
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

