#pragma once
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
//#include <Eigen/Core>
//#include <GenEigsSolver.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
//using namespace Spectra;

struct Center {
	double blueValue;
	double greenValue;
	double redValue;
};

struct Feature {
	int x;             //x轴坐标
	int y;             //y轴坐标
	double x_diff;     //x轴一阶导数 |rgb(x+1,y)-rgb(x,y)|
	double y_diff;     //y轴一阶导数 |rgb(x,y+1)-rgb(x,y)|
};

class KmeansAlgorithm {

public:
	KmeansAlgorithm();
	~KmeansAlgorithm();

	static void createClustersInfo(cv::Mat imgInput, int clusters_number, std::vector<cv::Scalar> & clustersCenters, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> & ptInClusters);
	static double computeColorDistance(cv::Scalar pixel, cv::Scalar clusterPixel);
	static double* computeXYDistance(Feature pixelF, Feature clusterFeature);
	static void findAssociatedCluster(cv::Mat imgInput, int clusters_number, std::vector<cv::Scalar> clustersCenters, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> & ptInClusters, double * weight);
	static double adjustClusterCenters(cv::Mat imgInput, int clusters_number, std::vector<cv::Scalar> & clustersCenters, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> ptInClusters, double & oldCenter, double newCenter, double * weight);
	static cv::Mat applyFinalClusterToImage(cv::Mat & imgOutput,cv::Mat & imgLabel, int clusters_number, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> ptInClusters);
	static double calError(cv::Mat & imgOutputKNN, cv::Mat & imgLable);
};