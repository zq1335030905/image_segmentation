#include "seg.h"

KmeansAlgorithm::KmeansAlgorithm() {
}

KmeansAlgorithm::~KmeansAlgorithm() {
}


void KmeansAlgorithm::createClustersInfo(cv::Mat imgInput, int clusters_number, std::vector<cv::Scalar> & clustersCenters, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> & ptInClusters) {
	//随机选择聚类中心以及得到聚类中心的特征向量{颜色（rgb），距离（x，y），纹理（一阶导）}
	cv::RNG random(cv::getTickCount());

	for (int k = 0; k<clusters_number; k++) {

		//得到随机的中心点
		cv::Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols - 1);
		centerKPoint.y = random.uniform(0, imgInput.rows - 1);
		cv::Scalar centerPixel = imgInput.at<cv::Vec3b>(centerKPoint.y, centerKPoint.x);

		//得到中心点的特征矩阵
		cv::Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		double x_diff = computeColorDistance(centerK, cv::Scalar(imgInput.at<cv::Vec3b>(centerKPoint.y, centerKPoint.x + 1)));
		double y_diff = computeColorDistance(centerK, cv::Scalar(imgInput.at<cv::Vec3b>(centerKPoint.y + 1, centerKPoint.x)));
		Feature centerF;
		centerF.x = centerKPoint.x;
		centerF.y = centerKPoint.y;
		centerF.x_diff = x_diff;
		centerF.y_diff = y_diff;
		clustersFeatures.push_back(centerF);

		std::vector<cv::Point> ptInClusterK;
		ptInClusters.push_back(ptInClusterK);
	}
}

double KmeansAlgorithm::computeColorDistance(cv::Scalar pixel, cv::Scalar clusterPixel) {

	//计算颜色特征之间的距离
	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	//欧式距离
	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));

	return distance;
}


double* KmeansAlgorithm::computeXYDistance(Feature pixelF, Feature clusterFeature) {
	//计算位置距离以及纹理距离

	double dis[2] = { 0,0 };
	double xyDis = sqrt(pow((pixelF.x - clusterFeature.x), 2) + pow((pixelF.y - clusterFeature.y), 2));
	double diffDis = sqrt(pow((pixelF.x_diff - clusterFeature.x_diff), 2) + pow((pixelF.y_diff - clusterFeature.y_diff), 2));
	dis[0] = xyDis;
	dis[1] = diffDis;
	return dis;
}

void KmeansAlgorithm::findAssociatedCluster(cv::Mat imgInput, int clusters_number, std::vector<cv::Scalar> clustersCenters, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> & ptInClusters, double * weight) {

	// 对于图像中的每个点找到距离最近的聚类中心，并将该点暂时归于这个类
	for (int r = 0; r<imgInput.rows; r++) {
		for (int c = 0; c<imgInput.cols; c++) {

			double minDistance = INFINITY;
			int closestClusterIndex = 0;
			cv::Scalar pixel = imgInput.at<cv::Vec3b>(r, c);
			Feature pixelF;
			pixelF.x = r; pixelF.y = c;
			if (r != imgInput.rows - 1 & c != imgInput.cols - 1) {
				pixelF.x_diff = computeColorDistance(pixel, imgInput.at<cv::Vec3b>(r + 1, c));
				pixelF.y_diff = computeColorDistance(pixel, imgInput.at<cv::Vec3b>(r, c + 1));
			}
			else if (r == imgInput.rows - 1 & c != imgInput.cols - 1) {
				pixelF.x_diff = computeColorDistance(imgInput.at<cv::Vec3b>(r - 1, c), pixel);
				pixelF.y_diff = computeColorDistance(pixel, imgInput.at<cv::Vec3b>(r, c + 1));
			}
			else if (r != imgInput.rows - 1 & c == imgInput.cols - 1) {
				pixelF.x_diff = computeColorDistance(pixel, imgInput.at<cv::Vec3b>(r + 1, c));
				pixelF.y_diff = computeColorDistance(imgInput.at<cv::Vec3b>(r, c - 1), pixel);
			}

			for (int k = 0; k<clusters_number; k++) {
				cv::Scalar clusterPixel = clustersCenters[k];
				Feature clusterFeature = clustersFeatures[k];

				//使用特征向量计算距离,weight分别为三个特征向量的权重
				double distance;
				double colorDis = computeColorDistance(pixel, clusterPixel);
				double * featureDis = computeXYDistance(pixelF, clusterFeature);
				double allDis = colorDis + featureDis[0] + featureDis[1];
				distance = weight[0] * colorDis + weight[1] * featureDis[0] + weight[2] * featureDis[1];

				//更新聚类中心
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}
			//将该点归于最近的聚类中心
			ptInClusters[closestClusterIndex].push_back(cv::Point(r,c));
		}
	}
}

double KmeansAlgorithm::adjustClusterCenters(cv::Mat imgInput, int clusters_number, std::vector<cv::Scalar> & clustersCenters, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> ptInClusters, double & oldCenter, double newCenter, double * weight) {

	double diffChange;

	//将聚类中心调整为该类的均值点
	for (int k = 0; k<clusters_number; k++) {

		std::vector<cv::Point> ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;
		int newx = 0;
		int newy = 0;
		double newxdiff = 0;
		double newydiff = 0;

		for (int i = 0; i<ptInCluster.size(); i++) {
			cv::Scalar pixel = imgInput.at<cv::Vec3b>(ptInCluster[i].x, ptInCluster[i].y);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
			newx += ptInCluster[i].x;
			newy += ptInCluster[i].y;
			if (ptInCluster[i].x == imgInput.rows - 1){ 
				newxdiff += computeColorDistance(imgInput.at<cv::Vec3b>(ptInCluster[i].x - 1,ptInCluster[i].y), pixel); 
			}
			else{ 
				newxdiff += computeColorDistance(pixel, imgInput.at<cv::Vec3b>(ptInCluster[i].x + 1,ptInCluster[i].y)); 
			}
			if(ptInCluster[i].y == imgInput.cols - 1){ 
				newydiff += computeColorDistance(imgInput.at<cv::Vec3b>(ptInCluster[i].x,ptInCluster[i].y - 1), pixel);
			}
			else{ 
				newydiff += computeColorDistance(pixel, imgInput.at<cv::Vec3b>(ptInCluster[i].x,ptInCluster[i].y + 1));
			}
		}

		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();
		newx /= ptInCluster.size();
		newy /= ptInCluster.size();
		newxdiff /= ptInCluster.size();
		newydiff /= ptInCluster.size();

		cv::Scalar newPixel(newBlue, newGreen, newRed);
		Feature newFeature;
		newFeature.x, newFeature.y, newFeature.x_diff, newFeature.y_diff = int(newx), int(newy), newxdiff, newydiff;
		
		double * dis = computeXYDistance(newFeature, clustersFeatures[k]);
		double allDis = computeColorDistance(newPixel, clustersCenters[k]) + dis[0] + dis[1];
		// add distance 
		//newCenter += (weight[0] * computeColorDistance(newPixel, clustersCenters[k])  + weight[1] * dis[0]   + weight[2] * dis[1] )/allDis;
		//std::cout<<"newCenter:"<<newCenter<<"   oldCenter"<<oldCenter<<std::endl;
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		clustersCenters[k] = newPixel;
	}

	newCenter /= clusters_number;

	//get difference between previous iteration change
	diffChange = abs(oldCenter - newCenter);
	std::cout << "diffChange is: " << diffChange << std::endl;
	oldCenter = newCenter;

	return diffChange;
}

cv::Mat KmeansAlgorithm::applyFinalClusterToImage(cv::Mat & imgOutput, cv::Mat & imgLabel, int clusters_number, std::vector<Feature> & clustersFeatures, std::vector<std::vector<cv::Point>> ptInClusters) {

	srand(time(NULL));
	//assign random color to each cluster
	for (int k = 0; k<clusters_number; k++){
		std::vector<cv::Point> ptInCluster = ptInClusters[k];
		int randIndex = rand() % ptInCluster.size();
		cv::Scalar randomColor(imgLabel.at<cv::Vec3b>(ptInCluster[randIndex].x, ptInCluster[randIndex].y));
		//for each pixel in cluster change color to fit cluster
		for (int i = 0; i<ptInCluster.size(); i++) {

			cv::Scalar pixelColor = imgOutput.at<cv::Vec3b>(ptInCluster[i].x,ptInCluster[i].y);
			pixelColor = randomColor;

			imgOutput.at<cv::Vec3b>(ptInCluster[i].x, ptInCluster[i].y)[0] = pixelColor.val[0];
			imgOutput.at<cv::Vec3b>(ptInCluster[i].x, ptInCluster[i].y)[1] = pixelColor.val[1];
			imgOutput.at<cv::Vec3b>(ptInCluster[i].x, ptInCluster[i].y)[2] = pixelColor.val[2];
		}
	}
	return imgOutput;
}

double KmeansAlgorithm::calError(cv::Mat & imgOutputKNN, cv::Mat & imgLable) {
	double error =0 ;
	for (int r = 0; r<imgOutputKNN.rows; r++) {
		for (int c = 0; c<imgOutputKNN.cols; c++) {
			if(imgOutputKNN.at<cv::Vec3b>(r, c) != imgLable.at<cv::Vec3b>(r,c)){ error += 1;}
		}
	}
	error /= (imgOutputKNN.rows*imgOutputKNN.cols);
	error *= 100;
	return error;
}