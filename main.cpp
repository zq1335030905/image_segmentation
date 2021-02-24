#include "seg.h"

int main(int argc, const char * argv[]) {
	vector<Scalar> clustersCenters;
	vector<Feature> clustersFeatures;
	vector< vector<Point> > ptInClusters;
	double threshold;
	double weight[3];
	double oldCenter;
	double newCenter;
	double diffChange;
	double alpha;//color weight
	double beta; //xy weight
	double gamma;//diff weight
	double error;
	double error_threshold;
	int round = 1;

    string path;
	std::cout << "Please enter the image path you want to segment:";
	std::cin >> path;

	string label_path;
	std::cout << "Please enter the labeled image path:";
	std::cin >> label_path;

    Mat imgInput = imread(path,CV_LOAD_IMAGE_COLOR);
	Mat imgOutputKNN = imgInput.clone();
	Mat imgLabel = imread(label_path, CV_LOAD_IMAGE_COLOR);
    
    if(imgInput.empty()){
        printf("Error opening image.\n");
        return -1;
    }
     //The number of cluster is the only parameter to choose
    int clusters_number;
	std::cout << "Please enter the number of cluster centers: ";
	std::cin >> clusters_number;

	while (1) {
	int choose;
	error = 100;
	std::cout<<"Please choose the segmentation mode: 1.simple, 2.middle, 3.hard, 4.quit :";
	std::cin>>choose;
    
	switch (choose)
	{
	//simple
	case 1:
		error_threshold = 70;
		break;
	//middle
	case 2:
		error_threshold = 60;
		break;
	//hard
	case 3:
		error_threshold = 50;
		break;
	case 4:
		return 0;
	}
   
    //set up cluster center, cluster vector, and parameter to stop the iterations
    while(error>error_threshold){
		std::cout<<   "Round "<<round<<"~   "<<endl;
		round += 1;
		//init
		threshold = 0.1;
		oldCenter=INFINITY;
		newCenter=0;
		diffChange = oldCenter - newCenter;
		alpha = rand()%10;//color weight
		beta = rand()%10; //xy weight
		gamma = rand()%10;//diff weight
		weight[0] = alpha; weight[1]=beta; weight[2]=gamma;

		//create ramdom clusters centers and clusters vectors
		KmeansAlgorithm::createClustersInfo(imgInput, clusters_number, clustersCenters, clustersFeatures, ptInClusters);
    
		//iterate until cluster centers nearly stop moving (using threshold)
		while( diffChange > threshold){
        
			//reset change
			newCenter = 0;
        
			//clear associated pixels for each cluster
			for(int k=0; k<clusters_number; k++){
				ptInClusters[k].clear();
			}
        
			//find all closest pixel to cluster centers
			KmeansAlgorithm::findAssociatedCluster(imgInput, clusters_number, clustersCenters, clustersFeatures, ptInClusters, weight);
        
			//recompute cluster centers values
			diffChange = KmeansAlgorithm::adjustClusterCenters(imgInput, clusters_number, clustersCenters, clustersFeatures, ptInClusters, oldCenter, newCenter, weight);
		}
		imgOutputKNN = KmeansAlgorithm::applyFinalClusterToImage(imgOutputKNN,imgLabel, clusters_number, clustersFeatures, ptInClusters);
		//calculate the error of segmentation
		error = KmeansAlgorithm::calError(imgOutputKNN, imgLabel);
		std::cout<<"Error:"<<error<<"%"<<std::endl;
	}
	std::cout<<"alpha: "<<alpha<<"  beta: "<<beta<<"  gamma: "<<gamma<<std::endl;
	
    cv::imshow("Segmentation", imgOutputKNN);
    cv::imwrite("KMeansSegmentation.jpg", imgOutputKNN);
    cv::waitKey(0);
	cv::destroyWindow("Segmentation");
    //return 0;
	}
	
}

