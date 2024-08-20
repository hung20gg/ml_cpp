#include "basemodel.hpp"
#include <iostream>
#include <vector>
#include <cmath>

struct Point {
    std::vector<double> features;
    int cluster;

    Point(std::vector<double> features, int cluster) : features(features), cluster(cluster) {}

    double distance(Point p) {
        double sum = 0;
        for (int i = 0; i < this->features.size(); i++) {
            sum += pow(this->features[i] - p.features[i], 2);
        }
        return sqrt(sum);
    }
};

class Kmeans : public UnsuperivedBasedModel {
    private:
        int max_iter;
        
    public:
        std::vector<Point> centroids;
        int n_clusters;

        Kmeans(int n_clusters, int max_iter = 500): UnsuperivedBasedModel(){
            this->n_clusters = n_clusters;
            this->max_iter = max_iter;

            this->name = "Kmeans";
            this->description = "Kmeans is a clustering algorithm that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.";
            this->type = "clustering";
        };

        void fit(std::vector<std::vector<double>> X) override;
        int clustering(std::vector<Point>&points); // Return the change in clusters
        std::vector<double> predict(std::vector<std::vector<double>> X);
};