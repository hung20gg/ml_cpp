#pragma once
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

class DBSCAN : public ClusteringModel {
    private:
        double eps;
        int min_samples;
        
    public:
        std::vector<Point> points;
        std::vector<int> labels;

        DBSCAN(double eps, int min_samples): ClusteringModel(){
            this->eps = eps;
            this->min_samples = min_samples;

            this->name = "DBSCAN";
            this->description = "DBSCAN is a clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.";
            this->type = "clustering";
        };

        void fit(std::vector<std::vector<double>> X) override;
        void expand_cluster(Point &p, int cluster_id);
        std::vector<int> range_query(Point &p);
        std::vector<int> predict(std::vector<std::vector<double>> X) override;

}