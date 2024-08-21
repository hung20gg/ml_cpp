#pragma once
#include "basemodel.hpp"
#include <iostream>
#include <vector>
#include <cmath>

struct Point {
    std::vector<double> features;
    int cluster = 0;
    int id;

    Point(std::vector<double> features, int cluster, int id) : features(features), cluster(cluster), id(id) {}

    void change_cluster(int cluster) {
        this->cluster = cluster;
    }

    double distance(Point p) {
        double sum = 0;
        for (int i = 0; i < this->features.size(); i++) {
            sum += pow(this->features[i] - p.features[i], 2);
        }
        return sqrt(sum);
    }
};

class DBSCAN : public UnsuperivedBasedModel {
    private:
        double eps;
        int min_samples;
        
    public:
        std::vector<Point> points;
        std::vector<int> labels;

        DBSCAN(double eps, int min_samples): UnsuperivedBasedModel(){
            this->eps = eps;
            this->min_samples = min_samples;

            this->name = "DBSCAN";
            this->description = "DBSCAN is a clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.";
            this->type = "clustering";
        };

        // Gonna change it to pointer soon
        void fit(std::vector<std::vector<double>> &X) override;
        void expand_cluster(Point *p, std::vector<Point*> &neighbor, int &cluster_id);
        std::vector<Point*> range_query(Point* p);

};