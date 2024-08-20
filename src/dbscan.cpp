#include "../headers/dbscan.hpp"
#include <iostream>
#include <vector>
#include <cmath>

std::vector<int> DBSCAN::range_query(Point &p){
    std::vector<int> neighbors;
    for (int i = 0; i < this->points.size(); i++){
        if (p.distance(this->points[i]) <= this->eps){
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

void DBSCAN::expand_cluster(Point &p, int cluster_id){
    std::vector<int> neighbors = range_query(p);
    if (neighbors.size() < this->min_samples){
        p.cluster = -1;
    } else {
        p.cluster = cluster_id;
        for (int i = 0; i < neighbors.size(); i++){
            if (this->points[neighbors[i]].cluster == 0){
                expand_cluster(this->points[neighbors[i]], cluster_id);
            }
        }
    }
}

void fit(std::vector<std::vector<double>> X){
    std::vector<int> labels(X.size(), 0);
    int cluster = 0;
    for (int i = 0; i < X.size(); i++){
        if (labels[i] != -1){
            continue;
        }
        std::vector<int> neighbors = range_query(X, i);
        if (neighbors.size() < min_samples){
            labels[i] = -1;
        } else {
            cluster += 1;
            expand_cluster(X, labels, i, neighbors, cluster);
        }
    }

}