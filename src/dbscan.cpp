#include "../headers/dbscan.hpp"
#include <iostream>
#include <vector>
#include <cmath>

std::vector<Point*> DBSCAN::range_query(Point* p){
    std::vector<Point*> neighbors;
    for (int i = 0; i < this->points.size(); i++){
        if (p->distance(this->points[i]) <= this->eps){
            neighbors.push_back(&this->points[i]);
        }
    }
    return neighbors;
}

void DBSCAN::expand_cluster(Point *p, std::vector<Point*> neighbor, int cluster_id){

    // Change the cluster of the point
    p->change_cluster(cluster_id);
    this->labels[p->id] = cluster_id;

    // Iterate over the neighbors

    while (!neighbor.empty()){
        Point* current_point = neighbor.back();
        neighbor.pop_back();

        // If the point is an outlier, change it to a border point
        if (this->labels[current_point->id] == -1){

            this->labels[current_point->id] = cluster_id;
            current_point->change_cluster(cluster_id);
        }

        else if (this->labels[current_point->id] == 0){

            // If the point is not visited, change it to a border point
            this->labels[current_point->id] = cluster_id;
            current_point->change_cluster(cluster_id);

            // Get the neighbors of the current point
            std::vector<Point*> current_neighbors = range_query(current_point);

            // If the number of neighbors is greater than min_samples, add them to the neighbor list
            if (current_neighbors.size() >= this->min_samples){
                neighbor.insert(neighbor.end(), current_neighbors.begin(), current_neighbors.end());
            }
        }
    }   
}

void DBSCAN::fit(std::vector<std::vector<double>> X){
    this->labels.resize(X.size(), 0.0);
    for (int i = 0; i < X.size(); i++){
        this->points.push_back(Point(X[i], 0, i));
    }
    int cluster = 0;
    for (int i = 0; i < X.size(); i++){
        if (this->labels[i] != 0){
            continue;
        }
        std::vector<Point*> neighbors = range_query(&this->points[i]);

        // If the number of neighbors is less than min_samples, it's an outlier
        if (neighbors.size() < min_samples){
            this->labels[i] = -1;
            this->points[i].change_cluster(-1);
        } else {

            // Expand the cluster
            cluster += 1;
            expand_cluster(&this->points[i], neighbors, cluster);
        }
    }

}
