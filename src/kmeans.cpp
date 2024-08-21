#include "../headers/kmeans.hpp"
#include "../headers/function.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int Kmeans::clustering(std::vector<Point>& points){
    int changes = 0;

    for (int i = 0; i < this->n_clusters; i++){    
        for (int j = 0; j < points.size(); j++){
            if (points[j].distance(this->centroids[i]) < points[j].distance(this->centroids[points[j].cluster])){
                points[j].cluster = this->centroids[i].cluster;

                changes++;
            }
        }
    }
    return changes;

}

void Kmeans::fit(std::vector<std::vector<double>>& X){
    std::cout << "Fitting the model" << std::endl;

    std::vector<Point> points;
    for (int i = 0; i < X.size(); i++){
        Point p(X[i], -1);
        points.push_back(p);
    }
    // Initialize random centroids

    int n_samples = X.size();
    std::vector<int> random_indices = pickKRandomInRangeN(this->n_clusters, n_samples);
    
    for (int i = 0; i < this->n_clusters; i++){
        Point p(X[random_indices[i]], i);
        this->centroids.push_back(p);
    }

    // Assign points to the nearest centroid

    int current_iter = 0;

    while (current_iter < this->max_iter){
        int changes = clustering(points);
        if (changes == 0){
            break;
        }

        // Update the centroids
        std::vector<std::vector<double>>distances(n_clusters, std::vector<double>(X[0].size(), 0));
        std::vector<int> counts(n_samples, 0);

        for (int i = 0; i < points.size(); i++){
            int cluster = points[i].cluster;
            for (int j = 0; j < X[i].size(); j++){
                distances[cluster][j] += X[i][j];
            }
            counts[cluster]++;
        }
        
        for (int i = 0; i < this->n_clusters; i++){
            for (int j = 0; j < X[0].size(); j++){
                this->centroids[i].features[j] = distances[i][j] / counts[i];
            }
        }

        current_iter++;

    }

    std::cout<<"Fitting completed"<<std::endl;

};

std::vector<double> Kmeans::predict(std::vector<std::vector<double>>& X){
    std::cout << "Predicting" << std::endl;
    std::vector<double> y_pred;

    // Assign points to the nearest centroid
    for (int i = 0; i < X.size(); i++){
        double min_distance = INT_MAX;
        int cluster = -1;

        // Find the nearest centroid
        for (int j = 0; j < this->n_clusters; j++){
            double current_distance = distance(X[i], this->centroids[j].features);
            if (current_distance < min_distance){
                min_distance = current_distance;
                cluster = j;
            }
        }
        y_pred.push_back(cluster);
    }
    return y_pred;
}