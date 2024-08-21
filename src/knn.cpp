#include "../headers/knn.hpp"
#include "../headers/function.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>

// #include <utility>

void KNN::fit(std::vector<std::vector<double>> &X, std::vector<double> &y){
    std::cout << "Fitting the model" << std::endl;


    // Store the training data, not actually fitting anything  
    this->_X_train = X;
    this->_y_train = y; 
    std::set<int>labels;
    for (int i = 0; i < y.size(); i++){
        labels.insert(y[i]);
    }
    this->num_classes = labels.size();

};

double KNN::__predict_one(std::vector<double> x){
    std::cout << "Predicting one" << std::endl;

    std::vector<std::pair<double, double>> distances(this->_X_train.size());

    // Calculate the distance between x and all the training data
    for (int i = 0; i < this->_X_train.size(); i++){
        distances[i] = std::make_pair(distance(_X_train[i], x) , this->_y_train[i]);
    }
    sort(distances.begin(), distances.end());

    // Get the labels of the k nearest neighbors
    std::vector<int> vote(this->num_classes, 0);
    for (int i = 0; i < this->n_neighbors; i++){
        vote[distances[i].second]++;
    }
    int max_vote = 0;
    double max_label = -1;


    // Get the label with the most votes
    for (int i = 0; i < this->num_classes; i++){
        if (vote[i] > max_vote){
            max_vote = vote[i];
            max_label = i;
        }
    }

    return max_label;
};

std::vector<double> KNN::predict(std::vector<std::vector<double>> &X){
    std::cout << "Predicting" << std::endl;
    std::vector<double> y_pred;

    for (int i = 0; i < X.size(); i++){
        double y = this->__predict_one(X[i]);
        y_pred.push_back(y);
    }

    return y_pred;

};

std::vector<double> KNN::predict_proba(std::vector<std::vector<double>> &X){
    return this->predict(X);

};