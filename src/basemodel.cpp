#include "../headers/basemodel.hpp"
#include <iostream>
#include <vector>
#include <cmath>

std::vector<double> SupervisedBasedModel::predict_proba(std::vector<std::vector<double>> X){

    std::vector<double> y_pred;
    for (int i = 0; i < X.size(); i++){
        double sum = 0;
        for (int j = 0; j < X[i].size(); j++){
            sum += X[i][j] * this->weights[j];
        }
        y_pred.push_back(sum);
    }
    return y_pred;
}

std::vector<double> SupervisedBasedModel::predict(std::vector<std::vector<double>> X){
    std::vector<double> y_pred;
    for (int i = 0; i < X.size(); i++){
        double sum = 0;
        for (int j = 0; j < X[i].size(); j++){
            sum += X[i][j] * this->weights[j];
        }
        y_pred.push_back(sum > 0 ? 1 : 0);
    }
    return y_pred;
}
