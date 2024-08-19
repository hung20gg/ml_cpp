#include "../headers/basemodel.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// void BaseModel::fit(std::vector<std::vector<double>> X, std::vector<double> y)  {
//     std::cout << "Fitting the model" << std::endl;

//     // Initialize the weights
//     for (int i = 0; i < X[0].size(); i++){
//         this->weights.push_back(0);
//     }

// };

std::vector<double> BaseModel::predict_proba(std::vector<std::vector<double>> X){

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

std::vector<double> BaseModel::predict(std::vector<std::vector<double>> X){
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
