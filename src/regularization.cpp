#include "../headers/regularization.hpp"
#include <iostream>
#include <vector>
#include <cmath>

double L1_regularization(std::vector<double> weights){
    double sum = 0;
    for (int i = 0; i < weights.size(); i++){
        sum += abs(weights[i]);
    }
    return sum;
}

double L2_regularization(std::vector<double> weights){
    double sum = 0;
    for (int i = 0; i < weights.size(); i++){
        sum += pow(weights[i], 2);
    }
    return sum;
}
