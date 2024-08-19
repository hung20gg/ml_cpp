#include "../headers/function.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

std::vector<size_t> argsort(const std::vector<double>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

double distance(std::vector<double> &x1, std::vector<double> &x2){
    double sum = 0;
    for (int i = 0; i < x1.size(); i++){
        sum += pow(x1[i] - x2[i], 2);
    }
    return sqrt(sum);
}

double mean_squared_error(std::vector<double> &y_true, std::vector<double> &y_pred){
    double sum = 0;
    for (int i = 0; i < y_true.size(); i++){
        sum += pow(y_true[i] - y_pred[i], 2);
    }
    return sum / y_true.size();
}

double mean_absolute_error(std::vector<double> &y_true, std::vector<double> &y_pred){
    double sum = 0;
    for (int i = 0; i < y_true.size(); i++){
        sum += abs(y_true[i] - y_pred[i]);
    }
    return sum / y_true.size();
}

double binary_crossentropy(std::vector<double> &y_true, std::vector<double> &y_pred){
    double sum = 0;
    for (int i = 0; i < y_true.size(); i++){
        sum += y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]);
    }
    return -sum / y_true.size();
}

double cross_entropy(std::vector<std::vector<int>> &y_true, std::vector<std::vector<double>> &y_pred){
    double sum = 0;
    for (int i = 0; i < y_true.size(); i++){
        for (int j = 0; j < y_true[i].size(); j++){
            sum = y_true[i][j] * log(y_pred[i][j]);
        }
    }
    return -sum / y_true.size();
}

double cross_entropy(std::vector<int> &y_true, std::vector<std::vector<double>> &y_pred){
    double sum = 0;
    for (int i = 0; i < y_pred.size(); i++){
        for (int j = 0; j < y_pred[i].size(); j++){
            if (y_true[i] == j)
                sum += log(y_pred[i][j]);
        }
        
    }
    return -sum / y_true.size();
}

double entropy(std::vector<double> &y_true, std::vector<double> &y_pred){
    double val = 0;
    for (int i = 0; i < y_true.size(); i++){
        val += y_true[i] * log(y_pred[i]);
    }
}

std::vector<double> softmax(std::vector<double> &x){
    std::vector<double> result(x.size());
    double sum = 0;
    for (int i = 0; i < x.size(); i++){
        sum += exp(x[i]);
    }
    for (int i = 0; i < x.size(); i++){
        result[i] = exp(x[i]) / sum;
    }
    return result;
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

std::vector<double> sigmoid(std::vector<double> &x){
    std::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++){
        result[i] = 1 / (1 + exp(-x[i]));
    }
    return result;
}

std::vector<int> pickKRandomInRangeN(int k, int n) {
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 0);  // Fill with 0 to n-1
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    std::shuffle(result.begin(), result.end(), g);
    result.resize(k);
    
    return result;
}