#include "../headers/loss_function.hpp"
#include "../headers/function.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>


// Mean Squared Error
double MeanSquaredError::forward(std::vector<double> y_true, std::vector<double> y_pred) const{

    return mean_squared_error(y_true, y_pred);
};

std::vector<double> MeanSquaredError::backward(std::vector<double> y_true, std::vector<double> y_pred) const{

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(2 * (y_pred[i] - y_true[i]));
    }
    return grad;
};


// Mean Absolute Error
double MeanAbsoluteError::forward(std::vector<double> y_true, std::vector<double> y_pred) const{

    return mean_absolute_error(y_true, y_pred);
};

std::vector<double> MeanAbsoluteError::backward(std::vector<double> y_true, std::vector<double> y_pred) const{

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(y_pred[i] - y_true[i]);
    }
    return grad;
};


// Binary Cross Entropy
double BinaryCrossEntropy::forward(std::vector<double> y_true, std::vector<double> y_pred) const{

    return binary_crossentropy(y_true, y_pred);
};

std::vector<double> BinaryCrossEntropy::backward(std::vector<double> y_true, std::vector<double> y_pred) const{

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back((y_pred[i] - y_true[i]) / (y_pred[i] * (1 - y_pred[i])));
    }
    return grad;
};

// Unusable, for overwriting purposes only
double CrossEntropyWithSoftmax::forward(std::vector<double> y_true, std::vector<double> y_pred) const{
    return 0;
};


// Cross Entropy with Softmax
double CrossEntropyWithSoftmax::forward(std::vector<std::vector<int>> y_true, std::vector<std::vector<double>> y_pred){

    return cross_entropy(y_true, y_pred);
};

double CrossEntropyWithSoftmax::forward(std::vector<int> y_true, std::vector<std::vector<double>> y_pred){

    return cross_entropy(y_true, y_pred);
};


std::vector<double> CrossEntropyWithSoftmax::backward(std::vector<std::vector<int>> y_true, std::vector<std::vector<double>> y_pred){

    std::vector<double> grad;
    std::vector<std::vector<double>> softmax_values;
    for (int i = 0; i < y_pred.size(); i++){
        softmax_values.push_back(softmax(y_pred[i]));
    }
    for (int i = 0; i < y_true.size(); i++){
        double gard_sum = 0;
        for (int j = 0; j < y_true[i].size(); j++){
            gard_sum += softmax_values[i][j] - y_true[i][j];
        }
        grad.push_back(gard_sum);
    }
    return grad;
};

std::vector<double> CrossEntropyWithSoftmax::backward(std::vector<int> y_true, std::vector<std::vector<double>> y_pred){

    std::vector<double> grad;
    std::vector<double> softmax_values;
    for (int i = 0; i < y_pred.size(); i++){
        softmax_values.push_back(softmax(y_pred[i])[y_true[i]]);
    }
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(softmax_values[i] - 1);
    }
    return grad;
};

// Unusable, for overwriting purposes only
std::vector<double> CrossEntropyWithSoftmax::backward(std::vector<double> y_true, std::vector<double> y_pred) const{

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(y_pred[i] - y_true[i]);
    }
    return grad;
};


// Hinge Loss
double Hinge::forward(std::vector<double> y_true, std::vector<double> y_pred)const{
    double loss = 0;
    for (int i = 0; i < y_true.size(); i++){
        loss += std::max(0.0, 1 - y_true[i] * y_pred[i]);
    }
    return loss;
};

std::vector<double> Hinge::backward(std::vector<double> y_true, std::vector<double> y_pred) const {
    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(1 - y_true[i] * y_pred[i] > 0 ? -y_true[i] : 0);
    }
    return grad;
};


// L1 Regularization
double L1Loss::forward(std::vector<double> weights) const {
    return L1_regularization(weights);
};

std::vector<double> L1Loss::backward(std::vector<double> weights) const {
    std::vector<double> grad;
    for (int i = 0; i < weights.size(); i++){
        grad.push_back(weights[i] > 0 ? 1 : -1);
    }
    return grad;
};


// L2 Regularization
double L2Loss::forward(std::vector<double> weights) const {
    return L2_regularization(weights);
};

std::vector<double> L2Loss::backward(std::vector<double> weights) const {
    std::vector<double> grad;
    for (int i = 0; i < weights.size(); i++){
        grad.push_back(2 * weights[i]);
    }
    return grad;
};