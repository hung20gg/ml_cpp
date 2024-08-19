#include "../headers/loss_function.hpp"
#include "../headers/function.hpp"
#include <iostream>
#include <vector>
#include <cmath>

double MeanSquaredError::forward(std::vector<double> y_true, std::vector<double> y_pred){

    return mean_squared_error(y_true, y_pred);
};

std::vector<double> MeanSquaredError::backward(std::vector<double> y_true, std::vector<double> y_pred){

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(2 * (y_pred[i] - y_true[i]));
    }
    return grad;
};

double MeanAbsoluteError::forward(std::vector<double> y_true, std::vector<double> y_pred){

    return mean_absolute_error(y_true, y_pred);
};

std::vector<double> MeanAbsoluteError::backward(std::vector<double> y_true, std::vector<double> y_pred){

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back(y_pred[i] - y_true[i]);
    }
    return grad;
};

double BinaryCrossEntropy::forward(std::vector<double> y_true, std::vector<double> y_pred){

    return binary_crossentropy(y_true, y_pred);
};

std::vector<double> BinaryCrossEntropy::backward(std::vector<double> y_true, std::vector<double> y_pred){

    std::vector<double> grad;
    for (int i = 0; i < y_true.size(); i++){
        grad.push_back((y_pred[i] - y_true[i]) / (y_pred[i] * (1 - y_pred[i])));
    }
    return grad;
};


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
};
