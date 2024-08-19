#include "../headers/logistic_regression.hpp"
#include "../headers/function.hpp"
#include "../headers/regularization.hpp"
#include "../headers/loss_function.hpp"
#include <iostream>
#include <vector>
#include <cmath>


void LogisticRegression::fit(std::vector<std::vector<double>> X, std::vector<double> y)  {
    std::cout << "Fitting the model" << std::endl;

    // Initialize the weights
    for (int i = 0; i < X[0].size(); i++){
        this->weights.push_back(0);
    }

    // Loss function
    BinaryCrossEntropy loss;

    for(int i = 0 ; i < this->_max_iter ; i ++){
        // Forward pass
        std::vector<double> y_pred = predict_proba(X);
        double error = loss.forward(y, y_pred);

        // Break if the error is less than the tolerance
        if (error < this->_tol){
            break;
        }

        //Backward pass
        // Update the weights
        // for (int j = 0; j < X[0].size(); j++){
        //     double sum = 0;

        //     for (int k = 0; k < X.size(); k++){
        //         sum += (y_pred[k] - y[k]) * X[k][j];
        //     }
        //     this->weights[j] -= this->_learning_rate * sum / X.size();
        // }

        // Chain rules (not more efficient but easier to understand)
        std::vector<double> grad_loss = loss.backward(y, y_pred);
        std::vector<double> grad_sigmoid;
        for (int j = 0; j < X.size(); j++){
            grad_sigmoid.push_back(y_pred[j] * (1 - y_pred[j]));
        }
        std::vector<double> sum(X[0].size(), 0);

        // DL/dw = (y_pred - y)TX 
        for (int j = 0; j < X.size(); j++){
            for (int k = 0; k < X[0].size(); k++){
                sum[k] += (grad_loss[j] * grad_sigmoid[j] * X[j][k]);
            }
        }

        // Update the weights
        for (int j = 0; j < X[0].size(); j++){
            this->weights[j] -= this->_learning_rate * sum[j] / X.size();
        }


        // Regularization
        if (this->_regularization == "L1"){
            this->weights -= this->_learning_rate * this->_lambda * L1_regularization(this->weights) / X.size();
        }
        else if (this->_regularization == "L2"){
            this->weights -= this->_learning_rate * this->_lambda * L2_regularization(this->weights) / X.size();
        }

    }

};

std::vector<double> LogisticRegression::predict_proba(std::vector<std::vector<double>> X){
    std::vector<double> result;
    for (int i = 0; i < X.size(); i++){
        double sum = 0;
        for (int j = 0; j < X[0].size(); j++){
            sum += this->weights[j] * X[i][j];
        }
        result.push_back(sum);
    }
    return sigmoid(result);
};

std::vector<double> LogisticRegression::predict(std::vector<std::vector<double>> X){
    std::vector<double> result;
    std::vector<double> y_pred = predict_proba(X);
    for (int i = 0; i < y_pred.size(); i++){
        if (y_pred[i] > 0.5){
            result.push_back(1);
        }
        else{
            result.push_back(0);
        }
    }
    return result;
};