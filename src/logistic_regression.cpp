#include "../headers/logistic_regression.hpp"
#include "../headers/function.hpp"
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

    // Loss function and Regularization
    LossFunction* loss = createLossFunction("binary");
    Regularization* reg = createRegularization(this->_regularization);


    // Regularization
    // if (this->_penalty == "l1"){

    for(int i = 0 ; i < this->_max_iter ; i ++){
        // Forward pass
        std::vector<double> y_pred = predict_proba(X);
        double error = loss->forward(y, y_pred) + this->_lambda * reg->forward(this->weights);
        double reg_error = this->_lambda * reg->forward(this->weights);

        // Break if the error is less than the tolerance
        if (error + reg_error < this->_tol){
            break;
        }

        //Backward pass
        // Update the weights (Fast but hard to understand)
        // for (int j = 0; j < X[0].size(); j++){
        //     double sum = 0;

        //     for (int k = 0; k < X.size(); k++){
        //         sum += (y_pred[k] - y[k]) * X[k][j];
        //     }
        //     this->weights[j] -= this->_learning_rate * sum / X.size();
        // }

        // Chain rules (not more efficient but easier to understand)
        std::vector<double> grad_loss = loss->backward(y, y_pred);
        std::vector<double> grad_reg = reg->backward(this->weights);

        std::vector<double> grad_sigmoid;
        for (int j = 0; j < X.size(); j++){
            grad_sigmoid.push_back(y_pred[j] * (1 - y_pred[j]));
        }
        std::vector<double> Dw(X[0].size(), 0);

        // DL/dw = XT*DL/dy * dy/dz
        for (int j = 0; j < X.size(); j++){
            for (int k = 0; k < X[0].size(); k++){
                Dw[k] += (grad_loss[j] * grad_sigmoid[j] * X[j][k]);
            }
        }

        // Update the weights
        
        for (int j = 0; j < X[0].size(); j++){
            this->weights[j] -= this->_learning_rate * Dw[j] / X.size();
        }

        // Regularization
        // w(k+1) = w(k) - lr * (DL/dw + lambda * dR/dw)
        for (int j = 0; j < X[0].size(); j++){
            this->weights[j] -= this->_learning_rate * grad_reg[j] / X.size();
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