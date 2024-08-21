#include "../headers/sgd.hpp"
#include "../headers/function.hpp"
#include "../headers/loss_function.hpp"
#include <iostream>
#include <vector>
#include <cmath>


std::vector<double> SGDClassifier::predict_proba(std::vector<std::vector<double>> &X){
    std::vector<double> result;
    for (int i = 0; i < X.size(); i++){
        double sum = 0;
        for (int j = 0; j < X[0].size(); j++){
            sum += this->weights[j] * X[i][j];
        }
        result.push_back(sum);
    }
    return result;
};

// Not implementing try-catch num labels for now
void SGDClassifier::fit(std::vector<std::vector<double>>& X, std::vector<double>& y){
    std::cout << "Fitting the model" << std::endl;

    // Initialize the weights
    for (int i = 0; i < X[0].size(); i++){
        this->weights.push_back(0);
    }

    // Loss function and Regularization
    LossFunction* loss = createLossFunction(this->_loss);
    Regularization* reg = createRegularization(this->_regularization);

    double prev_error = INT_MAX;

    // Epochs
    for(int epoch = 0 ; epoch < this->_max_iter ; epoch ++){
        
        // Mini-batch
        for (int i = 0; i < X.size(); i += this->_batch_size){

            std::vector<std::vector<double>>X_batch = slice2D(X, i, this->_batch_size);
            std::vector<double> y_batch = slice1D(y, i, this->_batch_size);
            // Forward pass
            std::vector<double> y_pred = this->predict_proba(X_batch);
            double error = loss->forward(y_batch, y_pred);
            double reg_error = this->_lambda * reg->forward(this->weights);
            double tol_error = error + reg_error;

            // Break if the error is less than the tolerance
            if (std::abs(tol_error - prev_error) < this->_tol){
                break;
            }

            std::vector<double> grad_loss = loss->backward(y, y_pred);
            std::vector<double> grad_reg = reg->backward(this->weights);
            
            std::vector<double> Dw(X[0].size(), 0);

            // DL/dw = XT*Dl/dy 
            for (int j = 0; j < X.size(); j++){
                for (int k = 0; k < X[0].size(); k++){
                    Dw[k] += (grad_loss[j] * X[j][k]);
                }
            }

            // Update the weights
            for (int j = 0; j < X[0].size(); j++){
                this->weights[j] -= this->_learning_rate * Dw[j] / X.size();
            }

            // Regularization
            // w(k+1) = w(k) - lr * (DL/dw + lambda * dR/dw)
            for (int j = 0; j < X[0].size(); j++){
                this->weights[j] -= this->_learning_rate * this->_lambda * grad_reg[j] / X.size();
            }

        }
    }

};



std::vector<double> SGDClassifier::predict(std::vector<std::vector<double>> &X){
    std::vector<double> y_pred;
    for (int i = 0; i < X.size(); i++){
        double sum = 0;
        for (int j = 0; j < X[i].size(); j++){
            sum += X[i][j] * this->weights[j];
        }
        y_pred.push_back(sum > 0 ? 1 : 0);
    }

    if (this->_loss == "logistic"){
        for (int i = 0; i < y_pred.size(); i++){
            y_pred[i] = sigmoid(y_pred[i]) > 0 ? 1 : 0;
        }
    }
    else if (this->_loss == "hinge"){
        for (int i = 0; i < y_pred.size(); i++){
            y_pred[i] = y_pred[i] > 0 ? 1 : -1;
        }
    }

    
};