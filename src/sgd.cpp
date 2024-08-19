#include "../headers/sgd.hpp"
#include "../headers/function.hpp"
#include "../headers/regularization.hpp"
#include "../headers/loss_function.hpp"
#include <iostream>
#include <vector>
#include <cmath>

void SGDClassifier::fit(std::vector<std::vector<double>> X, std::vector<double> y){
    std::cout << "Fitting the model" << std::endl;

    // Initialize the weights
    for (int i = 0; i < X[0].size(); i++){
        this->weights.push_back(0);
    }

    // Loss function
    // if (this->_loss == "binary"){
    //     BinaryCrossEntropy loss ;
    // }else if (this->_loss == "mse"){
    //     MeanSquaredError loss;
    // }else if (this->_loss == "mae"){
    //     MeanAbsoluteError loss;
    // }else {//  if (this->_loss == "hinge"){
    //     Hinge loss;
    // }
    // }else {
    //     CrossEntropyWithSoftmax loss;
    // }

    // Implement dynamic loss later

    BinaryCrossEntropy loss;


    for(int epoch = 0 ; epoch < this->_max_iter ; epoch ++){
        
        // Mini-batch
        for (int i = 0; i < X.size(); i += this->_batch_size){

            std::vector<std::vector<double>>X_batch = slice2D(X, i, this->_batch_size);
            std::vector<double>y_batch = slice1D(y, i, this->_batch_size);
            // Forward pass
            std::vector<double> y_pred = predict_proba(X_batch);
            double error = loss.forward(y_batch, y_pred);

            // Break if the error is less than the tolerance
            if (error < this->_tol){
                break;
            }

            std::vector<double> grad_loss = loss.backward(y_batch, y_pred);
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
    }

};