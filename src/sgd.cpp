#include "../headers/sgd.hpp"
#include "../headers/function.hpp"
#include "../headers/loss_function.hpp"
#include <iostream>
#include <vector>
#include <cmath>


// Not implementing try-catch num labels for now
void SGDClassifier::fit(std::vector<std::vector<double>> X, std::vector<double> y){
    std::cout << "Fitting the model" << std::endl;

    // Initialize the weights
    for (int i = 0; i < X[0].size(); i++){
        this->weights.push_back(0);
    }

    // Loss function and Regularization
    LossFunction* loss = createLossFunction(this->_loss);
    Regularization* reg = createRegularization(this->_regularization);

    for(int epoch = 0 ; epoch < this->_max_iter ; epoch ++){
        
        // Mini-batch
        for (int i = 0; i < X.size(); i += this->_batch_size){

            std::vector<std::vector<double>>X_batch = slice2D(X, i, this->_batch_size);
            std::vector<double>y_batch = slice1D(y, i, this->_batch_size);
            // Forward pass
            std::vector<double> y_pred = predict_proba(X_batch);
            double error = loss->forward(y_batch, y_pred);
            double reg_error = this->_lambda * reg->forward(this->weights);

            // Break if the error is less than the tolerance
            if (error + reg_error < this->_tol){
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
                this->weights[j] -= this->_learning_rate * grad_reg[j] / X.size();
            }

        }
    }

};