#include "../headers/svm.hpp"
#include "../headers/function.hpp"
#include <vector>
#include <cmath>

// Kernel functions
double SVC::__linear_kernel(std::vector<double>& x1, std::vector<double>& x2){
    return dot(x1, x2);
}

double SVC::__polynomial_kernel(std::vector<double>& x1, std::vector<double>& x2){
    return pow(dot(x1, x2) + 1, this->_degree);
}

double SVC::__rbf_kernel(std::vector<double>& x1, std::vector<double>& x2){
    return exp(-this->_gamma * distance(x1, x2));
}

// Currently only supports 2 classes 1 and -1
void SVC::fit(std::vector<std::vector<double>>& X, std::vector<double>& y){


    // Store the data
    this->_X = X;
    this->_y = y;

    // Initialize weights
    std::vector<double> weights(X[0].size(), 0);

    // Initialize learning rate
    double learning_rate = this->_learning_rate;

    // Initialize beta
    double beta = this->_beta;

    // Initialize kernel
    std::string kernel = this->_kernel;

    // Initialize coef0
    double coef0 = this->_coef0;

    // Initialize tolerance
    double tol = this->_tol;

    // Initialize max_iter
    int max_iter = this->_max_iter;

    // Initialize C
    double C = this->_C;

    // Initialize alpha
    std::vector<double> alpha(X.size(), 0);


    // Initialize error cache
    std::vector<double> error_cache(X.size(), 0);

    // Initialize kernel matrix
    std::vector<std::vector<double>> kernel_matrix(X.size(), std::vector<double>(X.size(), 0));

    // Initialize kernel matrix
    for (int i = 0; i < X.size(); i++){
        for (int j = 0; j < X.size(); j++){
            if (kernel == "linear"){
                kernel_matrix[i][j] = __linear_kernel(X[i], X[j]);
            }
            else if (kernel == "polynomial"){
                kernel_matrix[i][j] = __polynomial_kernel(X[i], X[j]);
            }
            else if (kernel == "rbf"){
                kernel_matrix[i][j] = __rbf_kernel(X[i], X[j]);
            }
        }
    }

    bool check = false;
    double prev_obj = INT_MAX;

    // Start training
    for (int iter = 0; iter < max_iter; iter++){
        int num_changed_alphas = 0;

        std::vector<double> gradient(X.size(), 0);
        
        // 3 parts of the objective function
        double obj_1 = 0;
        double obj_2 = 0;
        double obj_3 = 0;

        for (int i = 0; i < X.size(); i++){
            double grad_i = 1;
            double obj_2_i = 0;
            obj_1 += alpha[i];

            for (int j = 0; j < X.size(); j++){

                // Gradient
                grad_i -= alpha[j] * y[i] * y[j] * kernel_matrix[j][i];
                grad_i -= beta * alpha[j] * y[i] * y[j] * kernel_matrix[j][i];

                //Local objective function update
                obj_2_i -= grad_i;
            }
            
            obj_2_i *= 1/2 * alpha[i];

            //Global objective function update
            obj_2 += obj_2_i;
            obj_3 += alpha[i] * y[i];            

            // Store the gradient
            gradient[i] = grad_i;
            
        }
        // Update Beta

        double grad_B = 0;
        for (int i = 0; i< X.size();i++){
            grad_B += alpha[i] * y[i];
        }

        beta += 1/2 * grad_B * grad_B;

        // Update the weights
        for (int i = 0; i < X.size(); i++){
            alpha[i] = alpha[i] + learning_rate * gradient[i];

            // Clip the weights to be within the range (0, C)
            if (alpha[i] < 0){
                alpha[i] = 0;
            }
            else if (alpha[i] > C){
                alpha[i] = C;
            }
        }

        // Check if the error is less than the tolerance
        if (std::abs(obj_1 + obj_2 + obj_3 - prev_obj) < this->_tol){
            check = true;
        }
        prev_obj = obj_1 + obj_2 + obj_3;

        if (check){
            break;
        }
    }

    this->_alpha = alpha;
    this->_beta = beta;

    //Store weight if using linear kernel (only possible for linear kernel)
    if (kernel == "linear"){
        for (int i = 0; i < X.size(); i++){
            for (int j = 0; j < X[0].size(); j++){
                weights[j] += alpha[i] * y[i] * X[i][j];
            }
        }
    }

    // Get the bias term
    // bias = sum(y - sum(alpha * y * kernel(x, xi))) / N where alpha > 0 and alpha < C
    double _count = 0;
    for (int i = 0; i < X.size(); i++){
        if (alpha[i] > 0 && alpha[i] < C){
            _count ++;
            coef0 += y[i];
            for (int j = 0; j < X.size(); j++){
                coef0 -= alpha[j] * y[j] * kernel_matrix[j][i];
            }
        }
    }
    coef0 /= _count;
    this->_coef0 = coef0;

};

std::vector<double> SVC::predict(std::vector<std::vector<double>>& X){
    std::vector<double> y_pred;
    for (int i = 0; i < X.size(); i++){
        double sum = 0;

        // f(xk) = wT * xk + b
        if (this->_kernel == "linear"){
            for (int j = 0; j < this->weights.size(); j++){
                sum += this->weights[j] * X[i][j];
            }
        }

        // Dual form
        // f(xk) = sum(alpha * y * kernel(xk, xi)) + coef0
        else if (this->_kernel == "polynomial"){
            for (int j = 0; j < this->_X.size(); j++){
                sum += this->_alpha[j] * this->_y[j] * __polynomial_kernel(this->_X[j], X[i]);
            }
        }
        else if (this->_kernel == "rbf"){
            for (int j = 0; j < this->_X.size(); j++){
                sum += this->_alpha[j] * this->_y[j] * __rbf_kernel(this->_X[j], X[i]);
            }
        }

        sum += this->_coef0;

        // Predict the class
        if (sum >= 0){
            y_pred.push_back(1);
        }
        else{
            y_pred.push_back(-1);
        }
    }
    return y_pred;
}

std::vector<double> SVC::predict_proba(std::vector<std::vector<double>>& X){
    return this->predict(X);
}