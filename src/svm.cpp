#include "../headers/svm.hpp"
#include "../headers/function.hpp"
#include <vector>
#include <cmath>

double SVC::__linear_kernel(std::vector<double> x1, std::vector<double> x2){
    return dot(x1, x2);
}

double SVC::__polynomial_kernel(std::vector<double> x1, std::vector<double> x2, int degree = 3){
    return pow(dot(x1, x2) + 1, degree);
}

double SVC::__rbf_kernel(std::vector<double> x1, std::vector<double> x2, double gamma = 0.1){
    return exp(-gamma * distance(x1, x2));
}

void SVC::fit(std::vector<std::vector<double>> X, std::vector<double> y){

    // Initialize weights
    std::vector<double> weights(X[0].size(), 0);
    std::vector<double> bias = {0};

    // Initialize learning rate
    double learning_rate = this->_learning_rate;

    // Initialize beta
    double beta = this->_beta;

    // Initialize kernel
    std::string kernel = this->_kernel;

    // Initialize gamma
    double gamma = this->_gamma;

    // Initialize degree
    int degree = this->_degree;

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
                kernel_matrix[i][j] = __polynomial_kernel(X[i], X[j], degree);
            }
            else if (kernel == "rbf"){
                kernel_matrix[i][j] = __rbf_kernel(X[i], X[j], gamma);
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

        // Update the weights
        for (int i = 0; i < X.size(); i++){
            alpha[i] = alpha[i] + learning_rate * gradient[i];
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

    //Store weight if using linear kernel





};

std::vector<double> SVC::predict(std::vector<std::vector<double>> X){

}