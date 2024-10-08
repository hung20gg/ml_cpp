#pragma once
#include "basemodel.hpp"
#include <iostream>
#include <vector>


class SGDClassifier : public SupervisedBasedModel {

    private:
        double _learning_rate;
        std::string _regularization;
        double _lambda;
        int _max_iter;
        double _tol;
        int _batch_size;
        std::string _loss;

    public:

        SGDClassifier(double learning_rate = 0.01, int batch_size = 32, std::string regularization = "L1", double lambda = 0.001, int max_iter = 1000, double tol = 0.00001, std::string loss = "softmax"): SupervisedBasedModel(){
            
            this->_learning_rate = learning_rate;
            this->_regularization = regularization;
            this->_lambda = lambda;
            this->_max_iter = max_iter;
            this->_tol = tol;
            this->_loss = loss;
            this->_batch_size = batch_size;

            // this->name = "Logistic Regression";
            // this->description = "Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability.";
            // this->type = "classifier";
        };

        void fit(std::vector<std::vector<double>>& X, std::vector<double>& y) override;
        std::vector<double> predict(std::vector<std::vector<double>>& X) override;
        std::vector<double> predict_proba(std::vector<std::vector<double>>& X) override;
};