#pragma once
#include "basemodel.hpp"
#include <iostream>
#include <vector>

class KNN: public SupervisedBasedModel{

    private:
        std::vector<std::vector<double>> _X_train;
        std::vector<double> _y_train;
        
        double __predict_one(std::vector<double> x);

    public:
        int n_neighbors;
        int num_classes;
        KNN(int n_neighbors): SupervisedBasedModel(){
            this->n_neighbors = n_neighbors;

            this->name = "KNN";
            this->description = "K-Nearest Neighbors";
            this->type = "classifier";
        };

        void fit(std::vector<std::vector<double>> &X, std::vector<double> &y) override;
        std::vector<double> predict(std::vector<std::vector<double>> &X) override;
        std::vector<double> predict_proba(std::vector<std::vector<double>> &X) override;
};