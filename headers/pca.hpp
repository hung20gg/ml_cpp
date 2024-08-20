#pragma once 
#include "basemodel.hpp"
#include <iostream>
#include <vector>
#include <cmath>

class PCA : public SupervisedBasedModel{

    private:
        std::vector<std::vector<double>> _eigenvectors;
        std::vector<double> _eigenvalues;

};