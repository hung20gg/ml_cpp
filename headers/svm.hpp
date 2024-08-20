#include "basemodel.hpp"
#include <vector>
#include <iostream>

// Kernel functions

class SVC : public BaseModel{

    private:
        double _learning_rate;
        double _beta;
        std::string _kernel;
        double _gamma;
        int _degree;
        double _coef0;
        double _tol;
        int _max_iter;
        double _C;


        double __linear_kernel(std::vector<double> x1, std::vector<double> x2);
        double __polynomial_kernel(std::vector<double> x1, std::vector<double> x2, int degree = 3);
        double __rbf_kernel(std::vector<double> x1, std::vector<double> x2, double gamma = 0.1);


    public:
        SVC(double learning_rate = 0.01, double beta = 0.9, std::string kernel = "rbf", double gamma = 0.1, int degree = 3, double coef0 = 0.0, double tol = 0.0001, int max_iter = 1000, double C = 1.0): BaseModel(){
            
            this->_learning_rate = learning_rate;
            this->_beta = beta;
            this->_kernel = kernel;
            this->_gamma = gamma;
            this->_degree = degree;
            this->_coef0 = coef0;
            this->_tol = tol;
            this->_max_iter = max_iter;
            this->_C = C;

            this->name = "Support Vector Classifier";
            this->description = "Support Vector Classifier is a supervised machine learning model that uses classification algorithms for two-group classification problems.";
            this->type = "classifier";
        };

        void fit(std::vector<std::vector<double>> X, std::vector<double> y) override;
        std::vector<double> predict(std::vector<std::vector<double>> X) override;
        std::vector<double> predict_proba(std::vector<std::vector<double>> X) override;

};