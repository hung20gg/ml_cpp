#pragma once
#include <iostream>
#include <vector>


class BaseModel {

    public:
        std::string name;
        std::string description;
        std::string type;
        std::vector<double> weights;

        virtual void fit(std::vector<std::vector<double>> X, std::vector<double> y) = 0;
        
        // For binary classification only
        virtual std::vector<double> predict(std::vector<std::vector<double>> X) = 0;
        virtual std::vector<double> predict_proba(std::vector<std::vector<double>> X) = 0;
        virtual void save(std::string filename) = 0;
};

class TreeBasedModel {
    
        public:
            std::string name;
            std::string description;
            std::string type;
            std::vector<double> weights;
    
            virtual void fit(std::vector<std::vector<double>> X, std::vector<double> y) = 0;
            virtual std::vector<double> predict(std::vector<std::vector<double>> X);
            virtual std::vector<double> predict_proba(std::vector<std::vector<double>> X);
    
            // void fit(std::vector<std::vector<double>> X, std::vector<double> y);
            // void predict();
            // void predict_proba();
};

class UnsuperivedBasedModel {

        public:
            std::string name;
            std::string description;
            std::string type;

            virtual void fit(std::vector<std::vector<double>> X) = 0;
            virtual std::vector<double> predict(std::vector<std::vector<double>> X) = 0;
            virtual std::vector<double> predict_proba(std::vector<std::vector<double>> X) = 0;
};