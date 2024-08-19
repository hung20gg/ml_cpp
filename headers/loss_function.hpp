#pragma once
#include <iostream>
#include <vector>

class LossFunction {
    public:
        virtual double forward(std::vector<double> y_true, std::vector<double> y_pred) = 0;
        virtual std::vector<double> backward(std::vector<double> y_true, std::vector<double> y_pred) = 0;
};

class MeanSquaredError : public LossFunction {
    public:
        double forward(std::vector<double> y_true, std::vector<double> y_pred) override;
        std::vector<double> backward(std::vector<double> y_true, std::vector<double> y_pred) override;
};

class MeanAbsoluteError : public LossFunction {
    public:
        double forward(std::vector<double> y_true, std::vector<double> y_pred) override;
        std::vector<double> backward(std::vector<double> y_true, std::vector<double> y_pred) override;
};

class BinaryCrossEntropy : public LossFunction {
    public:
        double forward(std::vector<double> y_true, std::vector<double> y_pred) override;
        std::vector<double> backward(std::vector<double> y_true, std::vector<double> y_pred) override;
};

class CrossEntropyWithSoftmax : public LossFunction {
    public:
        double forward(std::vector<double> y_true, std::vector<double> y_pred) override;
        double forward(std::vector<std::vector<int>> y_true, std::vector<std::vector<double>> y_pred);
        double forward(std::vector<int> y_true, std::vector<std::vector<double>> y_pred);

        std::vector<double> backward(std::vector<std::vector<int>> y_true, std::vector<std::vector<double>> y_pred);
        std::vector<double> backward(std::vector<int> y_true, std::vector<std::vector<double>> y_pred);
        std::vector<double> backward(std::vector<double> y_true, std::vector<double> y_pred) override;
};

class Hinge : public LossFunction {
    public:
        double forward(std::vector<double> y_true, std::vector<double> y_pred) override;
        std::vector<double> backward(std::vector<double> y_true, std::vector<double> y_pred) override;
};