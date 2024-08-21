#pragma once
#include <iostream>
#include <vector>
#include <memory>


class LossFunction {
    public:
        virtual double forward(std::vector<double>& y_true, std::vector<double>& y_pred) const = 0;
        virtual std::vector<double> backward(std::vector<double>& y_true, std::vector<double>& y_pred) const = 0;
};

// LossFunction* createLossFunction(const std::string& loss_type);

class Regularization {
    public:
        virtual double forward(std::vector<double>& weights) const = 0;
        virtual std::vector<double> backward(std::vector<double>& weights) const = 0;
};

class MeanSquaredError : public LossFunction {
    public:
        double forward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
        std::vector<double> backward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
};

class MeanAbsoluteError : public LossFunction {
    public:
        double forward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
        std::vector<double> backward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
};

class BinaryCrossEntropy : public LossFunction {
    public:
        double forward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
        std::vector<double> backward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
};

class CrossEntropyWithSoftmax : public LossFunction {
    public:
        double forward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
        double forward(std::vector<std::vector<int>>& y_true, std::vector<std::vector<double>>& y_pred);
        double forward(std::vector<int>& y_true, std::vector<std::vector<double>>& y_pred);

        std::vector<double> backward(std::vector<std::vector<int>>& y_true, std::vector<std::vector<double>>& y_pred);
        std::vector<double> backward(std::vector<int>& y_true, std::vector<std::vector<double>>& y_pred);
        std::vector<double> backward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
};

class Hinge : public LossFunction {
    public:
        double forward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
        std::vector<double> backward(std::vector<double>& y_true, std::vector<double>& y_pred) const override;
};


class L1Loss : public Regularization {
    public:
        double forward(std::vector<double>& weights) const override;
        std::vector<double> backward(std::vector<double>& weights) const override;
};

class L2Loss : public Regularization {
    public:
        double forward(std::vector<double>& weights) const override;
        std::vector<double> backward(std::vector<double>& weights) const override;
};

LossFunction* createLossFunction(const std::string& loss_type){
    if (loss_type == "mse"){
        return new MeanSquaredError();
    }else if (loss_type == "mae"){
        return new MeanAbsoluteError();
    }else if (loss_type == "binary"){
        return new BinaryCrossEntropy();
    }else if (loss_type == "hinge"){
        return new Hinge();
    }else if (loss_type == "crossentropy"){
        return new CrossEntropyWithSoftmax();
    }else{
        throw std::invalid_argument("Invalid loss function");
    }
};

Regularization* createRegularization(const std::string& regularization_type){
    if (regularization_type == "L1"){
        return new L1Loss();
    }else if (regularization_type == "L2"){
        return new L2Loss();
    }else{
        throw std::invalid_argument("Invalid regularization function");
    }
};