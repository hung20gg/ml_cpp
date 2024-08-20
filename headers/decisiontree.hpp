#pragma once
#include "basemodel.hpp"
#include <iostream>
#include <vector>
#include <map>


struct CategoricalNode{
    std::vector<int> feature_index;
    double value = -1;
    int split_index = -1;
    double entropy;
    int depth;

    std::vector<double> unique_values;
    std::vector<CategoricalNode*> children;

    void set_label(double value){
        this->value = value;
    };

    void set_property(int split_index, std::vector<double>& unique_values){
        this->split_index = split_index;
        this->unique_values = unique_values;
    };

    void add_child(CategoricalNode *child){
        this->children.push_back(child);
    };

    CategoricalNode* next_node(std::vector<double> x){
        for(int i = 0; i < this->unique_values.size(); i++){
            if(x[this->split_index] == this->unique_values[i]){
                return this->children[i];
            }
        }
        return nullptr;
    };

};



struct NumericalNode{
    std::vector<int> feature_index;
    int split_index = -1;
    double threshold;
    double value = -1;
    double entropy;
    int depth;

    std::vector<NumericalNode> children;
    NumericalNode* left;
    NumericalNode* right;

    void set_label(double value){
        this->value = value;
    };

    void set_threshold(double threshold){
        this->threshold = threshold;
    };

    void set_left(NumericalNode* left){
        this->left = left;
    };

    void set_right(NumericalNode* right){
        this->right = right;
    };

    NumericalNode* next_node(std::vector<double> x){
        if(x[this->split_index] <= this->threshold){
            return this->left;
        }else{
            return this->right;
        };
    };

};


// Implement Categorical first
class DecisionTree : TreeBasedModel{

    private:
        int _max_depth;
        int _min_samples_split;
        int _min_samples_leaf;
        double _min_gain;
        std::string _criterion;
        std::string _max_features;

        std::vector<CategoricalNode*> __split(CategoricalNode* node, std::vector<std::vector<double>>& X, std::vector<double> &y);

        
        double __entropy(std::vector<double>&y);
        double __information_gain(std::vector<std::vector<double>>&y, double entropy);
        void __set_label(CategoricalNode* node);
    public:
        DecisionTree(int max_depth = 20, int min_samples_split = 2, int min_samples_leaf = 1, double min_gain = 0.02, std::string criterion = "gini", std::string max_features = "auto"): TreeBasedModel(){
            
            this->_max_depth = max_depth;
            this->_min_samples_split = min_samples_split;
            this->_min_samples_leaf = min_samples_leaf;
            this->_criterion = criterion;
            this->_max_features = max_features;
            this->_min_gain = min_gain;

            this->name = "Decision Tree";
            this->description = "Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.";
            this->type = "classifier";
        };
        CategoricalNode* root;

        void fit(std::vector<std::vector<double>> X, std::vector<double> y) override;
        std::vector<double> predict(std::vector<std::vector<double>> X) override;
        
};