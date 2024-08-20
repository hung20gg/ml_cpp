#include "../headers/decisiontree.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

// Implementing the DecisionTree class
// https://github.com/tiepvupsu/DecisionTreeID3/blob/master/id3.py


double DecisionTree::__entropy(std::vector<double>&y){
    std::map<double, int> value_count;
    for(int i = 0; i < y.size(); i++){
        value_count[y[i]]++;
    }

    double entropy = 0;
    for(auto const& [key, val] : value_count){
        double p = val / y.size();
        entropy += -p * log2(p);
    }

    return entropy;
}

double DecisionTree::__information_gain(std::vector<std::vector<double>>&y_splits, double entropy){
    double total_entropy = 0;
    for(int i = 0; i < y_splits.size(); i++){
        total_entropy += y_splits[i].size() / y_splits.size() * __entropy(y_splits[i]);
    }

    return entropy - total_entropy;

}

std::vector<CategoricalNode*> DecisionTree::_split(CategoricalNode* node, std::vector<std::vector<double>>& X, std::vector<double> &y){
    std::vector<CategoricalNode*> children;
    
    double best_gain = 0;
    int best_attribute = -1;
    std::vector<std::vector<int>> best_splits;

    // Get the subtable of the feature
    std::vector<int> feature_index = node->feature_index;

    // Getting the unique values of the feature
    std::map<double, std::vector<int>> value_indices;

    // Getting the unique values of the selected feature
    std::vector<double> values;

    // Getting the entropies of the selected feature
    std::vector<double> entropies;

    for(int i = 0; i < X[0].size(); i++){

        // Getting the unique values of the feature
        std::vector<double> unique_values;
        for (auto const& index : feature_index){
            value_indices[X[index][i]].push_back(index);
        }

        // Gini of this is the same as the parent, entropy = 0
        if (value_indices.size() == 1){
            continue;
        }

        std::vector<std::vector<double>> y_splits;
        std::vector<std::vector<int>> splits;

        int min_samples_split = this->_min_samples_split;
        for(auto const& [key, val] : value_indices){
            unique_values.push_back(key);
            std::vector<double> y_split;
            std::vector<int> split;
            for(auto const& index : val){
                y_split.push_back(y[index]);
                split.push_back(index);
                
            }

            min_samples_split = std::min<int>(min_samples_split, y_split.size());
            y_splits.push_back(y_split);
            splits.push_back(split);
        }

        // Not splitting if the min_samples_split is not met
        if (min_samples_split < this->_min_samples_split){
            continue;
        }

        double Hxs = 0;
        for(int i = 0; i < y_splits.size(); i++){
            double entropy = __entropy(y_splits[i]);
            Hxs += y_splits[i].size() / y_splits.size() * entropy;
            entropies.push_back(entropy);
        }

        double gain = node->entropy - Hxs;
        y_splits.clear();

        // Not splitting if the information gain is less than the min gain
        if (gain < this->_min_gain){
            continue;
        }

        // Getting the best split
        if (gain > best_gain){
            best_gain = gain;
            best_splits = splits;
            best_attribute = i;
            values = unique_values;
        }
        splits.clear();
    }
    value_indices.clear();

    // Splitting the node
    if (best_attribute != -1){
        
        node->set_property(best_attribute, values);

        for(int i = 0; i < best_splits.size(); i++){
            CategoricalNode child;
            child.feature_index = best_splits[i];
            child.split_index = best_attribute;
            child.entropy = entropies[i];
            node->add_child(&child); 
            children.push_back(&child);
        }
    }

    best_splits.clear();
    entropies.clear();
    return children;
}

void DecisionTree::__set_label(CategoricalNode* node){

    // Getting the most common value
    std::map<double, int> value_count;
    for(auto const& index : node->feature_index){
        value_count[index]++;
    }
    int max_count = 0;
    double max_value = -1;
    for(auto const& [key, val] : value_count){
        if (val > max_count){
            max_count = val;
            max_value = key;
        }
    }
    node->set_label(max_value);
}

void DecisionTree::fit(std::vector<std::vector<double>> X, std::vector<double> y){
    
    std::vector<int>root_index;
    for(int i = 0; i < X.size(); i++){
        root_index.push_back(i);
    }
    
    this->root.entropy = __entropy(y);
    this->root.feature_index = root_index;
    this->root.depth = 0;

    std::vector<CategoricalNode*> stack;
    stack.push_back(&this->root); 
    while (stack.size() > 0){
        CategoricalNode* node = stack.back();
        stack.pop_back();

        std::vector<CategoricalNode*> children = _split(node, X, y);

        // Depth first search, add the children to the stack
        for(int i = 0; i < children.size(); i++){
            stack.push_back(children[i]);
        }

        // If the node is a leaf node, set the label
        if (children.size() == 0){
            __set_label(node);
        }
    }

}

std::vector<double> DecisionTree::predict(std::vector<std::vector<double>> X){
    std::vector<double> y_pred;
    for(int i = 0; i < X.size(); i++){
        CategoricalNode* current_node = &this->root;
        while (current_node->children.size() > 0){
            current_node = current_node->next_node(X[i]);
        }
        y_pred.push_back(current_node->value);
    }
    return y_pred;
}

