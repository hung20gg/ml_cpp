#include <iostream>
#include <vector>
#include <cmath>

double dot(const std::vector<double> &a, const std::vector<double> &b);

std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b);

std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b);

std::vector<double> operator*(const std::vector<double> &a, const std::vector<double> &b);

std::vector<std::vector<double>> matrix_mul(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

double L1_regularization(std::vector<double> &weights);

double L2_regularization(std::vector<double> &weights);

std::vector<size_t> argsort(const std::vector<double>& v);

double distance(std::vector<double> &x1, std::vector<double> &x2);

double mean_squared_error(std::vector<double>& y_true, std::vector<double>& y_pred);

double mean_absolute_error(std::vector<double>& y_true, std::vector<double>& y_pred);

double binary_crossentropy(std::vector<double>& y_true, std::vector<double>& y_pred);

double cross_entropy(std::vector<std::vector<int>> &y_true, std::vector<std::vector<double>> &y_pred);

double cross_entropy(std::vector<int> &y_true, std::vector<std::vector<double>> &y_pred);

double entropy(std::vector<double> &y_true, std::vector<double> &y_pred);

double sigmoid(double x);

std::vector<double> sigmoid(std::vector<double> &x);

std::vector<double> softmax(std::vector<double> &x);

std::vector<int> pickKRandomInRangeN(int k, int n);

std::vector<std::vector<double>> slice2D(const std::vector<std::vector<double>>& x, int start ,int batch_size);

std::vector<double> slice1D(const std::vector<double>& x, int start ,int batch_size);
