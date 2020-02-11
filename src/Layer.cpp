#include <utility>
#include <fstream>
#include <vector>
#include <iostream>
#include "Layer.h"

//
// Created by mrtheamir on 6/15/19.
//

/*************************************/
/** FULLY CONNECTED LAYER INTERFACE **/
/*************************************/

FCLayer::FCLayer(int input, int output, const std::string& activation){
    srand((unsigned int) time(nullptr));
    weight = Eigen::MatrixXd::Random(output, input)*1.5/output;
    bias = Eigen::VectorXd::Ones(output) * 0.1;
    setActivation(activation);
    weightMomentum = Eigen::MatrixXd::Zero(weight.rows(), weight.cols());
    biasMomentum = Eigen::VectorXd::Zero(bias.rows());
}

FCLayer::FCLayer(Eigen::MatrixXd weight, Eigen::VectorXd bias, const std::string& activation){
    this -> weight = std::move(weight);
    this -> bias = std::move(bias);
    setActivation(activation);
    weightMomentum = Eigen::MatrixXd::Zero(this -> weight.rows(), this -> weight.cols());
    biasMomentum = Eigen::VectorXd::Zero(this -> bias.rows());
}

Eigen::VectorXd FCLayer::feedForward(Eigen::VectorXd input) {
    x = input;
    input = weight*input + bias;
    z = Eigen::VectorXd(input.size());
    std::transform(input.begin(), input.end(), z.begin(), activationDerivative);
    std::transform(input.begin(), input.end(), input.begin(), activationFunction);
    return input;
}

Eigen::VectorXd FCLayer::feedBackward(Eigen::VectorXd input) {
    input = z.cwiseProduct(input);
    biasDerivative = input;
    weightDerivative = input*x.transpose();
    return  weight.transpose() * input;
}

Eigen::VectorXd FCLayer::predict(Eigen::VectorXd input) const {
    // This validation is not necessary but it helps when you miscalculate the number of nodes.
    if(input.size() != weight.cols())
        throw std::invalid_argument("Input and weight not compatible. Expected " + std::to_string(input.rows()));
    input = weight*input + bias;
    std::transform(input.begin(), input.end(), input.begin(), activationFunction);
    return input;
}

/*
 * The setActivation function is extremely practical because it allows the user to add or modify activation functions.
 * This simple function stores all the activation functions that you can use and if you want to add a new one, simply
 * specify its name in the list of names and add it as a lambda function to the vector below. Also don't forget to
 * define the derivative of the function and that is all. The function also covers validation, so in case that you input
 * a wrong function name, you will get an exception.
 */
void FCLayer::setActivation(const std::string &t_activation) {
    this->activation = t_activation;
    std::vector<std::string> names{"Sigmoid", "ReLU", "Hyperbolic", "Leaky_ReLU", "Linear", "SoftPlus", "Step"};

    std::vector<std::function<double(const double)>> afun {
            [](const double x_) { //Sigmoid
                return 1 / (1 + exp(-x_));
            },
            [](const double x_) { //ReLU
                return std::max(0.0, x_);
            },
            [](const double x_) { //Hyperbolic
                return tanh(x_);
            },
            [](const double x_) { //Leaky ReLU
                return std::max(0.1 * x_, x_);
            },
            [](const double x_) { //Linear
                return x_;
            },
            [](const double x_) { //SoftPlus
                return log(1 + exp(x_));
            },
            [](const double x_) { //Step
                return double(x_ >= 0);
            }
    };
    std::vector<std::function<double(const double)>> afunderivative {
            [](const double x_) { // Sigmoid
                return 1 / (1 + exp(-x_)) * (1 - 1 / (1 + exp(-x_)));
            },
            [](const double x_) { //ReLU
                return x_ >= 0;
            },
            [](const double x_) { //Hyperbolic
                return 1 - pow(tanh(x_), 2);
            },
            [](const double x_) { //Leaky ReLU
                return (x_ < 0) * 0.1 + (x_ >= 0);
            },
            [](const double) { //Linear
                return 1;
            },
            [](const double x_) { //SoftPlus
                return 1 / (1 + exp(-x_));
            },
            [](const double x_) { //Step
                return double(bool(x_) * std::numeric_limits<double>::infinity());
            }
    };
    auto it = std::find(names.begin(), names.end(), t_activation);
    if(it == names.end())
        throw std::invalid_argument("Activation function not found");
    int n = (distance(names.begin(), it));
    activationFunction = afun[n];
    activationDerivative = afunderivative[n];
}

void FCLayer::resetDerivative(){
    weightAccumulation = Eigen::MatrixXd::Zero(weight.rows(), weight.cols());
    biasAccumulation = Eigen::VectorXd::Zero(bias.rows());
}

void FCLayer::accumulateDerivative(){
    weightAccumulation += weightDerivative;
    biasAccumulation += biasDerivative;
}

void FCLayer::derive(int batchSize, double learningRate, double regularization, double momentum){
    /*
     * Momentum formula:
     * vk+1 = momentum ∗ v + ∇L
     * wn = wn−1 - α ∗ vk+1
     */
    weightMomentum = momentum * weightMomentum + weightAccumulation;
    biasMomentum = momentum * biasMomentum + biasAccumulation;
    weight -= learningRate * weightMomentum / batchSize + regularization * weight;
    bias -= learningRate * biasMomentum / batchSize + regularization * bias;
}

void FCLayer::save(std::ofstream& config, const std::string &path) const {
    config << "f " << weight.rows() << " " << weight.cols() << " ";
    std::ofstream write(path + "weight");
    write << weight;
    write.close();
    write.open(path + "bias");
    write << bias;
    write.close();
    write.open(path + "function");
    write << activation;
}

void FCLayer::load(std::ifstream& config, const std::string &path) {
    int rows, cols;
    config >> rows;
    config >> cols;
    weight = Eigen::MatrixXd(rows, cols);
    bias = Eigen::VectorXd(rows);
    std::ifstream read(path + "weight");
    for(int j{}; j<rows; j++){
        for(int k{}; k<cols; k++){
            read >> weight(j,k);
        }
    }
    read.close();
    read.open(path + "bias");
    for(int j{}; j<rows; j++){
        read >> bias(j);
    }
    read.close();
    read.open(path + "function");
    read >> activation;
    setActivation(activation);
    weightMomentum = Eigen::MatrixXd::Zero(weight.rows(), weight.cols());
    biasMomentum = Eigen::VectorXd::Zero(bias.rows());
}

std::vector<Eigen::MatrixXd> FCLayer::feedForward(std::vector<Eigen::MatrixXd>) {
    throw std::logic_error("Matrix input passed to fully connected layer");
}

std::vector<Eigen::MatrixXd> FCLayer::feedBackward(std::vector<Eigen::MatrixXd>) {
    throw std::logic_error("Matrix input passed to fully connected layer");
}

Eigen::MatrixXd FCLayer::predict(Eigen::MatrixXd) const {
    throw std::logic_error("Matrix input passed to fully connected layer");
}

/**********************************
** CONVOLUTIONAL LAYER INTERFACE **
**********************************/

Eigen::MatrixXd convolution(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel, int stride){
    if(input.rows() < kernel.rows() || input.cols() < kernel.cols())
        throw std::invalid_argument("Kernel bigger than input");
    if(stride < 1)
        throw std::invalid_argument("Stride can't be negative");

    double rows = double(input.rows() - kernel.rows())/stride + 1;
    double cols = double(input.cols() - kernel.cols())/stride + 1;
    if(rows != int(rows) || cols != int(cols))
        throw std::invalid_argument("Invalid stride");
    Eigen::MatrixXd output((int)rows, (int)cols);
    for(int i{}; i<=rows-1; i+=stride){
        for(int j{}; j<=cols-1; j+=stride){
            output(i,j) = (input.block(i, j, kernel.rows(), kernel.cols()).cwiseProduct(kernel)).sum();
        }
    }
    return output;
}

Eigen::MatrixXd matrixPadding(const Eigen::MatrixXd& matrix, int padding, double value){
    if(padding < 0)
        throw std::invalid_argument("Argument padding can't be negative");

    Eigen::MatrixXd result = Eigen::MatrixXd::Constant(matrix.rows() + 2 * padding, matrix.cols() + 2 * padding, value);
    result.block(padding, padding, matrix.rows(), matrix.cols()) = matrix;
    return result;
}

ConvLayer::ConvLayer(int rows, int cols, int width, int kernels, int padding, int stride) :
    rows(rows),cols(cols), width(width), padding(padding), stride(stride){
    if(rows <= 0 || cols <= 0 || width<=0 || kernels <= 0 || padding < 0 || stride <= 0)
        throw std::invalid_argument("Passed arguments must be positive");
    this -> kernels = std::vector<std::vector<Eigen::MatrixXd>> (kernels,
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Random(rows, cols)*1.5/(rows*cols)));
    kernelDerivative = std::vector<std::vector<Eigen::MatrixXd>>(kernels,
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Zero(rows, cols)));
    kernelAccumulation = std::vector<std::vector<Eigen::MatrixXd>>(kernels,
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Zero(rows, cols)));
    kernelMomentum = std::vector<std::vector<Eigen::MatrixXd>>(kernels,
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Zero(rows, cols)));
}

ConvLayer::ConvLayer(std::vector<std::vector<Eigen::MatrixXd>> kernels, int padding, int stride) :
    padding(padding), stride(stride){

    if(kernels.empty())
        throw std::invalid_argument("No kernels passed");
    if(kernels[0].empty())
        throw std::invalid_argument("Kernel has no width");

    rows = kernels[0][0].rows();
    cols = kernels[0][0].cols();
    width = kernels[0].size();

    if(!rows || !cols || padding < 0 || stride <= 0)
        throw std::invalid_argument("Passed arguments must be positive");

    for(const std::vector<Eigen::MatrixXd> &v : kernels) {
        if (int(v.size()) != width)
            throw std::invalid_argument("Kernels inconsistent in size");
        for(const Eigen::MatrixXd &mat : v)
            if(mat.rows() != rows || mat.cols() != cols)
                throw std::invalid_argument("Matrices inside kernel have inconsistent sizes");
    }

    this -> kernels = kernels;
    kernelDerivative = std::vector<std::vector<Eigen::MatrixXd>>(kernels.size(),
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Zero(rows, cols)));
    kernelAccumulation = std::vector<std::vector<Eigen::MatrixXd>>(kernels.size(),
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Zero(rows, cols)));
    kernelMomentum = std::vector<std::vector<Eigen::MatrixXd>>(kernels.size(),
            std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd::Zero(rows, cols)));
}

std::vector<Eigen::MatrixXd> ConvLayer::feedForward(std::vector<Eigen::MatrixXd> input){
    if(int(input.size()) != width)
        throw std::invalid_argument("Mismatching kernel and input width");
    std::transform(input.begin(), input.end(), input.begin(), [this](const Eigen::MatrixXd& m){
        return matrixPadding(m, padding);
    });
    x = input;
    Eigen::MatrixXd image;
    std::vector<Eigen::MatrixXd> results;

    double _rows = double(input[0].rows() - kernels[0][0].rows())/stride + 1;
    double _cols = double(input[0].cols() - kernels[0][0].cols())/stride + 1;
    if(_rows != int(_rows) || _cols != int(_cols))
        throw std::invalid_argument("Invalid input, kernel or stride");

    for(const std::vector<Eigen::MatrixXd>& kernel : kernels){
        image = Eigen::MatrixXd::Zero((int)_rows, (int)_cols);
        for(int i{}; i<int(kernel.size()); i++){
            image += convolution(input[i], kernel[i], stride);
        }
        results.push_back(image);
    }
    return results;
}

std::vector<Eigen::MatrixXd> ConvLayer::feedBackward(std::vector<Eigen::MatrixXd> input){
    for(int i{}; i<int(input.size()); i++){
        for(int j{}; j<int(x.size()); j++){
            kernelDerivative[i][j] = convolution(x[j], input[i], stride);
        }
    }

    std::transform(input.begin(), input.end(), input.begin(), [this](Eigen::MatrixXd& m){
        return matrixPadding(m, int(kernels[0][0].rows() - 1));
    });

    std::vector<Eigen::MatrixXd> output(x.size(), Eigen::MatrixXd::Zero(x[0].rows(), x[0].cols()));

    for (int i{}; i < int(output.size()); i++) {
        for (int j{}; j < int(kernels.size()); j++) {
            output[i] += convolution(input[j], kernels[j][i].reverse(), stride);
        }
        output[i] = output[i].block(padding, padding, output[i].rows()-2*padding, output[i].cols()-2*padding);
        //output[i] /= kernels.size();
    }
    return output;
}

Eigen::MatrixXd ConvLayer::predict(Eigen::MatrixXd) const {
    throw std::logic_error("ConvLayer can't predict");
}

void ConvLayer::resetDerivative() {
    kernelAccumulation = std::vector<std::vector<Eigen::MatrixXd>>(kernels.size(), std::vector<Eigen::MatrixXd>
            (width, Eigen::MatrixXd::Zero(rows, cols)));
}

void ConvLayer::accumulateDerivative() {
    for (int i{}; i < int(kernels.size()); i++) {
        for(int j{}; j<width; j++){
            kernelAccumulation[i][j] += kernelDerivative[i][j];
        }
    }
}

void ConvLayer::derive(int batchSize, double learningRate, double regularization, double momentum) {
    for (int i{}; i < int(kernels.size()); i++) {
        for(int j{}; j<width; j++) {
            kernelMomentum[i][j] = momentum * kernelMomentum[i][j] + kernelAccumulation[i][j];
            kernels[i][j] -= learningRate * kernelMomentum[i][j] / batchSize + regularization * kernels[i][j];
        }
    }
}

void ConvLayer::save(std::ofstream &config, const std::string &path) const {
    config << "c " << rows << " " << cols << " " << width << " " << kernels.size() << " " << padding << " " << stride << " ";
    for (int i{}; i < int(kernels.size()); i++) {
        for(int j{}; j<width; j++) {
            std::ofstream write(path + "kernel" + std::to_string(i) + "_" + std::to_string(j));
            write << kernels[i][j];
        }
    }
}

void ConvLayer::load(std::ifstream &config, const std::string &path) {
    config >> rows;
    config >> cols;
    config >> width;
    int kernelSize;
    config >> kernelSize;
    config >> padding;
    config >> stride;
    kernels = std::vector<std::vector<Eigen::MatrixXd>>(kernelSize, std::vector<Eigen::MatrixXd>(width, Eigen::MatrixXd(rows,cols)));
    for(int i{}; i<kernelSize; i++){
        for(int j{}; j<width; j++) {
            std::ifstream read(path + "kernel" + std::to_string(i) + "_" + std::to_string(j));
            for (int m{}; m < rows; m++) {
                for (int n{}; n < cols; n++) {
                    read >> kernels[i][j](m, n);
                }
            }
        }
    }
    kernelDerivative = std::vector<std::vector<Eigen::MatrixXd>>(kernelSize, std::vector<Eigen::MatrixXd>
            (width, Eigen::MatrixXd::Zero(rows, cols)));
    kernelAccumulation = std::vector<std::vector<Eigen::MatrixXd>>(kernelSize, std::vector<Eigen::MatrixXd>
            (width, Eigen::MatrixXd::Zero(rows, cols)));
    kernelMomentum = std::vector<std::vector<Eigen::MatrixXd>>(kernelSize, std::vector<Eigen::MatrixXd>
            (width, Eigen::MatrixXd::Zero(rows, cols)));
}

Eigen::VectorXd ConvLayer::feedForward(Eigen::VectorXd) {
    throw std::logic_error("Vector input passed to convolutional layer");
}

Eigen::VectorXd ConvLayer::feedBackward(Eigen::VectorXd) {
    throw std::logic_error("Vector input passed to convolutional layer");
}

Eigen::VectorXd ConvLayer::predict(Eigen::VectorXd) const {
    throw std::logic_error("Vector input passed to convolutional layer");
}

/****************************
** MAXPOOL LAYER INTERFACE **
*****************************/

MaxPool::MaxPool(int rows, int cols) {
    if(rows <= 1 || cols <= 1)
        throw std::invalid_argument("Sizes must be bigger than 1");

    this -> rows = rows;
    this -> cols = cols;
}

std::vector<Eigen::MatrixXd> MaxPool::feedForward(std::vector<Eigen::MatrixXd> input) {
    double newRows = double(input[0].rows()) / rows;
    double newCols = double(input[0].cols()) / cols;
    if (newRows != int(newRows) || newCols != int(newCols))
        throw std::invalid_argument("Layer not compatible with pool mask");

    x = input;
    std::vector<Eigen::MatrixXd> output;
    for (int k{}; k < int(input.size()); k++) {
        output.emplace_back(Eigen::MatrixXd::Zero(newRows, newCols));
        for (int i{}; i < newRows; i++) {
            for (int j{}; j < newCols; j++) {
                output[k](i,j) = input[k].block(i * rows, j * cols, rows, cols).maxCoeff();
            }
        }
    }
    return output;
}

std::vector<Eigen::MatrixXd> MaxPool::feedBackward(std::vector<Eigen::MatrixXd> input) {
    std::vector<Eigen::MatrixXd> output;
    for (int k{}; k < int(input.size()); k++) {
        output.emplace_back(Eigen::MatrixXd::Zero(input[0].rows()*rows, input[0].cols()*cols));
        int maxRow{}, maxCol{};
        for (int i{}; i < int(input[k].rows()); i++) {
            for (int j{}; j < int(input[k].cols()); j++) {
                x[k].block(i*rows, j*cols, rows, cols).maxCoeff(&maxRow, &maxCol);
                output[k](i*rows + maxRow, j*cols + maxCol) = input[k](i,j);
            }
        }
    }
    return output;
}

Eigen::MatrixXd MaxPool::predict(Eigen::MatrixXd input) const {
    double newRows = double(input.rows())/rows;
    double newCols = double(input.cols())/cols;
    if(newRows != int(newRows) || newCols != int(newCols))
        throw std::invalid_argument("Layer not compatible with pool mask");

    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(newRows, newCols);
    for(int i{}; i<newRows; i++){
        for(int j{}; j<newCols; j++){
            output(i,j) = input.block(i*rows, j*cols, rows, cols).maxCoeff();
        }
    }
    return output;
}

void MaxPool::save(std::ofstream &config, const std::string &) const {
    config << "m " << rows << " " << cols << " ";
}

void MaxPool::load(std::ifstream &config, const std::string &) {
    config >> rows;
    config >> cols;
}

void MaxPool::resetDerivative(){}

void MaxPool::accumulateDerivative(){}

void MaxPool::derive(int, double, double, double) {}

Eigen::VectorXd MaxPool::feedForward(Eigen::VectorXd) {
    throw std::logic_error("Vector input passed to MaxPool layer");
}

Eigen::VectorXd MaxPool::feedBackward(Eigen::VectorXd) {
    throw std::logic_error("Vector input passed to MaxPool layer");
}

Eigen::VectorXd MaxPool::predict(Eigen::VectorXd) const {
    throw std::logic_error("Vector input passed to MaxPool layer");
}