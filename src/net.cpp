//
// Created by mrtheamir on 5/17/19.
//

#include "net.h"
#include <iostream>
#include <utility>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>

/*
 * The ma2vec vec2mat functions are simple helper functions that are supposed to convert and Eigen::Matrix to Eigen::Vector
 * and vice versa depending on the needs of the network.
 */

Eigen::VectorXd mat2vec(const Eigen::MatrixXd& matrix){
    Eigen::VectorXd vector(matrix.rows() * matrix.cols());
    for (int i{}; i < int(matrix.rows()); i++) {
        for (int j{}; j < int(matrix.cols()); j++) {
            vector(i * matrix.cols() + j) = matrix(i, j);
        }
    }
    return vector;
}

Eigen::VectorXd mat2vec(const std::vector<Eigen::MatrixXd>& matrix){
    if(matrix.empty())
        return Eigen::VectorXd();
    Eigen::VectorXd vector(matrix.size() * matrix[0].rows()*matrix[0].cols());
    for (int k{}; k < int(matrix.size()); k++) {
        for (int i{}; i < int(matrix[0].rows()); i++) {
            for (int j{}; j < int(matrix[0].cols()); j++) {
                vector(k*matrix[0].cols()*matrix[0].rows() + i * matrix[0].cols() + j) =
                        matrix[k](i, j);
            }
        }
    }
    return vector;
}

Eigen::MatrixXd vec2mat(const Eigen::VectorXd& vector, int rows, int cols){
    if(vector.size() != rows*cols)
        throw std::invalid_argument("Vector dimensions mismatch with arguments");
    Eigen::MatrixXd matrix(rows, cols);
    for(int i{}; i<rows; i++){
        for(int j{}; j<cols; j++){
            matrix(i,j) = vector(i*cols + j);
        }
    }
    return matrix;
}

std::vector<Eigen::MatrixXd> vec2mat(const Eigen::VectorXd& vector, int matrices, int rows, int cols){
    if(matrices <= 0)
        throw std::invalid_argument("Negative number of matrices");
    if(vector.size() != rows*cols*matrices)
        throw std::invalid_argument("Vector dimensions mismatch with arguments");
    std::vector<Eigen::MatrixXd> matrix(matrices, Eigen::MatrixXd(rows,cols));
    for(int k{}; k<matrices; k++) {
        for (int i{}; i < rows; i++) {
            for (int j{}; j < cols; j++) {
                matrix[k](i, j) = vector(k * rows * cols + i * cols + j);
            }
        }
    }
    return matrix;
}

Net::Net(std::initializer_list<int> l) {
    for(int i : l)
        if(i <= 0)
            throw std::invalid_argument("Layer can't have negative neurons");

    for(std::initializer_list<int>::iterator it{l.begin()+1}; it < l.end(); it++){
        addFCLayer(*(it), *(it - 1));
    }
}

void Net::addFCLayer(int neurons, int inputs, const std::string &activation) {
    layers.push_back(std::make_shared<FCLayer>(inputs, neurons, activation));
}

void Net::addFCLayer(const Eigen::MatrixXd &weight, const Eigen::VectorXd &bias, const std::string &activation) {
   layers.push_back(std::make_shared<FCLayer>(weight, bias, activation));
}

void Net::addConvLayer(int rows, int cols, int width, int kernels, int padding, int stride) {
    layers.push_back(std::make_shared<ConvLayer>(rows, cols, width, kernels, padding, stride));
}

void Net::addConvLayer(const std::vector<std::vector<Eigen::MatrixXd>> &kernels, int padding, int stride) {
    layers.push_back(std::make_shared<ConvLayer>(kernels, padding, stride));
}

void Net::addMaxPoolLayer(int rows, int cols) {
    layers.push_back(std::make_shared<MaxPool>(rows, cols));
}

Eigen::VectorXd Net::predict(Eigen::VectorXd input){
    for(std::shared_ptr<Layer> &l : layers){
        input = l -> predict(input);
    }
    return input;
}

Eigen::VectorXd Net::predict(Eigen::MatrixXd inputimg){
    std::vector<Eigen::MatrixXd> input{std::move(inputimg)};
    int i{};
    for (; i < int(layers.size()); i++) {
        if(std::dynamic_pointer_cast<FCLayer>(layers[i]))
            break;
        input = layers[i] -> feedForward(input);
    }
    Eigen::VectorXd inputv = mat2vec(input);
    for (; i < int(layers.size()); i++) {
        inputv = layers[i] -> predict(inputv);
    }
    return inputv;
}

int Net::predictLabel(Eigen::VectorXd input){
    input = predict(input);
    int position;
    input.maxCoeff(&position);
    return position;
}

int Net::predictLabel(Eigen::MatrixXd input){
    Eigen::VectorXd inputv;
    inputv = predict(std::move(input));
    int position;
    inputv.maxCoeff(&position);
    return position;
}


void Net::train(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd>&
        targets, int epochs, int batchSize, double learningRate, double regularization, double
        momentum, bool earlyStopping){
    if(epochs <= 0)
        throw std::invalid_argument("Epoch can't be negative");
    if(inputs.size() != targets.size())
        throw std::invalid_argument("Number of inputs doesn't match numbers of targets");
    if (batchSize < 0 || batchSize > int(inputs.size()))
        throw std::invalid_argument("Batch size invalid");

    int split{static_cast<int>(double(inputs.size()) *
                               0.8)}; //Change this parameter depending on the training/validation set ratios
    std::cout << "Split: " << split << std::endl;
    std::vector<Eigen::VectorXd> inputsTraining(inputs.begin(), inputs.begin() + split);
    std::vector<Eigen::VectorXd> inputsValidation(inputs.begin() + split, inputs.end());
    std::vector<Eigen::VectorXd> targetsTraining(targets.begin(), targets.begin() + split);
    std::vector<Eigen::VectorXd> targetsValidation(targets.begin() + split, targets.end());

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, split-1);
    int stop{};
    double validationLossOld{std::numeric_limits<double>::infinity()};
    int batches{(int(inputsTraining.size())+batchSize-1)/batchSize};

    for(int epoch{}; epoch<epochs; epoch++) {
        double trainingLoss{loss(inputsTraining, targetsTraining)};
        double validationLoss{loss(inputsValidation, targetsValidation)};
        //Early stopping module
        stop++;
        if (earlyStopping && !(stop % 10)) { // Change this parameter if you want early stopping to be triggered earlier or later. Early stopping is triggered when the validation loss grows for a specified number of iterations.
            stop = 0;
            if(validationLossOld < validationLoss) {
                std::cout << "EARLY STOP" << std::endl;
                std::cout << "Epoch: " << epoch << " Training Loss: " << trainingLoss << " Validation Loss: " << validationLoss << std::endl;
                return;
            }
            else{
                validationLossOld = validationLoss;
            }
        }

        //Logger
        if(!(epoch%1)) {
            std::cout << "Epoch: " << epoch << " Training Loss: " << trainingLoss
                      << " Validation Loss: " << validationLoss << std::endl;
        }
        for(int i{}; i < batches; i++) {
            for(std::shared_ptr<Layer> &l : layers) {
                l->resetDerivative();
            }
            for(int batch{}; batch<batchSize; batch++) {
                int sample = dist(rng);
                Eigen::VectorXd input = inputsTraining[sample];
                for (std::shared_ptr<Layer> &l : layers) {
                    input = l->feedForward(input);
                }
                input = 2 * (input - targetsTraining[sample]); //loss function
                for (int j{int(layers.size()) - 1}; j >= 0; j--) {
                    input = layers[j]->feedBackward(input);
                    layers[j]->accumulateDerivative();
                }
            }
            for (std::shared_ptr<Layer> &l : layers) {
                l->derive(batchSize, learningRate, regularization, momentum);
            }
        }
    }
}

void Net::train(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::VectorXd>&
        targets, int epochs, int batchSize, double learningRate, double regularization, double
        momentum, bool earlyStopping){
    if(epochs <= 0)
        throw std::invalid_argument("Epoch can't be negative");
    if(inputs.size() != targets.size())
        throw std::invalid_argument("Number of inputs doesn't match numbers of targets");
    if (batchSize < 0 || batchSize > int(inputs.size()))
        throw std::invalid_argument("Batch size invalid");

    int split{static_cast<int>(double(inputs.size()) * 0.8)}; // Change this parameter depending on the training/validation set ratios
    std::cout << "Split: " << split << std::endl;
    std::vector<Eigen::MatrixXd> inputsTraining(inputs.begin(), inputs.begin() + split);
    std::vector<Eigen::MatrixXd> inputsValidation(inputs.begin() + split, inputs.end());
    std::vector<Eigen::VectorXd> targetsTraining(targets.begin(), targets.begin() + split);
    std::vector<Eigen::VectorXd> targetsValidation(targets.begin() + split, targets.end());

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, split-1);
    int stop{};
    double validationLossOld{std::numeric_limits<double>::infinity()};
    int batches{(int(inputsTraining.size())+batchSize-1)/batchSize};

    for(int epoch{}; epoch<epochs; epoch++) {
        double trainingLoss{loss(inputsTraining, targetsTraining)};
        double validationLoss{loss(inputsValidation, targetsValidation)};
        //Early stopping module
        stop++;
        if (earlyStopping && !(stop % 20)) { // Change this parameter if you want early stopping to be triggered earlier or later. Early stopping is triggered when the validation loss grows for a specified number of iterations.
            stop = 0;
            if(validationLossOld < validationLoss) {
                std::cout << "EARLY STOP" << std::endl;
                std::cout << "Epoch: " << epoch << " Training Loss: " << trainingLoss << " Validation Loss: " << validationLoss << std::endl;
                return;
            }
            else{
                validationLossOld = validationLoss;
            }
        }

        //Logger
        if(!(epoch%1)) {
            std::cout << "Epoch: " << epoch << " Training Loss: " << trainingLoss
                      << " Validation Loss: " << validationLoss << std::endl;
        }
        for(int i{}; i < batches; i++) {
            for(std::shared_ptr<Layer> &l : layers) {
                l->resetDerivative();
            }
            for(int batch{}; batch<batchSize; batch++) {
                int sample = dist(rng);
                std::vector<Eigen::MatrixXd> input{inputsTraining[sample]};

                //FORWARD
                int layer{};
                for (; layer < int(layers.size()); layer++) {
                    if(std::dynamic_pointer_cast<FCLayer>(layers[layer]))
                        break;
                    input = layers[layer] -> feedForward(input);
                }
                int rows(input[0].rows()), cols(input[0].cols());
                int matrices = input.size(); // for later conversion
                Eigen::VectorXd inputv = mat2vec(input);
                for (; layer < int(layers.size()); layer++) {
                    inputv = layers[layer] -> feedForward(inputv);
                }

                //BACKWARD
                inputv = 2 * (inputv - targetsTraining[sample]); //loss function
                layer = int(layers.size()) - 1;
                for (; layer >= 0; layer--) {
                    if(std::dynamic_pointer_cast<ConvLayer>(layers[layer]) || std::dynamic_pointer_cast<MaxPool>(layers[layer]))
                        break;
                    inputv = layers[layer]->feedBackward(inputv);
                    layers[layer]->accumulateDerivative();
                }
                input = vec2mat(inputv, matrices, rows, cols);
                for (; layer >= 0; layer--) {
                    input = layers[layer]->feedBackward(input);
                    layers[layer]->accumulateDerivative();
                }
            }
            for (std::shared_ptr<Layer> &l : layers) {
                l->derive(batchSize, learningRate, regularization, momentum);
            }
        }
    }
}

/*
 * The loss functions currently simply sum the mean square error of the predictions of the network. This can be adjusted
 * however the user desires.
 */

double Net::loss(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targets){
    double sum = 0;
    for (int i{}; i < int(inputs.size()); i++) {
        Eigen::VectorXd output{predict(inputs[i])};
        sum += (output - targets[i]).transpose() * (output - targets[i]);
    }
    return sum / inputs.size();
}

double Net::loss(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::VectorXd>
        &targets){
    double sum = 0;
    for (int i{}; i < int(inputs.size()); i++) {
        Eigen::VectorXd output{predict(inputs[i])};
        sum += (output-targets[i]).transpose()*(output-targets[i]);
    }
    return sum/inputs.size();
}

void Net::save(const std::string& path) {
    std::ofstream config(path + "/config");
    config << layers.size() << " ";
    for (int i{}; i < int(layers.size()); i++) {
        layers[i] -> save(config, path + "/Layer" + std::to_string(i));
    }
}

void Net::load(const std::string& path) {
    layers.resize(0);
    std::ifstream config(path + "/config");
    int size;
    char type;
    config >> size;
    for(int i{}; i<size; i++){
        config >> type;
        switch(type){
            case 'f':
                layers.push_back(std::make_shared<FCLayer>());
                layers[i] -> load(config, path + "/Layer" + std::to_string(i));
                break;
            case 'c':
                layers.push_back(std::make_shared<ConvLayer>());
                layers[i] -> load(config, path + "/Layer" + std::to_string(i));
                break;
            case 'm':
                layers.push_back(std::make_shared<MaxPool>());
                layers[i] -> load(config, "");
                break;
            default:
                throw std::invalid_argument("Invalid type collected");
        }
    }
}
