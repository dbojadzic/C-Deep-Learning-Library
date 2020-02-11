//
// Created by mrtheamir on 6/10/19.

#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <string>

/*
 * The mnist.h file is supposed to load the MNIST dataset of handwritten digits. You can change the parameters TRAINING
 * and TEST to define how many images you want to load from the dataset.
 */


int TRAINING = 2000;
int TEST = 2000;

std::vector<Eigen::VectorXd> loadMnistVector(const std::string& path = "mnist");
std::vector<Eigen::MatrixXd> loadMnistMatrix(const std::string& path = "mnist");

std::vector<int> loadLabels(const std::string &path = "mnist");

std::vector<Eigen::VectorXd> loadLabelsVector(const std::string &path = "mnist");

std::vector<Eigen::VectorXd> loadMnistTestVector(const std::string& path = "mnist");
std::vector<Eigen::MatrixXd> loadMnistTestMatrix(const std::string& path = "mnist");

std::vector<int> loadTestLabels(const std::string &path = "mnist");
std::vector<Eigen::VectorXd> loadTestLabelsVector(const std::string& path = "mnist");


std::vector<int> loadLabels(const std::string &path) {
    std::ifstream stream(path + "/train-labels.idx1-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<int> mnist;

    for(int k{}; k<TRAINING; k++) {
        number = 0;
        stream.read(reinterpret_cast<char*>(&number), sizeof(char));
        mnist.push_back(number);
    }
    return mnist;
}

std::vector<Eigen::VectorXd> loadMnistVector(const std::string &path) {
    std::ifstream stream(path + "/train-images.idx3-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<Eigen::VectorXd> mnist;

    Eigen::VectorXd image(28*28);
    for(int k{}; k<TRAINING; k++) {
        for (int i{}; i < 28*28; i++) {
            number = 0;
            stream.read(reinterpret_cast<char *>(&number), sizeof(char));
            image(i) = double(number);

        }
        mnist.push_back(image);
    }
    return mnist;
}

std::vector<Eigen::MatrixXd> loadMnistMatrix(const std::string &path) {
    std::ifstream stream(path + "/train-images.idx3-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<Eigen::MatrixXd> mnist;

    Eigen::MatrixXd image(28,28);
    for(int k{}; k<TRAINING; k++) {
        for (int i{}; i < 28; i++) {
            for (int j{}; j < 28; j++) {
                number = 0;
                stream.read(reinterpret_cast<char *>(&number), sizeof(char));
                image(i, j) = double(number);
            }
        }
        mnist.push_back(image);
    }
    return mnist;
}

std::vector<int> loadTestLabels(const std::string &path) {
    std::ifstream stream(path + "/t10k-labels.idx1-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<int> mnist;

    for(int k{}; k<TEST; k++) {
        number = 0;
        stream.read(reinterpret_cast<char *>(&number), sizeof(char));
        mnist.push_back(number);
    }
    return mnist;
}

std::vector<Eigen::VectorXd> loadMnistTestVector(const std::string &path) {
    std::ifstream stream(path + "/t10k-images.idx3-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<Eigen::VectorXd> mnist;

    Eigen::VectorXd image(28*28);
    for(int k{}; k<TEST; k++) {
        for (int i{}; i < 28*28; i++) {
            number = 0;
            stream.read(reinterpret_cast<char *>(&number), sizeof(char));
            image(i) = double(number);

        }
        mnist.emplace_back(image);
    }
    return mnist;
}

std::vector<Eigen::VectorXd> loadLabelsVector(const std::string& path){
    std::ifstream stream(path + "/train-labels.idx1-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<Eigen::VectorXd> mnist;
    for(int k{}; k<TRAINING; k++) {
        Eigen::VectorXd labels = Eigen::VectorXd::Zero(10);
        number = 0;
        stream.read(reinterpret_cast<char*>(&number), sizeof(char));
        labels(number) = 1;
        mnist.push_back(labels);
    }
    return mnist;
}

std::vector<Eigen::VectorXd> loadTestLabelsVector(const std::string& path){
    std::ifstream stream(path + "/t10k-labels.idx1-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<Eigen::VectorXd> mnist;

    for(int k{}; k<TEST; k++) {
        Eigen::VectorXd labels = Eigen::VectorXd::Zero(10);
        number = 0;
        stream.read(reinterpret_cast<char *>(&number), sizeof(char));
        labels(number) = 1;
        mnist.push_back(labels);
    }
    return mnist;
}

std::vector<Eigen::MatrixXd> loadMnistTestMatrix(const std::string &path) {
    std::ifstream stream(path + "/t10k-images.idx3-ubyte", std::ios::binary);
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    int number{};
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);
    stream.read(reinterpret_cast<char*>(&number), sizeof number);

    std::vector<Eigen::MatrixXd> mnist;

    Eigen::MatrixXd image(28,28);
    for(int k{}; k<TEST; k++) {
        for (int i{}; i < 28; i++) {
            for (int j{}; j < 28; j++) {
                number = 0;
                stream.read(reinterpret_cast<char*>(&number), sizeof(char));
                image(i, j) = double(number);
            }
        }
        mnist.push_back(image);
    }
    return mnist;
}

