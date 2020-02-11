//
// Created by mrtheamir on 7/21/19.
//

#ifndef DAMKENET_SIGNLANG_H
#define DAMKENET_SIGNLANG_H

#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>

/*
 * The signlang.h file is supposed to load the MNIST dataset of sign language. You can change the parameters TRAINING
 * and TEST to define how many images you want to load from the dataset. The images are 28x28 pixels.
 * The dataset can be found on the following link:
 * https://www.kaggle.com/datamunge/sign-language-mnist
 */

int TRAINING = 27455; //27455
int TEST = 7172;      //7172

std::vector<Eigen::VectorXd> loadSignlangLabels(const std::string &path = "signlang");
std::vector<Eigen::MatrixXd> loadSignlangImages(const std::string& path = "signlang");
std::vector<Eigen::VectorXd> loadSignlangTestLabels(const std::string &path = "signlang");
std::vector<Eigen::MatrixXd> loadSignlangTestImages(const std::string& path = "signlang");
void exportimg(const Eigen::MatrixXd& image, const std::string& path = "txtimages/image.txt");

std::vector<Eigen::VectorXd> loadSignlangLabels(const std::string &path){
    std::ifstream stream(path + "/sign_mnist_train.csv");
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    std::vector<Eigen::VectorXd> labels;
    std::string line;
    std::getline(stream, line);
    int number;
    for(int i{}; i<TRAINING; i++) {
        Eigen::VectorXd label = Eigen::VectorXd::Zero(26);
        number = 0;
        stream >> number;
        label(number) = 1;
        labels.push_back(label);
        std::getline(stream, line);
    }
    return labels;
}

std::vector<Eigen::MatrixXd> loadSignlangImages(const std::string& path){
    std::ifstream stream(path + "/sign_mnist_train.csv");
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    std::vector<int> labels;
    std::string line;
    std::getline(stream, line);
    std::vector<Eigen::MatrixXd> images;
    Eigen::MatrixXd image(28,28);
    int number;
    char comma;

    for(int k{}; k<TRAINING; k++) {
        stream >> number;
        stream >> comma;
        number = 0;
        stream >> number;
        image(0, 0) = double(number);
        for(int i{1}; i<784; i++){
            stream >> comma;
            number = 0;
            stream >> number;
            image(i/28, i%28) = double(number);
        }
        images.push_back(image);
    }
    return images;
}

std::vector<Eigen::VectorXd> loadSignlangTestLabels(const std::string &path){
    std::ifstream stream(path + "/sign_mnist_test.csv");
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    std::vector<Eigen::VectorXd> labels;
    std::string line;
    std::getline(stream, line);
    int number;
    for(int i{}; i<TEST; i++) {
        Eigen::VectorXd label = Eigen::VectorXd::Zero(26);
        number = 0;
        stream >> number;
        label(number) = 1;
        labels.push_back(label);
        std::getline(stream, line);
    }
    return labels;
}

std::vector<Eigen::MatrixXd> loadSignlangTestImages(const std::string& path){
    std::ifstream stream(path + "/sign_mnist_test.csv");
    if(!stream.is_open())
        throw std::logic_error("Stream isn't open");
    std::vector<int> labels;
    std::string line;
    std::getline(stream, line);
    std::vector<Eigen::MatrixXd> images;
    Eigen::MatrixXd image(28,28);
    int number;
    char comma;

    for(int k{}; k<TEST; k++) {
        stream >> number;
        stream >> comma;
        number = 0;
        stream >> number;
        image(0, 0) = double(number);
        for(int i{1}; i<784; i++){
            stream >> comma;
            number = 0;
            stream >> number;
            image(i/28, i%28) = double(number);
        }
        images.push_back(image);
    }
    return images;
}

void exportimg(const Eigen::MatrixXd& image, const std::string& path){
    std::ofstream stream(path);
    stream << image;
}

#endif //DAMKENET_SIGNLANG_H
