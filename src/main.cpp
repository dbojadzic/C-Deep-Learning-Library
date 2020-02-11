#include <iostream>
#include <Eigen/Dense>
#include "net.h"
#include <fstream>
#include "signlang.h"
#include <vector>

int main(){
    Net net;
    net.addConvLayer(7, 7, 1, 3, 2);
    net.addMaxPoolLayer(2,2);
    net.addFCLayer(200, 507, "Leaky_ReLU");
    net.addFCLayer(26, 200, "Leaky_ReLU");

    std::vector<Eigen::VectorXd> targets{loadSignlangLabels()};
    std::vector<Eigen::MatrixXd> inputs{loadSignlangImages()};
    int correct{};
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << "Training set accuracy: " << correct << "/" << inputs.size() << std::endl;

    inputs = loadSignlangTestImages();
    targets = loadSignlangTestLabels();

    correct = 0;
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << "Testing set accuracy: " << correct << "/" << inputs.size() << std::endl;
    std::cout << std::endl;

    exportimg(inputs[1]); // open src/image.txt to see the image in txt form

    for (int i{1}; i <= 2000; i++) { // starts from 1 just for formatting, you can change the number of letters that get printed
        int label{};
        targets[i].maxCoeff(&label);
        std::cout << static_cast<char>(label + 65) << " " << static_cast<char>(net.predictLabel(inputs[i]) + 65) << "  ";
        if(!(i%50))
            std::cout << std::endl;
    }

    net.load("data");
    targets = loadSignlangLabels();
    inputs = loadSignlangImages();
    correct = 0;
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << std::endl;
    std::cout << "Training set accuracy: " << correct << "/" << inputs.size() << std::endl;

    inputs = loadSignlangTestImages();
    targets = loadSignlangTestLabels();

    correct = 0;
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << "Testing set accuracy: " << correct << "/" << inputs.size() << std::endl;
    std::cout << std::endl;

    for (int i{1}; i <= 2000; i++) {
        int label{};
        targets[i].maxCoeff(&label);
        std::cout << static_cast<char>(label + 65) << " " << static_cast<char>(net.predictLabel(inputs[i]) + 65) << "  ";
        if(!(i%50))
            std::cout << std::endl;
    }
}

int main_(){
    Net net;
    net.addConvLayer(7, 7, 1, 3, 2);
    net.addMaxPoolLayer(2,2);
    net.addFCLayer(200, 507, "Leaky_ReLU");
    net.addFCLayer(26, 200, "Leaky_ReLU");

    std::vector<Eigen::VectorXd> targets{loadSignlangLabels()};
    std::vector<Eigen::MatrixXd> inputs{loadSignlangImages()};

    int correct{};
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << "Training set accuracy: " << correct << "/" << inputs.size() << std::endl;

    inputs = loadSignlangTestImages();
    targets = loadSignlangTestLabels();

    correct = 0;
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << "Testing set accuracy: " << correct << "/" << inputs.size() << std::endl;
    std::cout << std::endl;

    inputs = loadSignlangImages();
    targets = loadSignlangLabels();

    net.train(inputs, targets, 50, 200, 0.000005, 0.000000001, 0.9, true);

    correct = 0;
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << std::endl << "Training set accuracy: " << correct << "/" << inputs.size() << std::endl;

    inputs = loadSignlangTestImages();
    targets = loadSignlangTestLabels();

    correct = 0;
    for(int i{}; i<int(inputs.size()); i++) {
        int label{};
        targets[i].maxCoeff(&label);
        if (net.predictLabel(inputs[i]) == label)
            correct++;
    }
    std::cout << "Testing set accuracy: " << correct << "/" << inputs.size() << std::endl;
    std::cout << std::endl;

    return 0;
}