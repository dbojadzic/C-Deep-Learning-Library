//
// Created by mrtheamir on 5/17/19.
//

#ifndef DAMKENET_NET_H
#define DAMKENET_NET_H

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <algorithm>
#include <memory>
#include "Layer.h"

/*
 * The class Net represents an interface between the user and the Layers. Generally speaking, the user should not have to
 * instantiate layers on his own and should not worry about the different inputs and outputs of the layer. The class Net
 * is supposed to bridge the gap between the user and the many layers. It provides an intuitive interface for the user
 * to create layers and to train the network and use it for predictions.
 */

class Net{
    std::vector<std::shared_ptr<Layer>> layers;

    double loss(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd> &targets);

    double loss(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::VectorXd> &targets);
public:
    /*
     * The class Net can be created using two constructors. The default constructor simply creates an empty neural net
     * which can be filled up with layers. The second constructor uses an initializer list of integers to create the net.
     * Please not that the network created using the initializer list is a simple fully connected neural network and
     * the elements specified represent the different numbers of neurons in each layer.
     */
    Net()= default;
    Net(std::initializer_list<int>);

    /*
     * The class Net offers a simple interface for creating layers. The user should already be familiar with the different
     * possible ways to initialize the layers from Layer.h and Layer.cpp. The functions below are a simple wrapper of the
     * constructors from the Layer classes.
     * addFCLayer adds a fully connected layer to the network
     * addConvLayer adds a convolutional layer to the network
     * addMaxPoolLayer adds a maxpooling layer to the network
     */

    void addFCLayer(int neurons, int inputs, const std::string &activation = "Leaky_ReLU");

    void addFCLayer(const Eigen::MatrixXd &weight, const Eigen::VectorXd &bias,
                    const std::string &activation = "Leaky ReLU");
    void addConvLayer(int rows, int cols, int width, int kernels = 1, int padding = 0, int stride = 1);

    void addConvLayer(const std::vector<std::vector<Eigen::MatrixXd>> &kernels, int padding = 0, int stride = 1);
    void addMaxPoolLayer(int rows, int cols);

    /*
     * The functions predict and predict label simply pass the input through the network and give the output to the user.
     * The difference between these functions is that the predict functions return the whole prediction vector, which is
     * useful when calculating the loss of the network, while the predictLabel functions simply return the maximal element
     * of those vectors. For example when training on the MNIST dataset, you want the whole output vector to be as close
     * to 1 0 0 0 0 0 0 0 0 0 (if the input is a zero) but when it comes to predicting, you're simply interested in the
     * element with the highest prediction.
     */
    Eigen::VectorXd predict(Eigen::VectorXd input);
    Eigen::VectorXd predict(Eigen::MatrixXd input);
    int predictLabel(Eigen::VectorXd input);
    int predictLabel(Eigen::MatrixXd input);

    /*
     * The train function is perhaps the most important function of the class Net since it controls everything that is
     * learning-related. The function expects an std::vector of vectors/matrices as input and an std::vector of vectors
     * as the target (this is due to the predict function outputting an std::vector). The function also allows tuning of
     * other important hyperparameters such as the number of epoch, the learning rate. the batchsize, momentum strength
     * and if early stopping should be employed as a tactic for the prevention of overfitting.
     */
    void train(const std::vector<Eigen::VectorXd> &inputs, const std::vector<Eigen::VectorXd>&
    targets, int epochs = 200, int batchSize = 1, double learningRate = 0.00001, double
               regularization = 0.0000001, double momentum = 0.9, bool earlyStopping = false);
    void train(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::VectorXd>&
    targets, int epochs = 200, int batchSize = 1, double learningRate = 0.00001, double
               regularization = 0.0000001, double momentum = 0.9, bool earlyStopping = false);

    void save(const std::string& path = "data");
    void load(const std::string& path = "data");
};


#endif //DAMKENET_NET_H
