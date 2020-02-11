//
// Created by mrtheamir on 6/15/19.
//

#ifndef DAMKENET_LAYER_H
#define DAMKENET_LAYER_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

/*
 * The files Layer.h and Layer.cpp represent an implementation of standard neural network layers and are therefore
 * one of the building blocks of neural networks. The already predefined functions are tested (tests/tests.cpp) and
 * should in general not be messed with (without reason). The implementation is, however, very flexible to user demands
 * and should the user want to expand this library with his own layer types, that should be done by inheriting the
 * abstract class "Layer".
 */

/*
 * The abstract class Layer provides an interface for existing and future types of layers which can be added. Please
 * familiarize yourself with the interface and the functions of the abstract class before changing existing layers or
 * implementing your own layers. A detailed guide will be provided in the documentation.
 */

class Layer{
public:
    virtual ~Layer() = default;

    virtual Eigen::VectorXd feedForward(Eigen::VectorXd input) = 0;
    virtual Eigen::VectorXd feedBackward(Eigen::VectorXd input) = 0;
    [[nodiscard]] virtual Eigen::VectorXd predict(Eigen::VectorXd input) const = 0;

    virtual std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> input) = 0;
    virtual std::vector<Eigen::MatrixXd> feedBackward(std::vector<Eigen::MatrixXd> input) = 0;
    [[nodiscard]] virtual Eigen::MatrixXd predict(Eigen::MatrixXd input) const = 0;

    virtual void resetDerivative() = 0;
    virtual void accumulateDerivative() = 0;
    virtual void derive(int batchSize, double learningRate, double regularization, double momentum) = 0;

    virtual void save(std::ofstream& config, const std::string &path) const = 0;
    virtual void load(std::ifstream& config, const std::string &path) = 0;
};

/*
 * The class FCLayer represents fully connected layers of a neural network. The class is very powerful but due to the
 * large amount of multiplications and operations it is the slowest of the layers.
 */

class FCLayer : public Layer{
    /* Layer attributes */
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    std::function<double(const double)> activationFunction;
    std::function<double(const double)> activationDerivative;
    std::string activation; // name of the activation function - used for saving and loading

    /* Intermediate values */
    Eigen::VectorXd x; // value after activation function
    Eigen::VectorXd z; // derivative of activation function z=df(x)

    Eigen::MatrixXd weightDerivative; // values by which the weights and biases change
    Eigen::VectorXd biasDerivative;

    Eigen::MatrixXd weightAccumulation; // an accumulation of the derivatives (used in batch learning)
    Eigen::VectorXd biasAccumulation;

    Eigen::MatrixXd weightMomentum; // momentum for adaptive learning
    Eigen::VectorXd biasMomentum;
public:
    /*
     * The fully connected layers can be created by specifying the number of inputs and outputs of the layer or by
     * directly providing the layer with a weight and bias matrix. It can also be created with the default constructor
     * although this is not advised unless you're using the load() function to initialize the layer.
     */
    FCLayer() = default;
    FCLayer(int input, int output, const std::string& activation);
    FCLayer(Eigen::MatrixXd weight, Eigen::VectorXd bias, const std::string& activation);

    Eigen::VectorXd feedForward(Eigen::VectorXd input) override;
    Eigen::VectorXd feedBackward(Eigen::VectorXd input) override;
    [[nodiscard]] Eigen::VectorXd predict(Eigen::VectorXd input) const override;
    void setActivation(const std::string& activation);

    void resetDerivative() override;
    void accumulateDerivative() override;
    void derive(int batchSize, double learningRate, double regularization, double momentum) override;

    void save(std::ofstream& config, const std::string &path) const override;
    void load(std::ifstream& config, const std::string &path) override;

    /*
     * In general, we do not want to pass matrices into fully connected layers, and therefore these functions simply
     * throw errors.
     */
    std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> input) override;
    std::vector<Eigen::MatrixXd> feedBackward(std::vector<Eigen::MatrixXd> input) override;
    [[nodiscard]] Eigen::MatrixXd predict(Eigen::MatrixXd input) const override;
};

/*
 * The functions convolution and matrixPadding represent helper functions which the convolutional layers use. These
 * functions can be used independently of the layers and are therefore quite reusable and robust.
 */

Eigen::MatrixXd convolution(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel, int stride = 1);
Eigen::MatrixXd matrixPadding(const Eigen::MatrixXd& matrix, int padding, double value = 0);

/*
 * The ConvLayer class represents convolutional layers of a neural network. Instead of weights and biases like in the
 * fully connected layers, the convolutional layer uses a set of kernels to compute its output.
 */

class ConvLayer : public Layer{
    /*
     * Layer attributes:
     * The rows and cols attribute represent the numbers of rows and columns of the kernels inside a convolutional layer.
     * The width represents the dimensionality of the kernel.
     * Padding represents the expansion of the input matrix by adding a border of chosen units.
     * Stride is the distance by which the filter is moved in a convolution - It should be noted that even though the
     * stride is implemented and exists as an attribute, it should not be changed or used. Backpropagation with the stride
     * was not implemented and therefore using stride != 1 is dangerous.
     * The kernels attribute represents the various kernels used for convolution - note that the attribute consists of
     * two std::vector-s. This is because the inner vector is used to simulate the dimensionality of the kernel, while
     * the outter one is used to store all kernels.
     *
     * The output of a convolutional layer can be computed using the following formula:
     *  output_dimension = (input_dimension + 2*padding) - kernel_size + 1
     */
    int rows{}, cols{}, width{}, padding{}, stride{};
    std::vector<std::vector<Eigen::MatrixXd>> kernels;

    /* Intermediate values */
    std::vector<Eigen::MatrixXd> x;

    std::vector<std::vector<Eigen::MatrixXd>> kernelDerivative;
    std::vector<std::vector<Eigen::MatrixXd>> kernelAccumulation;
    std::vector<std::vector<Eigen::MatrixXd>> kernelMomentum;
public:
    /*
     * Similar to the fully connected layer, the convolutional layer can also be initialized using layer attributes and
     * randomization, or by directly providing the layer with a set of kernels for it to use.
     */
    ConvLayer() = default;
    ConvLayer(int rows, int cols, int width, int kernels = 1, int padding = 0, int stride = 1);
    explicit ConvLayer(std::vector<std::vector<Eigen::MatrixXd>> kernels, int padding = 0, int stride = 1);

    void resetDerivative() override;
    void accumulateDerivative() override;
    void derive(int batchSize, double learningRate, double regularization, double momentum) override;

    std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> input) override;
    std::vector<Eigen::MatrixXd> feedBackward(std::vector<Eigen::MatrixXd> input) override;
    [[nodiscard]] Eigen::MatrixXd predict(Eigen::MatrixXd input) const override;

    void save(std::ofstream &config, const std::string &path) const override;
    void load(std::ifstream &config, const std::string &path) override;

    /*
     * In general, we do not want to pass vectors into convolutional layers, and therefore these functions simply
     * throw errors. Note that we could in theory do this, but it would require a much more complicated helper function
     * for convolution.
     */
    Eigen::VectorXd feedForward(Eigen::VectorXd input) override;
    Eigen::VectorXd feedBackward(Eigen::VectorXd input) override;
    [[nodiscard]] Eigen::VectorXd predict(Eigen::VectorXd input) const override;
};

/*
 * The class MaxPool represents the MaxPooling layer of neural networks. Its job is to simply "compress" given images
 * using a predefined "window".
 */

class MaxPool : public Layer{
    /*
     * Layer attributes rows and cols represent the number of rows and columns of the compression window. Therefore the
     * output size formula is given by:
     * output_size = input_size/window_size
     */
    int rows{}, cols{};

    /* Intermediate values */
    std::vector<Eigen::MatrixXd> x;
public:
    /*
     * The MaxPool layer can be initialized by simply passing the number of rows and columns of the window. It is a very
     * simple layer and therefore does not require any complicated math or initialization.
     */
    MaxPool() = default;
    MaxPool(int rows, int cols);

    std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> input) override;
    std::vector<Eigen::MatrixXd> feedBackward(std::vector<Eigen::MatrixXd> input) override;
    [[nodiscard]] Eigen::MatrixXd predict(Eigen::MatrixXd input) const override;

    void save(std::ofstream &config, const std::string &path) const override;
    void load(std::ifstream &config, const std::string &path) override;

    /*
     * Since the layer does not need to compute any derivatives but simply passes the incoming derivative back, the following
     * functionalities have been disabled (they can be called but do nothing).
     */
    void resetDerivative() override;
    void accumulateDerivative() override;
    void derive(int batchSize, double learningRate, double regularization, double momentum) override;

    /*
     * Similarly to the previous layers, these functionalities simply throw errors.
     */
    Eigen::VectorXd feedForward(Eigen::VectorXd input) override;
    Eigen::VectorXd feedBackward(Eigen::VectorXd input) override;
    [[nodiscard]] Eigen::VectorXd predict(Eigen::VectorXd input) const override;
};

#endif //DAMKENET_LAYER_H
