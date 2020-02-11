//
// Created by mrtheamir on 7/27/19.
//

#include <Eigen/Dense>
#include "../src/Layer.h"
#include "../src/net.h"
#include <string>
#include <cmath>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <fstream>
#include <vector>


TEST_CASE("Testing the convolution function") {
    Eigen::MatrixXd input(4,4);
    input << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;
    std::string error;

    SECTION("Errors"){
        try{
            Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(5,5);
            convolution(input, kernel);
        } catch (std::invalid_argument &e) {
            error = e.what();
        }
        REQUIRE(error == "Kernel bigger than input");
        try{
            Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(3,3);
            convolution(input, kernel, -2);
        } catch (std::invalid_argument &e) {
            error = e.what();
        }
        REQUIRE(error == "Stride can't be negative");
        try{
            Eigen::MatrixXd kernel = Eigen::MatrixXd::Random(3,3);
            convolution(input, kernel, 2);
        } catch (std::invalid_argument &e) {
            error = e.what();
        }
        REQUIRE(error == "Invalid stride");
    }

    SECTION("Convolution verification"){
        Eigen::MatrixXd kernel(1,1);
        kernel(0,0) = 1;
        Eigen::MatrixXd output = convolution(input, kernel);
        REQUIRE(input == output);

        Eigen::MatrixXd control(3,3);
        control << 10, 14, 18, 26, 30, 34, 42, 46, 50;
        kernel = Eigen::MatrixXd(2,2);
        kernel << 1, 1, 1, 1;
        output = convolution(input, kernel);
        REQUIRE(output == control);

        control = Eigen::MatrixXd(3,3);
        control << 5, 7, 9, 13, 15, 17, 21, 23, 25;
        kernel = Eigen::MatrixXd(2,2);
        kernel << 0, 1, 1, 0;
        output = convolution(input, kernel);
        REQUIRE(output == control);
    }
}

TEST_CASE("Testing matrix padding"){
    Eigen::MatrixXd matrix(2,2);
    matrix << 1, 1, 1, 1;
    SECTION("Errors"){
        std::string error;
        try{
            matrixPadding(matrix, -1);
        } catch(std::invalid_argument &e) {
            error = e.what();
        }
        REQUIRE(error == "Argument padding can't be negative");
    }
    SECTION("Padding verification"){
        Eigen::MatrixXd control(4,4);
        control << 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0;
        REQUIRE(control == matrixPadding(matrix, 1));
        control = Eigen::MatrixXd(4,4);
        control << 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2;
        REQUIRE(control == matrixPadding(matrix, 1, 2));
    }
}

TEST_CASE("Testing Fully Connected layer"){
    SECTION("Testing prediction"){
        Eigen::MatrixXd weights(2,3);
        Eigen::VectorXd bias(2);
        Eigen::VectorXd input(3);
        Eigen::VectorXd control(2);
        weights << 1, 2, 3, 4, 5, 6;
        bias << -1, -2;
        input << 1, 1, -1;
        control << 0, 1;
        FCLayer l1(weights, bias, "ReLU");
        REQUIRE(control == l1.predict(input));
        control = Eigen::VectorXd(2);
        control << -0.1, 1;
        FCLayer l2(weights, bias, "Leaky_ReLU");
        REQUIRE(control == l2.predict(input));
        control = Eigen::VectorXd(2);
        control << 0.268941, 0.731059;
        FCLayer l3(weights, bias, "Sigmoid");
        REQUIRE(abs(control[0] - l3.predict(input)[0]) < 0.00001);
        REQUIRE(abs(control[1] - l3.predict(input)[1]) < 0.00001);
    }
    SECTION("Testing forward pass"){
        Eigen::MatrixXd weights(2,3);
        Eigen::VectorXd bias(2);
        Eigen::VectorXd input(3);
        Eigen::VectorXd control(2);
        weights << 1, 2, 3, 4, 5, 6;
        bias << -1, -2;
        input << 1, 1, -1;
        control << 0, 1;
        FCLayer l1(weights, bias, "ReLU");
        REQUIRE(control == l1.feedForward(input));
        control = Eigen::VectorXd(2);
        control << -0.1, 1;
        FCLayer l2(weights, bias, "Leaky_ReLU");
        REQUIRE(control == l2.feedForward(input));
        control = Eigen::VectorXd(2);
        control << 0.268941, 0.731059;
        FCLayer l3(weights, bias, "Sigmoid");
        REQUIRE(abs(control[0] - l3.feedForward(input)[0]) < 0.00001);
        REQUIRE(abs(control[1] - l3.feedForward(input)[1]) < 0.00001);
    }
    SECTION("Testing forward and backward pass and gradient"){
        Eigen::MatrixXd weights(2,3);
        Eigen::VectorXd bias(2);
        Eigen::VectorXd input(3);
        Eigen::VectorXd control(3);
        weights << 1, 2, 3, 4, 5, 6;
        bias << -1, -2;
        input << 1, 1, -1;
        control << 4, 5, 6;
        FCLayer l1(weights, bias, "ReLU");
        l1.resetDerivative();
        Eigen::VectorXd result = l1.feedForward(input);
        Eigen::VectorXd truth(2);
        truth << 0, 0; // Assuming that the output has to be [0, 0]
        REQUIRE(control == l1.feedBackward(result - truth));
        l1.accumulateDerivative();
        l1.derive(1,1,0,0);
        REQUIRE(truth == l1.feedForward(input));
    }
    SECTION("Testing save/load functionality"){
        Eigen::MatrixXd weights(2,3);
        Eigen::VectorXd bias(2);
        Eigen::VectorXd input(3);
        Eigen::VectorXd control(2);
        weights << 1, 2, 3, 4, 5, 6;
        bias << -1, -2;
        input << 1, 1, -1;
        control << 0, 1;
        FCLayer l1(weights, bias, "ReLU");
        REQUIRE(control == l1.predict(input));
        std::ofstream config("datatesting/config");
        l1.save(config, "datatesting/testlayer");
        config.close();
        FCLayer l2;
        std::ifstream iconfig("datatesting/config");
        char a;
        iconfig >> a;
        l2.load(iconfig, "datatesting/testlayer");
        REQUIRE(control == l2.predict(input));
    }
}

TEST_CASE("Testing MaxPooling layer"){
    SECTION("Testing prediction"){
        //Even though Pool layers can't predict, this functionality was needed to be implemented for testing
        MaxPool l(2,2);
        Eigen::MatrixXd input(4,4);
        input << 1,2,5,6, 3,4,8,7, 9,11,20,19, 8,10,18,17;
        Eigen::MatrixXd control(2,2);
        control << 4,8,11,20;
        Eigen::MatrixXd control2(1,1);
        control2 << 20;
        Eigen::MatrixXd output;
        output = l.predict(input);
        REQUIRE(control == output);
        output = l.predict(output);
        REQUIRE(control2 == output);
        std::string error;
        try {
            output = l.predict(output);
        } catch (std::invalid_argument &e){
            error = e.what();
        }
        REQUIRE(error == "Layer not compatible with pool mask");
    }
    SECTION("Testing forward pass"){
        MaxPool l(2,2);
        std::vector<Eigen::MatrixXd> input{Eigen::MatrixXd(4,4)};
        input[0] << 1,2,5,6, 3,4,8,7, 9,11,20,19, 8,10,18,17;
        std::vector<Eigen::MatrixXd> control{Eigen::MatrixXd(2,2)};
        control[0] << 4,8,11,20;
        std::vector<Eigen::MatrixXd> control2{Eigen::MatrixXd(1,1)};
        control2[0] << 20;
        std::vector<Eigen::MatrixXd> output;
        output = l.feedForward(input);
        REQUIRE(control == output);
        output = l.feedForward(output);
        REQUIRE(control2 == output);
        std::string error;
        try {
            output = l.feedForward(output);
        } catch (std::invalid_argument &e){
            error = e.what();
        }
        REQUIRE(error == "Layer not compatible with pool mask");
    }
    SECTION("Testing forward and backward pass"){
        MaxPool l(2,2);
        std::vector<Eigen::MatrixXd> input{Eigen::MatrixXd(4,4)};
        input[0] << 1,2,5,6, 3,4,8,7, 9,11,20,19, 8,10,18,17;
        std::vector<Eigen::MatrixXd> control{Eigen::MatrixXd(4,4)};
        control[0] << 0,0,0,0, 0,1,1,0, 0,1,1,0, 0,0,0,0;
        std::vector<Eigen::MatrixXd> derivative{Eigen::MatrixXd(2,2)};
        derivative[0] << 1,1,1,1;
        std::vector<Eigen::MatrixXd> output;
        l.feedForward(input);
        output = l.feedBackward(derivative);
        REQUIRE(control == output);
    }
    SECTION("Testing save/load functionality"){
        MaxPool l(2,2);
        MaxPool l2;
        std::ofstream config("datatesting/config");
        l.save(config, "");
        config.close();
        std::ifstream iconfig("datatesting/config");
        char a;
        iconfig >> a;
        l2.load(iconfig, "");
        std::vector<Eigen::MatrixXd> input{Eigen::MatrixXd(4,4)};
        input[0] << 1,2,5,6, 3,4,8,7, 9,11,20,19, 8,10,18,17;
        REQUIRE(l.feedForward(input) == l2.feedForward(input));
    }
}

TEST_CASE("Testing Convoutional layer"){
    SECTION("Testing forward pass"){
        Eigen::MatrixXd kernel(1,1);
        kernel(0,0) = 1;
        std::vector<std::vector<Eigen::MatrixXd>> kernels{std::vector<Eigen::MatrixXd> {kernel}};
        ConvLayer l(kernels);
        std::vector<Eigen::MatrixXd> input{Eigen::MatrixXd(4,4)};
        input[0] << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;
        std::vector<Eigen::MatrixXd> output = l.feedForward(input);
        REQUIRE(input == output);

    }
    SECTION("Testing multi-kernel forward pass"){
        Eigen::MatrixXd kernel(2,2);
        std::vector<std::vector<Eigen::MatrixXd>> kernels(3, std::vector<Eigen::MatrixXd> {kernel});

        //Three 1D kernels
        kernels[0][0](0,0) = 1;
        kernels[0][0](0,1) = 1;
        kernels[0][0](1,0) = 1;
        kernels[0][0](1,1) = 1;

        kernels[1][0](0,0) = 0;
        kernels[1][0](0,1) = 1;
        kernels[1][0](1,0) = 1;
        kernels[1][0](1,1) = 0;

        kernels[2][0](0,0) = 1;
        kernels[2][0](0,1) = 0;
        kernels[2][0](1,0) = 0;
        kernels[2][0](1,1) = 1;

        ConvLayer l(kernels);
        std::vector<Eigen::MatrixXd> input{Eigen::MatrixXd(3,3)};
        input[0] << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        std::vector<Eigen::MatrixXd> output = l.feedForward(input);

        std::vector<Eigen::MatrixXd> control(3, Eigen::MatrixXd(2,2));
        control[0] << 8,12,20,24;
        control[1] << 4,6,10,12;
        control[2] << 4,6,10,12;
        REQUIRE(control == output);
    }
    SECTION("Testing multidimensional-kernel forward pass"){
        Eigen::MatrixXd kernel(2,2);
        std::vector<std::vector<Eigen::MatrixXd>> kernels{std::vector<Eigen::MatrixXd> (3,kernel)};

        //Three 1D kernels
        kernels[0][0](0,0) = 1;
        kernels[0][0](0,1) = 1;
        kernels[0][0](1,0) = 1;
        kernels[0][0](1,1) = 1;

        kernels[0][1](0,0) = 0;
        kernels[0][1](0,1) = 1;
        kernels[0][1](1,0) = 1;
        kernels[0][1](1,1) = 0;

        kernels[0][2](0,0) = 1;
        kernels[0][2](0,1) = 0;
        kernels[0][2](1,0) = 0;
        kernels[0][2](1,1) = 1;

        ConvLayer l(kernels);
        std::vector<Eigen::MatrixXd> input(3, Eigen::MatrixXd(3,3));
        input[0] << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        input[1] << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        input[2] << 2, 3, 4, 5, 6, 7, 8, 9, 10;
        std::vector<Eigen::MatrixXd> output = l.feedForward(input);

        std::vector<Eigen::MatrixXd> control{Eigen::MatrixXd(2,2)};
        control[0] << 22,30,46,54;

        REQUIRE(control == output);
    }
    SECTION("Testing forward and backward pass multidimensional kernel"){
        Eigen::MatrixXd kernel(2,2);
        std::vector<std::vector<Eigen::MatrixXd>> kernels{std::vector<Eigen::MatrixXd> (3,kernel)};

        //Three 1D kernels
        kernels[0][0](0,0) = 1;
        kernels[0][0](0,1) = 1;
        kernels[0][0](1,0) = 1;
        kernels[0][0](1,1) = 1;

        kernels[0][1](0,0) = 1;
        kernels[0][1](0,1) = 0;
        kernels[0][1](1,0) = 1;
        kernels[0][1](1,1) = 0;

        kernels[0][2](0,0) = 0;
        kernels[0][2](0,1) = 1;
        kernels[0][2](1,0) = 0;
        kernels[0][2](1,1) = 1;

        ConvLayer l(kernels);
        std::vector<Eigen::MatrixXd> input(3, Eigen::MatrixXd(3,3));
        input[0] << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        input[1] << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        input[2] << 2, 3, 4, 5, 6, 7, 8, 9, 10;
        l.feedForward(input);

        std::vector<Eigen::MatrixXd> control(3, Eigen::MatrixXd(3,3));
        std::vector<Eigen::MatrixXd> derivative{Eigen::MatrixXd(2,2)};
        derivative[0] << 1,1,1,1;

        std::vector<Eigen::MatrixXd> output = l.feedBackward(derivative);

        control[0] << 1,2,1,2,4,2,1,2,1;
        control[1] << 1,1,0,2,2,0,1,1,0;
        control[2] << 0,1,1,0,2,2,0,1,1;

        REQUIRE(control == output);
    }
    SECTION("Testing forward and backward pass multiple kernels kernel"){
        Eigen::MatrixXd kernel(2,2);
        std::vector<std::vector<Eigen::MatrixXd>> kernels(3, std::vector<Eigen::MatrixXd>{kernel});

        //Three 1D kernels
        kernels[0][0](0,0) = 1;
        kernels[0][0](0,1) = 1;
        kernels[0][0](1,0) = 1;
        kernels[0][0](1,1) = 1;

        kernels[1][0](0,0) = 1;
        kernels[1][0](0,1) = 0;
        kernels[1][0](1,0) = 1;
        kernels[1][0](1,1) = 0;

        kernels[2][0](0,0) = 0;
        kernels[2][0](0,1) = 1;
        kernels[2][0](1,0) = 0;
        kernels[2][0](1,1) = 1;

        ConvLayer l(kernels);
        std::vector<Eigen::MatrixXd> input(1, Eigen::MatrixXd(3,3));
        input[0] << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        l.feedForward(input);

        std::vector<Eigen::MatrixXd> control(1, Eigen::MatrixXd(3,3));
        std::vector<Eigen::MatrixXd> derivative(3, Eigen::MatrixXd(2,2));
        derivative[0] << 1,1,1,1;
        derivative[1] << 1,1,1,1;
        derivative[2] << 1,1,1,1;

        std::vector<Eigen::MatrixXd> output = l.feedBackward(derivative);

        control[0] << 2,4,2,4,8,4,2,4,2;

        REQUIRE(control == output);
    }
    SECTION("Testing save/load functionality"){
        Eigen::MatrixXd kernel(2,2);
        std::vector<std::vector<Eigen::MatrixXd>> kernels{std::vector<Eigen::MatrixXd> (3,kernel)};

        //Three 1D kernels
        kernels[0][0](0,0) = 1;
        kernels[0][0](0,1) = 1;
        kernels[0][0](1,0) = 1;
        kernels[0][0](1,1) = 1;

        kernels[0][1](0,0) = 0;
        kernels[0][1](0,1) = 1;
        kernels[0][1](1,0) = 1;
        kernels[0][1](1,1) = 0;

        kernels[0][2](0,0) = 1;
        kernels[0][2](0,1) = 0;
        kernels[0][2](1,0) = 0;
        kernels[0][2](1,1) = 1;

        ConvLayer l(kernels);
        std::vector<Eigen::MatrixXd> input(3, Eigen::MatrixXd(3,3));
        input[0] << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        input[1] << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        input[2] << 2, 3, 4, 5, 6, 7, 8, 9, 10;
        std::vector<Eigen::MatrixXd> output = l.feedForward(input);

        std::ofstream config("datatesting/config");
        l.save(config, "datatesting/testlayer");
        config.close();
        ConvLayer l2;
        std::ifstream iconfig("datatesting/config");
        char a;
        iconfig >> a;
        l2.load(iconfig, "datatesting/testlayer");
        REQUIRE(output == l2.feedForward(input));
    }
}


