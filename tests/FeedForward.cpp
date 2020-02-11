/*
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
 */

//
// Created by mrtheamir on 5/18/19.
/*
Net net{};
Eigen::MatrixXd mat(4,2);
Eigen::MatrixXd mat2(3,4);
Eigen::MatrixXd mat3(1,3);
Eigen::VectorXd vec(4);
Eigen::VectorXd vec2(3);
Eigen::VectorXd vec3(1);
mat << 1,2,3,4,5,6,7,8;
mat2 << 1,2,3,4,5,6,7,8,9,10,11,12;
mat3 << 3,2,1;
vec << 6,5,4,3;
vec2 << 4,3,2;
vec3 << 1;
try {
net.addLayer(mat, vec);
net.addLayer(mat2, vec2);
net.addFCLayer(mat3, vec3);
} catch (std::invalid_argument d) {
std::cout << d.what();
return 1;
}
Eigen::VectorXd input(2);
input << 1, 0;
Eigen::VectorXd d = net.feedForward(input);
std::cout << "result: " << std::endl << d << std::endl;

1105

Net net({2,4,3,1});
Eigen::VectorXd input(2);
input << 1, 0;
Eigen::VectorXd d = net.feedForward(input);
std::cout << "result: " << std::endl << d << std::endl;
return 0;
*/
//



/*
Net net{};
Eigen::MatrixXd mat(4,2);
Eigen::MatrixXd mat2(3,4);
Eigen::MatrixXd mat3(1,3);
Eigen::VectorXd vec(4);
Eigen::VectorXd vec2(3);
Eigen::VectorXd vec3(1);
mat << 1,2,3,4,5,6,7,8;
mat2 << 1,2,3,4,5,6,7,8,9,10,11,12;
mat3 << 3,2,1;
vec << 6,5,4,3;
vec2 << 4,3,2;
vec3 << 1;
try {
    net.addLayer(mat, vec);
    net.addLayer(mat2, vec2);
    net.addFCLayer(mat3, vec3);
} catch (std::invalid_argument d) {
    std::cout << d.what();
    return 1;
}
Eigen::VectorXd input(2);
input << 1, 0;
Eigen::VectorXd target(1);
target << 1;
net.gradient(input, target);
*/


/*
    Net net{};
    Eigen::MatrixXd mat(4,2);
    Eigen::MatrixXd mat2(3,4);
    Eigen::MatrixXd mat3(1,3);
    Eigen::VectorXd vec(4);
    Eigen::VectorXd vec2(3);
    Eigen::VectorXd vec3(1);
    mat << 1,2,3,4,5,6,7,8;
    mat2 << 1,2,3,4,5,6,7,8,9,10,11,12;
    mat3 << 3,2,1;
    vec << 6,5,4,3;
    vec2 << 4,3,2;
    vec3 << 1;
    try {
        net.addLayer(mat, vec);
        net.addLayer(mat2, vec2);
        net.addFCLayer(mat3, vec3);
    } catch (std::invalid_argument d) {
        std::cout << d.what();
        return 1;
    }
    Eigen::VectorXd input(2);
    input << 1, 0;
    Eigen::VectorXd d = net.feedForward(input);
    std::cout << "result: " << std::endl << d << std::endl;
*/

