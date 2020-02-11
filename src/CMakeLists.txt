find_package(Eigen3 REQUIRED)


#../tests/tests.cpp

add_executable(DamkeNet main.cpp net.h net.cpp ../tests/FeedForward.cpp mnist.h Layer.h Layer.cpp signlang.h)

target_link_libraries(DamkeNet Catch2 eigen)

target_compile_features(DamkeNet PUBLIC cxx_std_17)