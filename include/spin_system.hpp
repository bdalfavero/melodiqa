#ifndef SPINSYS
#define SPINSYS

#include "neural_state.hpp"

class SpinSystem {
public:
    int size;

    SpinSystem(int size) {
        this->size = size;
    }

    virtual std::complex<float> local_energy(NeuralState nqstate, Eigen::VectorXcf spins) {
        return std::complex<float>(-0.0, 0.0);
    } 
};

class IsingSystem: public SpinSystem {
public:
    float J, B;

    IsingSystem(int size, float J, float B): SpinSystem(size) {
        this->size = size;
        this->J = J;
        this->B = B;
    }
};

#endif