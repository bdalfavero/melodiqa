#ifndef TRAINING
#define TRAINING

struct training_step_result {
    float weight_grad_norm;
    float visible_grad_norm;
    float hidden_grad_norm;
    float energy;
};

struct training_step_result training_step(NeuralState nqstate, SpinSystem spin_sys, int nsweeps, float gamma);

#endif