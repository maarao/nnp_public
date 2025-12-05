/*
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *
 *  Header file for CUDA kernel functions
 */

#ifndef KERNELS_H
#define KERNELS_H

// Kernel function prototypes

/* Forward pass with ReLU activation: out = Relu(in * W + b) */
__global__ void k_forward_relu(float *d_in, float *d_W, float *d_b, float *d_out, float *d_out_a, int input_size, int output_size);

/* Forward pass without activation: out = in * W + b */
__global__ void k_forward(float *d_in, float *d_W, float *d_b, float *d_out, int input_size, int output_size);

/* Softmax + Cross Entropy Loss + Delta calculation for Output Layer */
__global__ void k_softmax_cross_entropy(float *d_out, float *d_out_a, float *d_label, float *d_delta, float *d_loss, int classes);

/* Backpropagation for hidden layers: calculates delta for current layer */
__global__ void k_backprop_relu(float *d_delta_next, float *d_W_next, float *d_out_a, float *d_delta, int size_next, int size_curr);

/* Update weights */
__global__ void k_update_weights(float *d_W, float *d_delta, float *d_prev_out_a, int size_in, int size_out, float lr);

/* Update biases */
__global__ void k_update_biases(float *d_b, float *d_delta, int size, float lr);

#endif
