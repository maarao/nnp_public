/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *
 *  Location for CUDA kernels
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "kernels.h"

// Device helper for ReLU
__device__ float d_relu(float x) { return x > 0 ? x : 0; }

// Device helper for ReLU derivative
__device__ float d_drelu(float y) { return y > 0 ? 1 : 0; }

/* Forward pass with ReLU activation: out = Relu(in * W + b)
 * Parallelism: One thread per output neuron (j)
 */
__global__ void k_forward_relu(float *d_in, float *d_W, float *d_b, float *d_out, float *d_out_a, int input_size, int output_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < output_size)
    {
        float sum = d_b[j];
        for (int i = 0; i < input_size; i++)
        {
            sum += d_in[i] * d_W[i * output_size + j];
        }
        d_out[j] = sum;
        d_out_a[j] = d_relu(sum);
    }
}

/* Forward pass without activation: out = in * W + b
 * Parallelism: One thread per output neuron (k)
 */
__global__ void k_forward(float *d_in, float *d_W, float *d_b, float *d_out, int input_size, int output_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < output_size)
    {
        float sum = d_b[k];
        for (int i = 0; i < input_size; i++)
        {
            sum += d_in[i] * d_W[i * output_size + k];
        }
        d_out[k] = sum;
    }
}

/* Softmax + Cross Entropy Loss + Delta calculation for Output Layer
 * Parallelism: Single block (assumed small number of classes, e.g., 10)
 */
__global__ void k_softmax_cross_entropy(float *d_out, float *d_out_a, float *d_label, float *d_delta, float *d_loss, int classes)
{
    int k = threadIdx.x;
    if (k >= classes)
        return;

    extern __shared__ float shared_mem[];
    float *s_out = shared_mem;

    s_out[k] = d_out[k];
    __syncthreads();

    float max_val = s_out[0];
    for (int i = 1; i < classes; i++)
    {
        if (s_out[i] > max_val)
            max_val = s_out[i];
    }

    // Compute exp
    float val = expf(s_out[k] - max_val);
    s_out[k] = val;
    __syncthreads();

    // Compute sum
    float sum = 0.0f;
    for (int i = 0; i < classes; i++)
        sum += s_out[i];

    // Compute softmax
    float prob = s_out[k] / sum;
    d_out_a[k] = prob;

    // Compute Delta3
    d_delta[k] = d_label[k] - prob;

    // Accumulate Loss
    if (d_label[k] > 0.0f)
    {
        atomicAdd(d_loss, -1.0f * logf(prob + 1e-8f));
    }
}

/* Backpropagation for hidden layers: calculates delta for current layer
 * Parallelism: One thread per neuron in current layer (j)
 */
__global__ void k_backprop_relu(float *d_delta_next, float *d_W_next, float *d_out_a, float *d_delta, int size_next, int size_curr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size_curr)
    {
        float err = 0.0f;
        for (int k = 0; k < size_next; k++)
        {
            // W_next is [size_curr * size_next], access W[j * size_next + k]
            err += d_delta_next[k] * d_W_next[j * size_next + k];
        }
        d_delta[j] = err * d_drelu(d_out_a[j]);
    }
}

/* Update weights
 * Parallelism: One thread per weight. gridDim/blockDim should cover size_in * size_out
 */
__global__ void k_update_weights(float *d_W, float *d_delta, float *d_prev_out_a, int size_in, int size_out, float lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = size_in * size_out;

    if (idx < total_weights)
    {
        // Map 1D index back to i (input) and j (output/next)
        // W is size_in * size_out. W[i * size_out + j]
        int i = idx / size_out;
        int j = idx % size_out;

        // model->W[i*H+j] += LR * delta[j] * prev_out[i];
        d_W[idx] += lr * d_delta[j] * d_prev_out_a[i];
    }
}

/* Update biases
 * Parallelism: One thread per bias unit
 */
__global__ void k_update_biases(float *d_b, float *d_delta, int size, float lr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < size)
    {
        d_b[j] += lr * d_delta[j];
    }
}
