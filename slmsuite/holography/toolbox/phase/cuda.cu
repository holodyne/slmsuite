#include <cupy/complex.cuh>
// This file is loaded into slmsuite.holography.toolbox.phase at runtime.
// cupy.RawKernels call functions inside it.

// Max basis size in register memory.
//   Our pixel-wise kernels sometimes require a basis of polynomials,
//   where the polynomial is computed for the coordinates of the given pixel and
//   stored in the basis. Every CUDA thread has 255 32-bit (sizeof int, float) registers,
//   so this limits the size of the basis that can be stored. We choose to limit the size
//   to BASIS_SIZE, leaving the other registers to be used for local loop/etc variables.
//   As far as I understand, the time to allocate static register memory does not depend on
//   array size, so having a large default basis is costless.
#define BASIS_SIZE 150

// Version 2 compressed kernel
//
extern "C" __device__ __forceinline__ void populate_basis(
    const float x,
    const float y,
    const int D,                    // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const int M,                    // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* c_md,              // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                // A key for where c_dm is nonzero for speed (size M*D) [shared]
    const int* pxy_m,               // Monomial coefficients (size 2*M) [shared]
    float* basis
) {
    int i = 0;
    int j = 0;
    int d = 0;
    int stride = 0;
    int nx = 0;
    int ny = 0;
    int nx0 = 0;
    int ny0 = 0;
    float monomial = 1.0f;

    // Initialize to zero (very important!)
    for (d = 0; d < D; d++) {
        basis[d] = 0.0f;
    }

    // Loop through all the monomial terms, adding them to the basis where needed.
    for (i = 0; i < M; i++) {
        nx = pxy_m[i];
        ny = pxy_m[i+M];

        if (nx < 0) {   // Special indices
            if (nx == -1) {
                // Vortex phase plate case
                if (x == 0 && y == 0) { monomial = 0.0f; }
                else { monomial = atan2f(x, y); }
            } else {
                monomial = 0.0f;
            }

            // Force reset next iteration.
            nx0 = 1000000;
            ny0 = 1000000;
        } else {        // Monomial case
            // Reset if we're starting a new path.
            if (nx - nx0 < 0 || ny - ny0 < 0) {
                nx0 = 0;
                ny0 = 0;
                monomial = 1.0f;
            }

            // Traverse the path in +x or +y.
            for (j = 0; j < nx - nx0; j++) {
                monomial *= x;
            }
            for (j = 0; j < ny - ny0; j++) {
                monomial *= y;
            }

            // Update the state of the monomial
            nx0 = nx;
            ny0 = ny;
        }

        // Now we need to add this monomial to all relevant basis states
        j = 0;
        stride = i*D;       // Only multiply once and find the starting point.
        d = i_md[stride];   // Get the first term to add.
        while (d >= 0) {    // d == -1 indicates no further terms to add.
            // Add the monomial to the result.
            basis[d] += c_md[stride + d] * monomial;

            // Determine if there's another basis state which needs this monomial
            j++;
            d = i_md[stride + j];   // Keep checking until we find a -1.
        }
    }
}

extern "C" __global__ void compressed_farfield2nearfield(
    const complex<float>* farfield, // Input (size N)
    const int W,                     // Width of nearfield
    const int H,                    // Height of nearfield
    const int N,                    // Size of farfield
    const int D,                    // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const int M,                    // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* a_dn,              // Spot Zernike coefficients (size D*N) [shared]
    const float* c_md,              // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                // A key for where c_dm is nonzero (size M*D) [shared]
    const int* pxy_m,               // Monomial coefficients (size 2*M) [shared]
    const float* X,                 // X grid (WH)
    const float* Y,                 // Y grid (WH)
    complex<float>* nearfield       // Output (size W*H)
) {
    // nf is the index of the pixel in the nearfield.
    int nf = blockDim.x * blockIdx.x + threadIdx.x;

    // Local variables.
    complex<float> result = complex<float>(0.0f, 0.0f);
    float exponent = 0.0f;
    int i = 0;
    int d = 0;
    int k = 0;
    bool inrange = nf < W*H;
    float x = 0.0f;
    float y = 0.0f;

    // Prepare local basis.
    // This is the phase for a given coefficient d at the current pixel.
    // It incurs an O(D) cost, but is amortized over O(N) points.
    float basis[BASIS_SIZE];


    if (inrange) {
        x = X[nf];
        y = Y[nf];
    } else {
        x = 0.0f;
        y = 0.0f;
    }

    populate_basis(
        x,
        y,
        D, M,
        c_md, i_md, pxy_m,
        basis
    );

    // Loop over all the spots.
    for (i = 0; i < N; i++) {
        exponent = 0.0f;
        // Loop over basis indices.
        k = i;
        for (d = 0; d < D; d++) {
            exponent += basis[d] * a_dn[k];
            k += N;
        }

        result += farfield[i] * complex<float>(cosf(exponent), sinf(exponent));
    }

    // Export the result to global memory.
    if (inrange) {
        nearfield[nf] = result;
    }
}

extern "C" __global__ void compressed_nearfield2farfield(
    const complex<float>* nearfield,        // Input (size H*W)
    const int W,                            // Width of nearfield
    const int H,                            // Height of nearfield
    const int N,                            // Size of farfield
    const int D,                            // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const int M,                            // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* a_dn,                      // Spot Zernike coefficients (size N*D) [shared]
    const float* c_md,                      // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                        // A key for where c_dm is nonzero (size M*D) [shared]
    const int* pxy_m,                       // Monomial coefficients (size 2*M) [shared]
    const float* X,                 // X grid (WH)
    const float* Y,                 // Y grid (WH)
    complex<float>* farfield_intermediate   // Output (gridDim.x*N)
) {
    // Allocate shared data which will store intermediate results.
    // (Hardcoded to 1024 block size).
    __shared__ float sdata_real[1024 + 64];  // 4 kB
    __shared__ float sdata_imag[1024 + 64];  // 4 kB

    // Make some IDs.
    int tid = threadIdx.x;                  // Thread ID
    int nf = blockDim.x * blockIdx.x + tid; // Nearfield index [0, W*H)
    bool inrange = nf < W*H;

    // Local variables.
    complex<float> result = 0;
    float exponent = 0;
    float coef = 0;
    int i = 0;
    int k = 0;
    int d = 0;
    float x = 0.0f;
    float y = 0.0f;

    // Prepare local basis.
    // This is the phase for a given coefficient d at the current pixel.
    // It incurs an O(D) cost, but is amortized over O(N) points.
    float basis[BASIS_SIZE];

    if (inrange) {
        x = X[nf];
        y = Y[nf];
    } else {
        x = 0.0f;
        y = 0.0f;
    }

    populate_basis(
        x,
        y,
        D, M,
        c_md, i_md, pxy_m,
        basis
    );

    // Iterate through farfield points. Use our basis to find the results.
    for (i = 0; i < N; i++) {
        // Loop over basis indices.
        exponent = 0.0f;
        k = i;
        for (d = 0; d < D; d++) {
            exponent += basis[d] * a_dn[k];
            k += N;
        }

        // Do the overlap integrand for one nearfield-farfield mapping.
        if (inrange) {
            result = conj(nearfield[nf]) * complex<float>(cosf(exponent), sinf(exponent));
        } else {
            result = complex<float>(0.0f, 0.0f);
        }

        sdata_real[tid] = result.real();
        sdata_imag[tid] = result.imag();

        __syncthreads();
        if (tid < 512) {
            sdata_real[tid] += sdata_real[tid + 512];
            sdata_imag[tid] += sdata_imag[tid + 512];
        }
        __syncthreads();
        if (tid < 256) {
            sdata_real[tid] += sdata_real[tid + 256];
            sdata_imag[tid] += sdata_imag[tid + 256];
        }
        __syncthreads();
        if (tid < 128) {
            sdata_real[tid] += sdata_real[tid + 128];
            sdata_imag[tid] += sdata_imag[tid + 128];
        }
        __syncthreads();
        if (tid < 64) {
            sdata_real[tid] += sdata_real[tid + 64];
            sdata_imag[tid] += sdata_imag[tid + 64];
        }
        __syncthreads();
        if (tid < 32) {
            // The last 32 threads don't require __syncthreads() as they can be warped.
            sdata_real[tid] += sdata_real[tid + 32];
            sdata_imag[tid] += sdata_imag[tid + 32];
            __syncwarp();
            sdata_real[tid] += sdata_real[tid + 16];
            sdata_imag[tid] += sdata_imag[tid + 16];
            __syncwarp();
            sdata_real[tid] += sdata_real[tid + 8];
            sdata_imag[tid] += sdata_imag[tid + 8];
            __syncwarp();
            sdata_real[tid] += sdata_real[tid + 4];
            sdata_imag[tid] += sdata_imag[tid + 4];
            __syncwarp();
            sdata_real[tid] += sdata_real[tid + 2];
            sdata_imag[tid] += sdata_imag[tid + 2];
            __syncwarp();
            sdata_real[tid] += sdata_real[tid + 1];
            sdata_imag[tid] += sdata_imag[tid + 1];
            __syncwarp();
        }
        __syncthreads();
        // Save the summed results to global memory.
        if (tid == 0) {
            farfield_intermediate[blockIdx.x + i * gridDim.x] = complex<float>(sdata_real[0], sdata_imag[0]);
        }
    }
}

extern "C" __global__ void zernike_test(
    const int WH,                   // Size of nearfield
    const int D,                    // Dimension of Zernike basis (2 [xy] for normal spot arrays)
    const int M,                    // Dimension of polynomial basis (2 [xy] for normal spot arrays)
    const float* c_md,              // Polynomial coefficients for Zernike (size M*D) [shared]
    const int* i_md,                // A key for where c_dm is nonzero (size M*D) [shared]
    const int* pxy_m,               // Monomial coefficients (size 2*M) [shared]
    const float* X,                 // X grid (WH)
    const float* Y,                 // Y grid (WH)
    float* out                      // Output (size W*H*D)
) {
    // nf is the index of the pixel in the nearfield.
    int nf = blockDim.x * blockIdx.x + threadIdx.x;

    if (nf < WH) {
        // Prepare local basis.
        float basis[BASIS_SIZE];

        populate_basis(
            X[nf],
            Y[nf],
            D, M,
            c_md, i_md, pxy_m,
            basis
        );

        // Export the result to global memory.
        int j = 0;
        for (int i = 0; i < D; i++) {
            out[nf + j] = basis[i];
            j += WH;
        }
    }
}

// Polynomial sum kernel

extern "C" __global__ void polynomial(
    const int WH,                   // Size of nearfield
    const int N,                    // Number of coefficients
    const float* coefficients,      // Monomial coefficients (1*N)
    const short* pxy,               // Monomial exponents (2*N)
    const float* X,                 // X grid (WH)
    const float* Y,                 // Y grid (WH)
    float* out                      // Output (WH)
) {
    // g is each pixel in the grid.
    int g = blockDim.x * blockIdx.x + threadIdx.x;

    if (g < WH) {
        // Make a local result variable to avoid talking with global memory.
        float result = 0;

        // Copy data that will be used multiple times per thread into local memory (this might not matter though).
        float local_X = X[g];
        float local_Y = Y[g];
        float coefficient;

        // Local helper variables.
        float monomial = 1;
        int i, j, nx, ny, nx0, ny0 = 0;

        // Loop over all the spots (compiler should handle optimizing the trinary).
        for (i = 0; i < N; i++) {
            coefficient = coefficients[i];

            if (coefficient != 0) {
                nx = pxy[i];
                ny = pxy[i+N];

                // Reset if we're starting a new path.
                if (nx - nx0 < 0 || ny - ny0 < 0) {
                    nx0 = ny0 = 0;
                    monomial = 1;
                }

                // Traverse the path in +x or +y.
                for (j = 0; j < nx - nx0; j++) {
                    monomial *= local_X;
                }
                for (j = 0; j < ny - ny0; j++) {
                    monomial *= local_Y;
                }

                // Add the monomial to the result.
                result += coefficients[i] * monomial;

                // Update the state of the monomial
                nx0 = nx;
                ny0 = ny;
            }
        }

        // Export the result to global memory.
        out[g] = result;
    }
}

// Weighting

extern "C" __global__ void update_weights_generic(
    float* weight_amp,                  // Input (N)
    const float* feedback_amp,          // Measured amplitudes (N)
    const float* target_amp,            // Desired amplitudes (N)
    const int N,                        // Size
    const int method,                   // Indexed WGS method
    const float feedback_norm,          // cupy-computed norm of feedback_amp
    const float feedback_exponent,      // Method-specific
    const float feedback_factor         // Method-specific
) {
    // i is each pixel in the weights.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        float feedback = feedback_amp[i] / feedback_norm;
        float target = target_amp[i];

        if (method != 4 && method != 5) {   // Multiplicative.
            if (target != 0) {
                feedback /= target;
            } else {
                feedback = 1;
            }

            if (method == 1 || method == 2) {   // Leonardo, Kim
                feedback = pow(-feedback, feedback_exponent);
            } else if (method == 3) {           // Nogrette
                feedback = 1 / (1 - feedback_factor * (1 - feedback));
            }
        } else {                            // Additive.
            if (method == 4) {                  // Wu
                feedback = exp(feedback_exponent * (target - feedback));
            } else if (method == 5) {           // tanh
                feedback = 1 + feedback_factor * tanh(feedback_exponent * (target - feedback));
            }
        }

        // Check nan, inf
        if (isinf(feedback) || isnan(feedback)) { feedback = 1; }

        // Export the result to global memory if changed.
        if (feedback != 1) {
            weight_amp[i] *= feedback;
        }
    }
}