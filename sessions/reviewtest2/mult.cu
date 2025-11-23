__global__ void mult(float* c, float* a, float* b, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { // 31*32 31*32+1 = 993, .. 31*32+8  
        c[idx] = a[idx] * b[idx];
    }
}


__global__ void mult(float* c, float* a, float* b, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return; // 31*32 31*32+1 = 993, .. 31*32+8  
    c[idx] = a[idx] * b[idx];
}

__global__ void gravsim(float* x, float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    float ax = 0, ay = 0, az = 0;
    for (int i = 0; i < n; i++) {
        if (i == idx) continue;
        float dx = x[i] - x[idx];
        float dy = y[i] - y[idx];
        float dz = z[i] - z[idx];
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrt(r2);
        float a_r = Gm[i]/(r2*r);
        ax += a_r*dx;
        ay += a_r*dy;
    }
    __syncthreads();
    x[idx] = x[idx] + vx[idx] * dt;
    y[idx] = y[idx] + vy[idx] * dt;
    z[idx] = z[idx] + vz[idx] * dt;
}

int main() {
    const int n = 1'000;
    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    gravsim<<<32, 32>>>(x, y, z, n);
}