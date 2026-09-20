inline float mha_project(__global const float* input, __global const float* weights,__global const float* biases, int token, int embedDim, int outputIndex) {
	float value = biases[outputIndex];
	for (int i = 0; i < embedDim; ++i) {
        value += input[token * embedDim + i] * weights[i * embedDim + outputIndex];
    }
		
	return value;
}

inline float mha_q(
    __global const float* input, 
    __global const float* weights,
    __global const float* biases, int token, int embedDim, int index
) {
	return mha_project(input, weights, biases, token, embedDim, index);
}

inline float mha_score(
    __global const float* input, 
    __global const float* qWeights,
	__global const float* kWeights, 
    __global const float* qBiases,
    __global const float* kBiases, 
    int query, 
    int key, 
    int head,
	int embedDim, int headDim, float scale
) {
	float value = 0.0f;

	for (int d = 0; d < headDim; ++d) {
        value += mha_q(input, qWeights, qBiases, query, embedDim, head * headDim + d) * mha_q(input, kWeights, kBiases, key, embedDim, head * headDim + d);
    }
		
	return value * scale;
}

inline float mha_softmax(
    __global const float* input, 
    __global const float* qWeights,
	__global const float* kWeights, 
    __global const float* qBiases,
	__global const float* kBiases, 
    int query, 
    int key, 
    int head,
    int embedDim, int seqLen, int headDim, float scale, int causal
) {

	if (causal && key > query) return 0.0f;
	float maximum = -3.402823e+38f;
	for (int candidate = 0; candidate < seqLen; ++candidate) {
        if (!(causal && candidate > query)) {
            maximum = fmax(maximum, mha_score(input, qWeights, kWeights, qBiases, kBiases, query, candidate, head, embedDim, headDim, scale));
        }
    }
		
	float denominator = 0.0f;
	for (int candidate = 0; candidate < seqLen; ++candidate) {
        if (!(causal && candidate > query)) {
            denominator += exp(mha_score(input, qWeights, kWeights, qBiases, kBiases, query, candidate, head, embedDim, headDim, scale) - maximum);
        }
    }
		
			
	return exp(mha_score(input, qWeights, kWeights, qBiases, kBiases, query, key, head, embedDim, headDim, scale) - maximum) / denominator;
}

__kernel void multi_head_attention(
	__global const float* input,
	__global const float* qWeights, 
    __global const float* kWeights, 
    __global const float* vWeights, 
    __global const float* oWeights,
	__global const float* qBiases, 
    __global const float* kBiases, 
    __global const float* vBiases, 
    __global const float* oBiases,
	__global float* qOutput, 
    __global float* kOutput, 
    __global float* vOutput, 
    __global float* mhaOutput,
	__global float* scores,
    __global float* finalOutput,
	int embedDim, 
    int seqLen, 
    int numHeads, 
    int headDim, 
    float scale, 
    int causal
) {
	int token = get_global_id(0);
	int column = get_global_id(1);
	if (token >= seqLen || column >= embedDim) return;

	qOutput[token * embedDim + column] = mha_q(input, qWeights, qBiases, token, embedDim, column);
	kOutput[token * embedDim + column] = mha_q(input, kWeights, kBiases, token, embedDim, column);
	vOutput[token * embedDim + column] = mha_q(input, vWeights, vBiases, token, embedDim, column);

	int head = column / headDim;
	int headColumn = column % headDim;
	float attentionValue = 0.0f;
	for (int key = 0; key < seqLen; ++key) {
		float probability = mha_softmax(input, qWeights, kWeights, qBiases, kBiases, token, key, head, embedDim, seqLen, headDim, scale, causal);
		attentionValue += probability * mha_q(input, vWeights, vBiases, key, embedDim, head * headDim + headColumn);
		scores[(head * seqLen + token) * seqLen + key] = probability;
	}

	mhaOutput[token * embedDim + column] = attentionValue;

	float result = oBiases[column];
	for (int i = 0; i < embedDim; ++i) result += mhaOutput[token * embedDim + i] * oWeights[i * embedDim + column];
	finalOutput[token * embedDim + column] = result;
}

__kernel void multi_head_attention_projection(
	__global const float* mhaOutput, 
    __global const float* oWeights, 
    __global const float* oBiases,
	__global float* finalOutput, 
    int embedDim, 
    int seqLen
) {
	int token = get_global_id(0);
	int column = get_global_id(1);
	if (token >= seqLen || column >= embedDim) return;
	float result = oBiases[column];
	for (int i = 0; i < embedDim; ++i) result += mhaOutput[token * embedDim + i] * oWeights[i * embedDim + column];
	finalOutput[token * embedDim + column] = result;
}

__kernel void multi_head_attention_backward(
	__global const float* incoming, 
    __global const float* qWeights, 
    __global const float* kWeights, 
    __global const float* vWeights, 
    __global const float* oWeights,
	__global const float* qBiases, 
    __global const float* kBiases, 
    __global const float* vBiases,
	__global const float* qCache, 
    __global const float* kCache, 
    __global const float* vCache, 
    __global const float* scores,
	__global float* dQ, 
    __global float* dK, 
    __global float* dV, 
    __global float* dMha, 
    __global float* dX,
	int embedDim, 
    int seqLen, 
    int numHeads, 
    int headDim, 
    float scale, 
    int causal
) {
	int token = get_global_id(0);
	int column = get_global_id(1);
	if (token >= seqLen || column >= embedDim) return;

	int head = column / headDim;
	int headColumn = column % headDim;
	float dq = 0.0f;
	float dk = 0.0f;
	float dv = 0.0f;
	for (int key = 0; key < seqLen; ++key) {
		float dS = 0.0f;
		for (int d = 0; d < headDim; ++d) dS += dMha[token * embedDim + head * headDim + d] * vCache[key * embedDim + head * headDim + d];
		float dot = 0.0f;
		for (int candidate = 0; candidate < seqLen; ++candidate) {
			float candidateDS = 0.0f;
			for (int d = 0; d < headDim; ++d) candidateDS += dMha[token * embedDim + head * headDim + d] * vCache[candidate * embedDim + head * headDim + d];
			dot += candidateDS * scores[(head * seqLen + token) * seqLen + candidate];
		}
		float dScaled = scores[(head * seqLen + token) * seqLen + key] * (dS - dot);
		if (causal && key > token) dScaled = 0.0f;
		dq += dScaled * kCache[key * embedDim + column];
	}
	for (int query = 0; query < seqLen; ++query) {
		float dS = 0.0f;
		for (int d = 0; d < headDim; ++d) dS += dMha[query * embedDim + head * headDim + d] * vCache[token * embedDim + head * headDim + d];
		float dot = 0.0f;
		for (int candidate = 0; candidate < seqLen; ++candidate) {
			float candidateDS = 0.0f;
			for (int d = 0; d < headDim; ++d) candidateDS += dMha[query * embedDim + head * headDim + d] * vCache[candidate * embedDim + head * headDim + d];
			dot += candidateDS * scores[(head * seqLen + query) * seqLen + candidate];
		}
		float dScaled = scores[(head * seqLen + query) * seqLen + token] * (dS - dot);
		if (causal && token > query) dScaled = 0.0f;
		dk += dScaled * qCache[query * embedDim + column];
		dv += scores[(head * seqLen + query) * seqLen + token] * dMha[query * embedDim + column];
	}
	dQ[token * embedDim + column] = dq * scale;
	dK[token * embedDim + column] = dk * scale;
	dV[token * embedDim + column] = dv;

	float dx = 0.0f;
	for (int e = 0; e < embedDim; ++e) {
		dx += dQ[token * embedDim + e] * qWeights[column * embedDim + e];
		dx += dK[token * embedDim + e] * kWeights[column * embedDim + e];
		dx += dV[token * embedDim + e] * vWeights[column * embedDim + e];
	}
	dX[token * embedDim + column] = dx;
}

__kernel void multi_head_attention_backward_dmha(
	__global const float* incoming, 
    __global const float* oWeights, 
    __global float* dMha,
	int embedDim, 
    int seqLen
) {
	int token = get_global_id(0);
	int column = get_global_id(1);
	if (token >= seqLen || column >= embedDim) return;
	float value = 0.0f;
	for (int output = 0; output < embedDim; ++output) value += incoming[token * embedDim + output] * oWeights[column * embedDim + output];
	dMha[token * embedDim + column] = value;
}

__kernel void multi_head_attention_backward_dx(
	__global const float* dQ, 
    __global const float* dK, 
    __global const float* dV,
	__global const float* qWeights, 
    __global const float* kWeights, 
    __global const float* vWeights,
	__global float* dX, 
    int embedDim
) {
	int token = get_global_id(0);
	int column = get_global_id(1);
	if (token >= get_global_size(0) || column >= embedDim) return;
	float value = 0.0f;
	for (int e = 0; e < embedDim; ++e) {
		value += dQ[token * embedDim + e] * qWeights[column * embedDim + e];
		value += dK[token * embedDim + e] * kWeights[column * embedDim + e];
		value += dV[token * embedDim + e] * vWeights[column * embedDim + e];
	}
	dX[token * embedDim + column] = value;
}

__kernel void accumulate_attention_weight_grads(
	__global const float* input,
	__global const float* mhaOutput,
	__global const float* dQ,
	__global const float* dK,
	__global const float* dV,
	__global const float* dMhaOutput,
	__global float* weightGrads,
	int embedDim,
	int seqLen
) {
	int block = get_global_id(0);
	int inputIndex = get_global_id(1);
	int outputIndex = get_global_id(2);
	if (block >= 4 || inputIndex >= embedDim || outputIndex >= embedDim) return;

	float sum = 0.0f;
	for (int t = 0; t < seqLen; t++) {
		float activation = block == 3
			? mhaOutput[t * embedDim + inputIndex]
			: input[t * embedDim + inputIndex];
		float delta = block == 0
			? dQ[t * embedDim + outputIndex]
			: (block == 1
				? dK[t * embedDim + outputIndex]
				: (block == 2
					? dV[t * embedDim + outputIndex]
					: dMhaOutput[t * embedDim + outputIndex]));
		sum += activation * delta;
	}
	weightGrads[block * embedDim * embedDim + inputIndex * embedDim + outputIndex] += sum;
}

__kernel void accumulate_attention_bias_grads(
	__global const float* dQ,
	__global const float* dK,
	__global const float* dV,
	__global const float* dMhaOutput,
	__global float* biasGrads,
	int embedDim,
	int seqLen
) {
	int block = get_global_id(0);
	int index = get_global_id(1);
	if (block >= 4 || index >= embedDim) return;

	float sum = 0.0f;
	for (int t = 0; t < seqLen; t++) {
		float delta = block == 0
			? dQ[t * embedDim + index]
			: (block == 1
				? dK[t * embedDim + index]
				: (block == 2
					? dV[t * embedDim + index]
					: dMhaOutput[t * embedDim + index]));
		sum += delta;
	}
	biasGrads[block * embedDim + index] += sum;
}
