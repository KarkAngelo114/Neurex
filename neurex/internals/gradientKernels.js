/**
 * Compute weight gradients (returns a 2D array)
 */
function computeWeightGradients(a_prev, delta, layer_name, weightGrads) {
    if (layer_name === "convolutional2D") {
        // console.log('a_prev',a_prev);
        // console.log('delta',delta);
        weightGrads.forEach(w => console.log('Corresponding weight grads',w));
        console.log('activated:',a_prev);
        console.log('Corresponding delta:',delta);

        throw new Error('Stopping');
    }
    else if (layer_name === "connected_layer") {
        
        const weightGrad = [];
        for (let i = 0; i < a_prev.length; i++) {
            const row = [];
            for (let j = 0; j < delta.length; j++) {
                row.push(a_prev[i] * delta[j]);
            }
            weightGrad.push(row);
        }
        const output = weightGrads.map((row, i) =>
            row.map((val, j) => val + weightGrad[i][j])
        );
        return output;
    }
    
}


/**
 * Scale gradients (1D or 2D)
 */
function scaleGradients(onGPU, grad, scalar) {
    if (Array.isArray(grad[0])) {
        // 2D array
        return grad.map(row => row.map(v => v / scalar));
    } else {
        // 1D array
        return grad.map(v => v / scalar);
    }
}

module.exports = {
    computeWeightGradients,
    scaleGradients,
};
