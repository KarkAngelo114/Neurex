
const reLu = (dot_product_output) => {
    /** 
        f(x) = max(0, x) 
        
        this basically means if X > 0 (x greater than 0), the output returns as is, 
        otherwise if x <= 0 (less than or equal to 0) it will return 0

    */
    return Math.max(0, dot_product_output); 
};

const dreLu = (x) => {

    /**
       f'(x) = { 1 if x > 0, 0 if x ≤ 0 }
     */
    return x > 0 ? 1 : 0
};

const sigmoid = (dot_product_output) => {
    /**
                        1
        F(x) =   ________________        
                    1 + e^(-x)

        where:

        e - is the Euler's number (2.718)
        x - is the input number
        F(x) - is the output activation function
    
     */

    return 1 / (1 + Math.exp(dot_product_output));
};

const dsigmoid = (x) => {
    /**
        f'(x) = f(x) * (1 - f(x))
     */
    const s = sigmoid(x);
    return s * (1 - s);
};

const tanh = (dot_product_output) => {
    /**
                e^(x) - e^(-x)
        F(x) = ________________
                e^(x) + e^(-x)

        where:
            e - is the Euler's number (2.718)
            x - is the input
            F(x) - is the output activation function
     */
    return Math.tanh(dot_product_output);
}

const dtanh = (x) => {
     /**
        f'(x) = 1 - tanh²(x)
     */
    const t = Math.tanh(x);
    return 1 - t * t;
};

const linear = (dot_product_output) => {
    /**

        F(x) = x

        whatever the input is still returns the same as output activation function

     */
    return dot_product_output;
};

const dlinear = () => {
    /**
        f'(x) = 1
    
     */
    return 1;
};



/**
 * Softmax activation function
 * @param {Array<number>} logits - Array of raw output values (logits) from the last layer
 * @returns {Array<number>} - Array of probabilities (sum to 1)
 */
const softmax = (logits) => {
    // For numerical stability, subtract max from logits
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
};

const dsoftmax = (x) => 1;

module.exports = {
    relu: reLu,
    sigmoid: sigmoid,
    tanh: tanh,
    linear: linear,
    softmax: softmax,
    derivatives: {
        relu: dreLu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        linear: dlinear,
        softmax: dsoftmax
    }
};
