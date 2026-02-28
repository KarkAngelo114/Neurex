const { transform_to_tensor } = require('../core/bindings');
const { red, reset } = require('../prettify')

/**
 * 
 * @param {Array<Number>} input - Scalar input
 * @param {Array<Number>} shape - array that contains the values for Height, Width, and Depth. They must be arrange in correct order as [Height, Width, Depth]
 * @returns tensor map having the same given shape of HxWxD
 */
const toTensor = (input, shape = [0, 0, 0]) => {
    const arr_Length = input.length;
    const [h, w, d] = shape;
    const arr = input.flat(Infinity);

    // if arr_Length not equal to h*w*d, append 0s to match the given shape
    if (arr_Length != (h * w * d)) {
        console.log(`${red}[ERROR]------- Failed to reshape: the length of the input is less than or not eqaul to the given h * w * d${reset}`);
        throw new Error();
    }

    return transform_to_tensor(arr, h, w, d);
}

module.exports = {
    toTensor
}