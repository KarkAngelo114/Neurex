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
        for (let i = 0; i < (h * w * d) - arr_Length; i++) {
            arr.push(0);
        }
    }

    let index = 0
    const tensor = [];

    for (let i = 0; i < h; i++) {
        const rows = [];
        for (let j = 0; j < w; j++) {
            const depth = [];
            for (let k = 0; k < d; k++) {
                depth.push(arr[index++])
            }
            rows.push(depth)
        }
        tensor.push(rows);
    }

    return tensor
}

module.exports = {
    toTensor
}