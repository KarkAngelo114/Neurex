const getKernels = (kernels) => {
    const output = [];

    for (let i = 0; i < kernels.length; i++) {
        const r = kernels[i].reverse().map(row => row.reverse());

        output.push(r);
    }

    return output
}



const ConvolveDeltaTest = (padded_dilated_delta, kernels, inputH, inputW) => {

    const kernels_array = getKernels(kernels);
    let KH = kernels_array[0].length;
    let KW = kernels_array[0][0].length;
    let D =  padded_dilated_delta[0][0].length > kernels_array[0][0][0].length ? kernels_array[0][0][0].length : padded_dilated_delta[0][0].length; 

    const output = Array.from({length: inputH}, () => Array.from({length: inputW}, () => Array(kernels_array.length).fill(0)));

    for (let f = 0; f < kernels_array.length; f++) {
        const kernel = kernels_array[f]; // get the current filter

        for (let y = 0; y < inputH; y++) {
            for (let x = 0; x < inputW; x++) {
                let sum = 0;

                for (let kh = 0; kh < KH; kh++) {
                    for (let kw = 0; kw < KW; kw++) {
                        for (let c = 0; c < D; c++) {
                            let input = padded_dilated_delta[y + kh][x + kw][c];
                            let kernel_value = kernel[kh][kw][c];

                            sum += input * kernel_value;
                        }
                    }
                }

                output[y][x][f] = sum;

            }
        }
    };

    return output

}

module.exports = {
    ConvolveDeltaTest
}