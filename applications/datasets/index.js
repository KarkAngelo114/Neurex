const { yellow, red, reset, gray } = require('../../color-code');

// fetches the MNIST digits dataset from source
const mnist_digits = async () => {
    let source = "https://cdn.jsdelivr.net/gh/KarkAngelo114/Neurex/applications/datasets/mnist-digits.json";
    console.log(`${yellow}[INFO]${reset} Fetching MNiST digits datasets from ${gray}${source}${reset}`);

    const res = await fetch(source);

    if (res.status != 200) {
        console.error(`${red}[ERROR]${reset} Failed to fetch dataset. It might be network error or the source isn't available. Error code: ${res.status}`)
        throw new Error("ERR_FAILED_TO_FETCH");
    }

    const parsed = await res.json();

    if (!parsed) {
        console.error(`${red}[ERROR]${reset} Failed to fetch dataset. It might be network error or the source isn't available.`)
        throw new Error("ERR_FAILED_TO_FETCH");
    }

    let dataset = [];
    let labels = [];

    for (let i = 0; i < parsed.length; i++) {
        dataset.push(new Float32Array(parsed[i].data));
        labels.push(parsed[i].label);
    }

    return {
        dataset: dataset,
        labels: labels
    }
}


module.exports = {
    mnist_digits
}