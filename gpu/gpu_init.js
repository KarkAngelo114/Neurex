let path = require('path');

const addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));

const detectGPU = () => {
    try {
        return addon.Detect_GPU();
    }
    catch (error) {
        console.error(error);
    }
};

module.exports = {
    detectGPU
}