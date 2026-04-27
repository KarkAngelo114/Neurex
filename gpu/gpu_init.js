let path = require('path');

const addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));

exports.detectGPU = () => addon.Detect_GPU();