const { GPU } = require('gpu.js');
const gpu = new GPU({mode:'gpu'});

const detect = () => {
    let backend = "Unknown";
    let isGPUAvailable = false;

    try {
        const testKernel = gpu.createKernel(function () {
            return 1;
        }).setOutput([1]);

        const result = testKernel();
        backend = testKernel.kernel.constructor.name;
        isGPUAvailable = backend.includes("WebGL");
    } catch (e) {
        console.warn("[WARNING] GPU backend detection failed:", e.message);
        backend = "CPUKernel";
    }

    return {
        gpu,
        backend,
        isGPUAvailable,
        isSoftwareGPU: backend === "HeadlessGLKernel",
        isFallbackCPU: backend === "CPUKernel"
    };
};


module.exports = detect;
