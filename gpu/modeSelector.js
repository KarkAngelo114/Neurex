const { reset, yellow, red } = require('../color-code');
const {detectGPU} = require('../gpu/gpu_init');
const data = detectGPU();
let hasGPU = false;
let force_Use_Default_JS_Float32_Module = false;


exports.modeConfiguration = (value) => {
    

    // if auto
    if (value.toLowerCase() === "auto") {
        if (data.devices[0].hostUnifiedMemory) {
            console.warn(`\n${yellow}[INFO]${reset} GPU compute is not available on this environment. Switching to CPU fallback...`);
            hasGPU = false;
            return;
        }

        hasGPU = true;
    }

    if (value.toLowerCase() === "gpu") {
        if (force_Use_Default_JS_Float32_Module) {
            throw new Error("[ERROR] Cannot detect if 'force_Use_Default_JS_Float32_Module' is true. In your configure(), ensure that 'onFloat32Module' is false");
        }

        if (data.devices[0].hostUnifiedMemory && value.toLowerCase() === "gpu") {
            throw new Error(`${red}[ERROR]${reset} Cannot use mode:"gpu" if host unified memory is true. This error can be avoided if you will only set mode:"gpu" if GPU is actually available, otherwise set mode:"cpu" or mode:"auto"`);
        }

        if (!data.devices[0].hostUnifiedMemory && value.toLowerCase() === "gpu") hasGPU = true;

        return;
    }

    if (value.toLowerCase() === "cpu") {
        hasGPU = false;
        return;
    }

    throw new Error(`${red}[ERROR] Invalid mode: ${value.toLowerCase()}. Use "gpu", "cpu" or "auto" only${reset}`);
}

exports.onFloat32Module = (value) => {

    if (value && hasGPU) {
        console.log(`${yellow}\n[WARN] Forcing to use default float32 module on JS${reset}`);
        hasGPU = false;
    }

    force_Use_Default_JS_Float32_Module = value;
};



exports.BooleanAvailability = () => {
    return {
        hasGPU, 
        force_Use_Default_JS_Float32_Module,
        data
    }
}