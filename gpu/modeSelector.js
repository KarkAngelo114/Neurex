const { reset, red, gray } = require('../color-code');
const { OpenCL_detectGPU } = require('../gpu/gpu_init');

let selectedDevice = null;
let computeBackend = null;
let isBackendSet = false;

// Vendors whose GPUs are shared-memory / integrated by design
const INTEGRATED_NAME_HINTS = /\b(UHD|Iris|HD Graphics|Radeon\(TM\) Graphics|Vega \d+ Graphics)\b/i;

/**
 * "Dedicated" = has its own VRAM. hostUnifiedMemory is the primary signal,
 * with a name-based sanity check because drivers report it inconsistently.
 */
const isDedicated = (d) => !d.hostUnifiedMemory && !INTEGRATED_NAME_HINTS.test(d.gpu);

/** Returns dedicated GPUs sorted best → worst (VRAM, then compute units, then clock). */
const OpenCL_rankDedicatedDevices = (devices = [], findBest) => {
    if (!findBest) {
        // if user pass `findBest` as false, no need to filter out for best devices. Return device list and let the `OpenCL_resolveBestDevice`
        // choose the first indexed device (this could be an iGPU)
        return devices;
    }

    return devices.filter(isDedicated).sort((a, b) => {
            if (a.globalMemBytes !== b.globalMemBytes) {
                return a.globalMemBytes > b.globalMemBytes ? -1 : 1;   // BigInt-safe compare
            }
            if (a.computeUnits !== b.computeUnits) return b.computeUnits - a.computeUnits;
            return b.maxClockMHz - a.maxClockMHz;
        });
}
    

/** 
* Detect + rank in one call. Never throws. OpenCL is vendoer agnostic, so we can let users select for best compute device or just use 
* whatever it is index first in the list of device (that includes iGPU) based on the findBest boolean this.state.
* @param {Boolean} findBest
* 
*/
const OpenCL_resolveBestDevice = (findbest) => {
    const data = OpenCL_detectGPU();
    if (!data?.ok) {
        return { 
            data: data, 
            best: null, 
            ranked: [] 
        }
    };

    const ranked = OpenCL_rankDedicatedDevices(data.devices, findbest);

    return { 
        data: data, 
        best: ranked[0] ?? null, 
        ranked: ranked
    };
};

const OpenCL_findBestDevice = (findBest) => {
    const { data, best } = OpenCL_resolveBestDevice(findBest);

    if (!best) {
        const reason = data?.ok === false ? `OpenCL error: ${data.error}` : `No dedicated GPU found. To use 'any' type of compute in OpenCL, disable 'findBestDevice' to 'false'. Read: ${gray}https://neurex-documentation.vercel.app/javascript-nodejs${reset} for info.`;

        console.error(reason);
        throw new Error("ERR_OPENCL_ERROR");
    }

    selectedDevice = best;
}

/**
 * 
 * @param {String} stringValue 
 * @param {Boolean} findBestDevice
 */
const setComputeBackend = (stringValue = "cpu", findBestDevice = true) => {

    if (!typeof stringValue === "string" || !stringValue instanceof String) {
        console.error('[ERROR] "type" is not a type of string');
        throw new Error("ERR_TYPE_ERROR");
    }

    if(!stringValue) {
        console.error("[ERROR] 'type' cannot be null, empty string or undefined");
        throw new Error(('ERR_TYPE_UNDEFINED'));
    }


    // using configure() on the Neurex class to set mode for GPU compute is now pointless if the underlying internal functions are at module-level
    // and allowing custom training loop since any instance shares the same module-level functions and state, like if one model configure for GPU and the other is
    // configure to CPU, the first state will be overridden and make all training on CPU only. To solve this, this function will acts as the
    // environment compute setter where users can set a device rather than configuring them on the configure, and this will affect all training instances since
    // it uses global shared internal state ("cpu",  "opencl", "pure-js", and possibly in the future, "cuda")

    if (isBackendSet) {
        console.error(`${red}[ERROR]${reset} Compute backend is already set. You can only set one time across training instances.`);
        throw new Error("ERR_COMPUTE_BACKEND_SETTING");
    }

    if (stringValue.toLowerCase() === "cpu" || stringValue.toLowerCase() === "pure-js") {
        computeBackend = stringValue;
        selectedDevice = null;
    }
    else if (stringValue.toLowerCase() === "opencl") {
        OpenCL_findBestDevice(findBestDevice);
        computeBackend = stringValue;
    }
    else {
        console.error(`"${stringValue}" is currently not supported in Neurex yet. Use "cpu", "pure-js" or "opencl" only`);
        throw new Error("ERR_UNSUPPORTED_COMPUTE_BACKEND");
    }

    isBackendSet = true;
}

/**
 * 
 * @returns {{computeBackend: string, devices: any}}
 */
const globalState = () => {

    if (!isBackendSet) {
        console.error(`${red}[ERROR]${reset} Compute backend is not yet initialized. Call "setComputeBackend()" first. Read: ${gray}https://neurex-documentation.vercel.app/javascript-nodejs${reset} for info.`);
        throw new Error("ERR_COMPUTE_BACKEND_NOT_SET");
    }

    return {
        computeBackend: computeBackend,
        device: selectedDevice
    }
}


module.exports = {
    setComputeBackend,
    globalState
}