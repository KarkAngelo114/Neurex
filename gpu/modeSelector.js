const { reset, yellow, red } = require('../color-code');
const { detectGPU } = require('../gpu/gpu_init');

let hasGPU = false;
let selectedDevice = null;
let force_Use_Default_JS_Float32_Module = false;

// Vendors whose GPUs are shared-memory / integrated by design
const INTEGRATED_NAME_HINTS = /\b(UHD|Iris|HD Graphics|Radeon\(TM\) Graphics|Vega \d+ Graphics)\b/i;

/**
 * "Dedicated" = has its own VRAM. hostUnifiedMemory is the primary signal,
 * with a name-based sanity check because drivers report it inconsistently.
 */
const isDedicated = (d) => !d.hostUnifiedMemory && !INTEGRATED_NAME_HINTS.test(d.gpu);

/** Returns dedicated GPUs sorted best → worst (VRAM, then compute units, then clock). */
const rankDedicatedDevices = (devices = []) =>
    devices
        .filter(isDedicated)
        .sort((a, b) => {
            if (a.globalMemBytes !== b.globalMemBytes) {
                return a.globalMemBytes > b.globalMemBytes ? -1 : 1;   // BigInt-safe compare
            }
            if (a.computeUnits !== b.computeUnits) return b.computeUnits - a.computeUnits;
            return b.maxClockMHz - a.maxClockMHz;
        });

/** Detect + rank in one call. Never throws. */
const resolveBestDevice = () => {
    const data = detectGPU();
    if (!data?.ok) return { data, best: null, ranked: [] };
    const ranked = rankDedicatedDevices(data.devices);
    return { data, best: ranked[0] ?? null, ranked };
};

exports.modeConfiguration = (value) => {
    const targetMode = String(value).toLowerCase();

    if (!["auto", "gpu", "cpu"].includes(targetMode)) {
        throw new Error(`${red}[ERROR] Invalid mode: ${targetMode}. Use "gpu", "cpu" or "auto" only${reset}`);
    }

    if (targetMode === "cpu") {
        hasGPU = false;
        selectedDevice = null;
        return;
    }

    if (targetMode === "gpu" && force_Use_Default_JS_Float32_Module) {
        throw new Error(`${red}[ERROR]${reset} Cannot use GPU mode when force_Use_Default_JS_Float32_Module is true.`);
    }

    const { data, best } = resolveBestDevice();

    if (!best) {
        const reason = data?.ok === false
            ? `OpenCL error: ${data.error}`
            : "No dedicated GPU found (integrated GPUs are not supported for compute)";

        if (targetMode === "gpu") {
            throw new Error(`${red}[ERROR]${reset} mode:"gpu" requested but ${reason}. Use mode:"cpu" or mode:"auto".`);
        }

        console.warn(`\n${yellow}[INFO]${reset} GPU compute unavailable (${reason}). Falling back to CPU...`);
        hasGPU = false;
        selectedDevice = null;
        return;
    }

    hasGPU = true;
    selectedDevice = best;
};

exports.onFloat32Module = (value) => {
    if (value) {
        console.log(`${yellow}[INFO]${reset} Forcing to use default float32 module on JS.`);
        hasGPU = false;
        selectedDevice = null;
    }
    force_Use_Default_JS_Float32_Module = value;
};

exports.BooleanAvailability = () => ({
    hasGPU,
    force_Use_Default_JS_Float32_Module,
    device: selectedDevice,   // includes .index → pass to Init_GPU(kernelPath, device.index)
});