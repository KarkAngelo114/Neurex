const { Worker } = require('worker_threads');
const path = require('path');
const { yellow, reset } = require('../../../color-code');

// core.js stores weights, biases, and each trainX/trainY row as
// Float32Array, not plain Array. JSON.stringify serializes a TypedArray as
// a plain object keyed by index ({"0":1,"1":2,...}), not as a JSON array --
// so without this replacer, every payload silently loses its "arrayness"
// on the way to the client. This is a bigger problem for weights/biases
// than trainX: `new Float32Array({0:1,1:2})` doesn't throw, it just quietly
// returns an empty Float32Array, so a corrupted model would fail silently
// instead of loudly.
function typedArrayReplacer(key, value) {
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
        return Array.from(value);
    }
    return value;
}

function lossLandscapeVisualizer({ renderEveryTargetEpoch = 100 } = {}) {
    let port = 7002; 
    let clients = new Set(); 
    let server;
    let wss;
    let worker = null;

    return {
        initialize: () => {
            return new Promise((resolve, reject) => {
            
                const staticDirectory = path.join(__dirname);
            
                worker = new Worker(path.join(__dirname, "..",'globalWorker.js'), {
                    workerData: { port: port, staticDir: staticDirectory  }
                });
            
                worker.on('message', (msg) => {
                    if (msg.type === 'SERVER_READY') {
                        console.log(`${yellow}[NOTICE]${reset} 🌐 Loss Landccape Visualizer running on worker thread: http://localhost:${msg.port} 🌐`);
                        resolve();
                    }
                });
            
                worker.on('error', reject);
                worker.on('exit', (code) => {
                    if (code !== 0) console.error(`Visualizer Worker stopped with exit code ${code}`);
                });
            });
        },

        visualize: async function(data, model, sampleX, sampleY) {
            if (
                data.epoch !== 1 &&
                renderEveryTargetEpoch &&
                (data.epoch % renderEveryTargetEpoch !== 0)
            ) {
                return;
            }

            // Parse it back to an object after applying typedArrayReplacer so postMessage handles it cleanly:
            const payloadObject = JSON.parse(
                JSON.stringify({
                    type: '3D_LANDSCAPE_DATA',
                    modelJson: model,
                    trainX: sampleX,
                    trainY: sampleY,
                    lossFunction: data.lossFunction,
                    learningRate: data.learningRate,
                    optimizer: data.optimizer,
                    batchSize: data.totalBatchSize,
                    version: data.version,
                    resolutionSize: 21,
                    current_epoch: data.epoch
                }, typedArrayReplacer)
            );

            if (worker) {
                // Send payload as an object, NOT a raw JSON string
                worker.postMessage({ type: "VISUALIZE", payload: payloadObject });
            }
        },

        abort: () => {
           return new Promise((resolve) => {
                if (!worker) return resolve();
                
                worker.postMessage({ type: 'ABORT' });
                worker.on('exit', () => {
                    console.log(`\n${yellow}[NOTICE]${reset} Visualizer server completely stopped.`);
                    resolve();
                });
            });
        }
    };
}

module.exports = { lossLandscapeVisualizer };