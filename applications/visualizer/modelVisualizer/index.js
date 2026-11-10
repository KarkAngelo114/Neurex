const { Worker } = require('worker_threads');
const path = require('path');
const { yellow, reset } = require('../../../color-code');

const modelVisualizer = () => {
    let port = 7003; 
    let worker = null;

    return {
        special_type: 'model_visualizer', // this will be use for model visualization even not during training (e.g: you want to inspect a model visually after loading a model)
        type: "visualizers",
        initialize: () => {
            return new Promise((resolve, reject) => {

                const staticDirectory = path.join(__dirname);

                worker = new Worker(path.join(__dirname, "..",'globalWorker.js'), {
                    workerData: { port:port, staticDir: staticDirectory  }
                });

                worker.on('message', (msg) => {
                    if (msg.type === 'SERVER_READY') {
                        console.log(`${yellow}[NOTICE]${reset} 🌐 Model Visualizer running on worker thread: http://localhost:${msg.port} 🌐`);
                        resolve();
                    }
                });

                worker.on('error', reject);
                worker.on('exit', (code) => {
                    if (code !== 0) console.error(`Visualizer Worker stopped with exit code ${code}`);
                });
            });
        },
        visualize: (visualizerData, modelData) => {

            const data = {
                layers: modelData.layers.map((layer) => ({
                    ...JSON.parse(JSON.stringify(layer)),
                    weights: layer.weights ? Array.from(layer.weights) : undefined,
                    biases: layer.biases ? Array.from(layer.biases) : undefined,
                })),
                version: visualizerData.version,
                inputShape: modelData.input_shape,
            };

            if (worker) {
                worker.postMessage({type: "VISUALIZE", payload: data});
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
    }
}


module.exports = { modelVisualizer }