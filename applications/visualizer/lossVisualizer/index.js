const { Worker } = require('worker_threads');
const path = require('path');
const { yellow, reset } = require('../../../color-code');

function lossVisualizer() {
    let worker = null;
    const port = 7001;

    return {
        type: "visualizers",
        initialize: () => {
            return new Promise((resolve, reject) => {

                const staticDirectory = path.join(__dirname);

                worker = new Worker(path.join(__dirname, "..",'globalWorker.js'), {
                    workerData: { port:port, staticDir: staticDirectory  }
                });

                worker.on('message', (msg) => {
                    if (msg.type === 'SERVER_READY') {
                        console.log(`${yellow}[NOTICE]${reset} 🌐 Loss Visualizer running on worker thread: http://localhost:${msg.port} 🌐`);
                        resolve();
                    }
                });

                worker.on('error', reject);
                worker.on('exit', (code) => {
                    if (code !== 0) console.error(`Visualizer Worker stopped with exit code ${code}`);
                });
            });
        },

        visualize: (data) => {
            if (worker) {

                worker.postMessage({ type: 'VISUALIZE', payload: data });
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

module.exports = { lossVisualizer };