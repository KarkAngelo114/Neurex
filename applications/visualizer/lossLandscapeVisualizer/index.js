const WebSocket = require('ws'); 
const { yellow, reset } = require('../../../color-code'); 
const http = require('http'); 
const fs = require('fs'); 
const path = require('path'); 

const MIME_TYPES = { 
    '.html': 'text/html', 
    '.css': 'text/css', 
    '.js': 'text/javascript', 
    '.json': 'application/json', 
    '.png': 'image/png', 
    '.jpg': 'image/jpeg' 
}; 

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

    return {
        initialize: () => {
            server = http.createServer((req, res) => { 
                const requestedPath = decodeURIComponent((req.url === "/" ? "index.html" : req.url).split('?')[0]);
                const targetPath = path.join(__dirname, requestedPath);

                if (!targetPath.startsWith(__dirname)) {
                    res.writeHead(403, { 'Content-Type': 'text/plain' });
                    res.end('Forbidden');
                    return;
                }

                fs.readFile(targetPath, (err, content) => { 
                    if (err) { 
                        res.writeHead(404, {'Content-Type': 'text/plain; charset=utf-8'}); 
                        res.end('Dashboard asset not found.');
                    } else { 
                        const ext = path.extname(targetPath).toLowerCase(); 
                        const baseType = MIME_TYPES[ext] || 'application/octet-stream';
                        const isText = baseType.startsWith('text/') || baseType === 'application/json';
                        res.writeHead(200, { 'Content-Type': isText ? `${baseType}; charset=utf-8` : baseType }); 
                        res.end(content, 'utf-8'); 
                    } 
                });
            }); 
                        
            wss = new WebSocket.Server({ server }); 
            wss.on('connection', ws => {
                clients.add(ws);
                ws.on('close', () => clients.delete(ws));
                ws.on('error', () => clients.delete(ws));
            }); 
                                    
            return new Promise(resolve => { 
                server.listen(port, 'localhost', () => { 
                    console.log(`\n${yellow}[NOTICE]${reset} 🌐 Loss Landscape Visualizer running on: http://localhost:${port}🌐\n`); 
                    resolve(); 
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

            const payload = JSON.stringify({
                type: '3D_LANDSCAPE_DATA',
                modelJson: model,
                trainX: sampleX,
                trainY: sampleY,
                lossFunction: data.lossFunction,
                learningRate: data.learningRate,
                optimizer: data.optimizer,
                batchSize: data.totalBatchSize,
                version: data.version,
                resolutionSize: 30,
                current_epoch: data.epoch
            }, typedArrayReplacer);

            for (const client of clients) {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(payload);
                }
            }
        },

        abort: () => {
            return new Promise((resolve, reject) => {
                for (const client of clients) client.close(1001);
                clients.clear(); 
                wss.close(() => server.close(resolve));
            });
        }
    };
}

module.exports = { lossLandscapeVisualizer };