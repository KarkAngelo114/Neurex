const WebSocket = require('ws'); 
const { yellow, reset } = require('../../../color-code'); 
const http = require('http'); const fs = require('fs'); 
const path = require('path'); 

const MIME_TYPES = { 
    '.html': 'text/html', 
    '.css': 'text/css', 
    '.js': 'text/javascript', 
    '.json': 'application/json', 
    '.png': 'image/png', 
    '.jpg': 'image/jpeg', 
    '.gif': 'image/gif', 
    '.svg': 'image/svg+xml', 
    '.ico': 'image/x-icon' 
}; 

const modelVisualizer = () => {
    let port = 7003; 
    let clients = new Set(); 
    let server;
    let wss; 

    return {
        initialize: () => {
            // initiate once so that static files will be serve on the browser 
            server = http.createServer((req, res) => { 
                const targetPath = path.join(__dirname, req.url === "/" ? "index.html" : req.url); 
                fs.readFile(targetPath, (err, content) => { 
                    if (err) { 
                        if (err.code === "ENOENT") { 
                            res.writeHead(404, {'Content-Type':'text/plain'}); 
                            res.end('Dashboard asset not found :( Please create an issue on: https://github.com/KarkAngelo114/Neurex/issues') 
                        }
                        else { 
                            res.writeHead(500, {'Content-Type':'text/plain'}); 
                            res.end("I've messed up, please report this issue on: https://github.com/KarkAngelo114/Neurex/issues") 
                        } 
                    } 
                    else { 
                        const ext = path.extname(targetPath).toLowerCase(); 
                        const contentType = MIME_TYPES[ext] || 'application/octet-stream'; 
                        res.writeHead(200, { 'Content-Type': contentType }); 
                        res.end(content, 'utf-8'); 
                    } 
                }); 
            }); 
                        
            wss = new WebSocket.Server({ server }); 
            
            wss.on('connection', ws => { 
                clients.add(ws);
            
                ws.on('close', () => { 
                    clients.delete(ws); 
                }) 
            }); 
                                    
            return new Promise(resolve => { 
                server.listen(port, 'localhost', () => { 
                    console.log(`\n${yellow}[NOTICE]${reset} 🌐  Model Visualizer running on: http://localhost:${port}🌐\n`); 
                    resolve(); 
                }); 
            }); 
        },
        visualize: (visualizerData, modelData) => {

            const data = {
                layers: modelData.layers,
                weights: modelData.weights,
                biases: modelData.biases,
                version: visualizerData.version
            }

            const liveData = JSON.stringify(data);

            for (const client of clients) {
                if (client.readyState === WebSocket.OPEN) { 
                    client.send(liveData); 
                } 
            }
        },
        abort: () => {
            
            return new Promise((resolve, reject) => {

                for (const client of clients) {
                    client.close(1001, `${yellow}[NOTICE]${reset} Server shutting down...`);  
                }

                clients.clear(); 

                wss.close((err) => {
                    if (err) console.error('Error closing WSS:', err);


                    server.close((serverErr) => {
                        if (serverErr) {
                            return reject(serverErr);
                        }

                        console.log(`\n${yellow}[NOTICE]${reset}Visualizer server completely stopped.`);
                        resolve();
                    });
                });
            });
        }
    }
}


module.exports = { modelVisualizer }