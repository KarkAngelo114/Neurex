const WebSocket = require('ws'); 
const { yellow, reset } = require('../../color-code'); 
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

function VisualizerBoard() { 
    let port = 7777; 
    let clients = new Set(); 
    return { 
        initialize: () => {
            console.log('\nInitializing Dashboard....');
            let wss; // initiate once so that static files will be serve on the browser 
            const server = http.createServer((req, res) => { 
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
                    console.log(`\n${yellow}[NOTICE]${reset} 🌐 Visualizer Board running on: http://localhost:${port}🌐\n`); 
                    resolve(); 
                }); 
            }); 
        }, 
        visualize: (data) => { 
            const liveData = JSON.stringify(data); 

            for (const client of clients) { 
                if (client.readyState === WebSocket.OPEN) { 
                    client.send(liveData); 
                } 
            } 
        }, 
        abort: () => { 
            process.exit(0); 
        } 
    } 
} 

module.exports = { VisualizerBoard, }