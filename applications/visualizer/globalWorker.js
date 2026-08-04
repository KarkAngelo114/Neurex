const { parentPort, workerData } = require('worker_threads');
const http = require('http');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');


const port = workerData?.port || 7001;
const staticDir = workerData?.staticDir || __dirname; 

const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png'
};

const clients = new Set();
const history = []; // Stores metric history so refreshed clients get caught up
const limit  = 5; // to avoid massive collection of data, we limit it

const server = http.createServer((req, res) => {
    // 2. Resolve relative to staticDir instead of __dirname
    const targetPath = path.join(staticDir, req.url === "/" ? "index.html" : req.url);
    
    fs.readFile(targetPath, (err, content) => {
        if (err) {
            res.writeHead(err.code === "ENOENT" ? 404 : 500, { 'Content-Type': 'text/plain' });
            res.end(err.code === "ENOENT" ? 'Asset not found' : 'Internal Server Error');
        } else {
            const ext = path.extname(targetPath).toLowerCase();
            res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(port, 'localhost', () => {
    // Tell the main thread that the web server is ready!
    parentPort.postMessage({ type: 'SERVER_READY', port });
});


// initiate web socket server
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    clients.add(ws);

    // Just in case web visualizer is closed or opened late, instantly catch up late/refreshed clients with past data
    if (history.length > 0) {
        ws.send(JSON.stringify({ type: 'HISTORICAL_DATA', payload: history }));
    }

    ws.on('close', () => clients.delete(ws));
});

// Listen for incoming data from the main Neurex training loop
parentPort.on('message', (msg) => {

    if (msg.type === 'VISUALIZE') {
        history.push(msg.payload);

        if (history.length > limit) {
            history.shift(); // Remove the oldest entry
        }

        const liveData = JSON.stringify({ type: 'LIVE_DATA', ...msg.payload });

        for (const client of clients) {
            if (client.readyState === WebSocket.OPEN) {
                client.send(liveData);
            }
        }
    }
    else if (msg.type === 'ABORT') {

        for (const client of clients) {
            client.close(1001, 'Training finished/aborted.');
        }

        clients.clear();
        
        wss.close(() => {
            server.close(() => {
                process.exit(0);
            });
        });
    }
});