const ws = new WebSocket(`ws://${location.host}`);


ws.onmessage = async (event) => {
    const {layers, weights, biases, version} = JSON.parse(event.data);
    
    document.getElementById('version').innerText = `Neurex v${version}` || '1.0.0';
    console.log(layers);
}