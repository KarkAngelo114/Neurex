const ws = new WebSocket(`ws://${location.host}`);
let deltaDirs = null;
let etaDirs = null;
const EPSILON = 1e-15;

const lossFunctions = {
    mse: (predictions, actuals) => {
        let occurrence = predictions.length;
        let sum = 0;
        for (let i = 0; i < occurrence; i++) {
            let difference = predictions[i] - actuals[i];
            sum += difference * difference;
        }
        return sum / occurrence;
    },

    mae: (predictions, actuals) => {
        let occurrence = predictions.length;
        let sum = 0;
        for (let i = 0; i < occurrence; i++) {
            sum += Math.abs(predictions[i] - actuals[i]);
        }
        return sum / occurrence;
    },

    categorical_cross_entropy: (predictions, actuals, epsilon = EPSILON) => {
        let loss = 0;
        for (let i = 0; i < predictions.length; i++) {
            loss -= actuals[i] * Math.log(Math.max(predictions[i], epsilon));
        }
        return loss;
    },

    sparse_categorical_cross_entropy: (predictions, actuals, epsilon = EPSILON) => {
        // actuals[0] contains the class index
        const p = Math.max(predictions[actuals[0]], epsilon);
        return -Math.log(p);
    },

    binary_cross_entropy: (predictions, actuals, epsilon = EPSILON) => {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = Math.max(Math.min(predictions[i], 1 - epsilon), epsilon);
            sum -= actuals[i] * Math.log(p) + (1 - actuals[i]) * Math.log(1 - p);
        }
        return sum / predictions.length;
    }
};




ws.onmessage = async (event) => {
    const rawMsg = JSON.parse(event.data);
    
    // Extract real payload whether sent live or from historical replay
    const payload = rawMsg.payload || rawMsg; 
    
    if (payload.type !== '3D_LANDSCAPE_DATA') return;

    try {
        const { modelJson, trainX, trainY, lossFunction, resolutionSize, range = 1.0 } = payload;

        let maxSampleSize = trainX.length > 30 ? 30 : trainX.length; // cap max sample size if samples greater than 30, otherwise, use the sample size of trainX

        const subsampleX = structuredClone(trainX.slice(0, maxSampleSize));
        const subSampleY = structuredClone(trainY.slice(0, maxSampleSize));
        const runtime = new NeurexRuntime.Runtime();
        const lossFunc = lossFunctions[lossFunction.toLowerCase()] || lossFunctions.mse;

        // 1. Pre-calculate X and Y axis coordinate ranges
        const stepSize = (range * 2) / (resolutionSize - 1);
        const xCoord = Array.from({ length: resolutionSize }, (_, i) => -range + (i * stepSize));
        const yCoord = Array.from({ length: resolutionSize }, (_, j) => -range + (j * stepSize));

        // 2. Initialize a 2D array filled with `null`
        const zGrid = Array.from({ length: resolutionSize }, () => new Array(resolutionSize).fill(0));

        // 3. Render the initial empty Plotly surface
        const trace = {
            z: zGrid,
            x: xCoord,
            y: yCoord,
            type: 'surface',
            colorscale: 'Rainbow',
            lighting: { ambient: 0.8, diffuse: 0.8, fresnel: 0.2, specular: 0.5, roughness: 0.5 },
            contours: { z: { show: true, usecolormap: true } }  
        };

        const layout = {
            autosize: true,
            scene: {
                xaxis: { tickformat: 'd' }, // Forces standard digits/hyphens
                yaxis: { tickformat: 'd' },
                zaxis: { tickformat: 'd' }
            }
        };

        Plotly.react('plot', [trace], layout, { autosize: true });

        // 4. Update UI labels
        document.getElementById('learning-rate').innerText = payload.learningRate || '-';
        document.getElementById('optimizer').innerText = payload.optimizer || '-';
        document.getElementById('batch-size').innerText = payload.batchSize || '-';
        document.getElementById('version').innerText = `Neurex v${payload.version}` || '1.0.0';
        document.getElementById('epoch').innerText = payload.current_epoch || 0;

        // 5. Start progressive rendering pipeline
        await computeGridLandscapeProgressive(
            runtime, 
            subsampleX, 
            subSampleY, 
            lossFunc, 
            resolutionSize, 
            range, 
            modelJson, 
            zGrid, 
            xCoord, 
            yCoord
        );

    } catch (error) {
        console.error('[LandscapeVisualizer] Failed to render epoch payload:', error);
    }
};

function normalizeDirection(direction, weights) {
    let weightNormSq = 0;
    let dirNormSq = 0;
    for (let k = 0; k < weights.length; k++) {
        weightNormSq += weights[k] * weights[k];
        dirNormSq += direction[k] * direction[k];
    }
    const weightNorm = Math.sqrt(weightNormSq);
    const dirNorm = Math.sqrt(dirNormSq);
    if (dirNorm === 0) return direction;

    const scale = weightNorm / dirNorm;
    for (let k = 0; k < direction.length; k++) {
        direction[k] *= scale;
    }
    return direction;
}

async function computeGridLandscapeProgressive(runtime, trainX, trainY, lossFunc, resolutionSize, range, originalModelJson, zGrid, xCoord, yCoord) {
    const origW = originalModelJson.weights.map(w => new Float32Array(w));

    // Handle direction vectors
    const shapeChanged = !deltaDirs || deltaDirs.length !== origW.length || deltaDirs.some((d, l) => d.length !== origW[l].length);
    if (shapeChanged) {
        deltaDirs = origW.map(w => normalizeDirection(new Float32Array(w.length).map(() => (Math.random() - 0.5) * 2), w));
        etaDirs = origW.map(w => normalizeDirection(new Float32Array(w.length).map(() => (Math.random() - 0.5) * 2), w));
    }

    const normalizedModelJson = {
        ...originalModelJson,
        input_size: originalModelJson.input_size ?? originalModelJson.inputSize,
        input_shape: originalModelJson.input_shape ?? originalModelJson.inputShape,
    };

    await runtime.loadSavedModel(normalizedModelJson);

    const CHUNK_SIZE = 2; // Number of rows to process before updating the plot

    for (let i = 0; i < resolutionSize; i++) {
        const alpha = xCoord[i];

        for (let j = 0; j < resolutionSize; j++) {
            const beta = yCoord[j];

            // Perturb weights
            for (let l = 0; l < origW.length; l++) {
                for (let k = 0; k < origW[l].length; k++) {
                    runtime.weights[l][k] = origW[l][k] + alpha * deltaDirs[l][k] + beta * etaDirs[l][k];
                }
            }

            const preds = await runtime.predict(trainX);
            if (!preds) {
                zGrid[i][j] = null;
                continue;
            }

            let batchLoss = 0;
            for (let s = 0; s < preds.length; s++) {
                batchLoss += lossFunc(preds[s], trainY[s]);
            }

            // Optional cap at 25 to match your existing colorbar scale cleanly
            zGrid[i][j] = Math.min(batchLoss / preds.length, 25);
        }

        // --- PROGRESSIVE RENDER STEP ---
        // Render every `CHUNK_SIZE` rows, or on the final row
        if (i % CHUNK_SIZE === 0 || i === resolutionSize - 1) {
            // Plotly.restyle efficient update: only re-passes matrix data without resetting camera view
            Plotly.restyle('plot', { z: [zGrid] });
            
            // Yield execution to the browser event loop so UI stays smooth & rotatable!
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }

    // Restore original baseline weights
    for (let l = 0; l < origW.length; l++) {
        runtime.weights[l].set(origW[l]);
    }
}