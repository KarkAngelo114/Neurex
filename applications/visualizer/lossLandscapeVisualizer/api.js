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
    const payload = JSON.parse(event.data);

    console.log('receiving data...');

    if (payload.type !== '3D_LANDSCAPE_DATA') return;

    try {
        const { modelJson, trainX, trainY, lossFunction, resolutionSize, range = 1.0, batchSize } = payload;

        const subsampleX = structuredClone(trainX.slice(0, batchSize));
        const subSampleY = structuredClone(trainY.slice(0, batchSize));

        // 2. Load model into NeurexRuntime
        const runtime = new NeurexRuntime.Runtime();

        // 3. Select loss function from registry
        const lossFunc = lossFunctions[lossFunction.toLowerCase()] || lossFunctions.mse;

        // 4. Compute 3D Landscape Grid on Client
        const { x, y, z } = await computeGridLandscape(runtime, subsampleX, subSampleY, lossFunc, resolutionSize, range, modelJson);

        // 5. Update UI & Plotly
        document.getElementById('learning-rate').innerText = payload.learningRate || '-';
        document.getElementById('optimizer').innerText = payload.optimizer || '-';
        document.getElementById('batch-size').innerText = payload.batchSize || '-';
        document.getElementById('version').innerText = payload.version || '1.0.0';

        const trace = {
            z: z,
            x: x,
            y: y,
            type: 'surface',
            colorscale: 'Rainbow'
        };

        Plotly.react('plot', [trace], { autosize: true });
    } catch (error) {
        // Without this, a failure here (bad payload, shape mismatch, etc.)
        // becomes a silent unhandled rejection and the dashboard just freezes
        // on the previous frame with no indication anything went wrong.
        console.error('[LandscapeVisualizer] Failed to render epoch payload:', error);
    }
};

// Rescales a random direction vector so its L2 norm matches the L2 norm of
// the actual trained weight layer it will perturb. Without this, a single
// fixed alpha/beta range (e.g. -1..1) is meaningless across layers: a layer
// whose real weights are small gets swamped by comparatively huge raw
// random noise (producing spiky, meaningless perturbations), while a layer
// with naturally large weights barely moves. This depends on the *actual
// trained* weight magnitudes, not layer shape, so a shape-based scheme like
// Xavier init can't substitute for it here -- two identically-shaped layers
// can end up with very different trained norms (e.g. one regularized,
// one not), and this model also mixes EmbeddingLayer / recurrent_cell /
// connected_layer types where "fan-in/fan-out" isn't even well-defined the
// way it is for a plain dense layer.
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

/**
 * Grid Calculation Engine on Client Thread
 */
async function computeGridLandscape(runtime, trainX, trainY, lossFunc, resolutionSize, range, originalModelJson) {
    // 1. Extract baseline weights directly from modelJson
    const origW = originalModelJson.weights.map(w => new Float32Array(w));

    // 2. (Re)initialize direction vectors whenever we don't have any yet, or
    //    the model's weight shape no longer matches the ones we generated
    //    previously (e.g. a new/different model started training). Reusing
    //    stale directions against a mismatched shape would silently produce
    //    a meaningless (or crashing) landscape. Each direction is normalized
    //    per-layer to match that layer's actual trained weight norm (see
    //    normalizeDirection above), so alpha/beta represent a comparable,
    //    meaningful step size across every layer instead of raw noise.
    const shapeChanged = !deltaDirs
        || deltaDirs.length !== origW.length
        || deltaDirs.some((d, l) => d.length !== origW[l].length);

    if (shapeChanged) {
        deltaDirs = origW.map(w =>
            normalizeDirection(new Float32Array(w.length).map(() => (Math.random() - 0.5) * 2), w)
        );
        etaDirs = origW.map(w =>
            normalizeDirection(new Float32Array(w.length).map(() => (Math.random() - 0.5) * 2), w)
        );
    }

    const stepSize = (range * 2) / (resolutionSize - 1);
    const xCoord = [], yCoord = [], zGrid = [];

    // 3. `loadSavedModel` refuses to reload once a runtime instance already
    //    has layers built (see neurex-runtime core.js), so calling it again
    //    per grid cell was a no-op after the very first cell -- every
    //    "perturbed" prediction was actually running against the same
    //    unperturbed baseline weights. It also expects snake_case
    //    `input_size` / `input_shape`, while the trainer serializes those as
    //    `inputSize` / `inputShape`, so shape validation inside predict()
    //    would throw on an untouched payload. Normalize once, load once,
    //    then mutate the runtime's own Float32Array weights directly per
    //    grid cell -- this also avoids rebuilding all layers resolutionSize^2
    //    times, which is the more expensive fix anyway.
    const normalizedModelJson = {
        ...originalModelJson,
        input_size: originalModelJson.input_size ?? originalModelJson.inputSize,
        input_shape: originalModelJson.input_shape ?? originalModelJson.inputShape,
    };

    await runtime.loadSavedModel(normalizedModelJson);

    // Fail fast with a clear message if trainX isn't shaped the way
    // predict() expects: an array of samples, each sample itself an array
    // of exactly `input_size` numbers. The most common way this breaks for
    // embedding/sequence models is sending a single flat token sequence
    // (e.g. [3, 588, 12, ...]) instead of an array of sequences
    // (e.g. [[3, 588, 12, ...], [...], ...]) -- in that case `input[i]` is a
    // *number*, not an array, and its `.length` is `undefined`, which is
    // exactly the symptom of "Input size/shape: undefined" from the runtime.
    const expectedInputSize = normalizedModelJson.input_size;
    if (!Array.isArray(trainX) || trainX.length === 0) {
        throw new Error(`trainX must be a non-empty array of samples; got: ${JSON.stringify(trainX)?.slice(0, 100)}`);
    }
    trainX.forEach((sample, idx) => {
        if (!Array.isArray(sample) && !(sample instanceof Float32Array)) {
            throw new Error(
                `trainX[${idx}] is not an array (got ${typeof sample}: ${JSON.stringify(sample)}). ` +
                `Did you send a single flat sequence instead of an array of samples? ` +
                `Expected an array of samples, each with length ${expectedInputSize}.`
            );
        }
        if (sample.length !== expectedInputSize) {
            throw new Error(
                `trainX[${idx}] has length ${sample.length}, but the model's input_size is ${expectedInputSize}. ` +
                `Each sample must be a fixed-length sequence matching input_size / input_shape.`
            );
        }
    });

    for (let i = 0; i < resolutionSize; i++) {
        const alpha = -range + (i * stepSize);
        if (i % 2 === 0) await new Promise(r => setTimeout(r, 0));
        xCoord.push(alpha);
        yCoord.push(alpha);
        zGrid[i] = [];

        for (let j = 0; j < resolutionSize; j++) {
            const beta = -range + (j * stepSize);

            // Perturb weights in place on the loaded runtime: θ = θ* + α·δ + β·η
            for (let l = 0; l < origW.length; l++) {
                for (let k = 0; k < origW[l].length; k++) {
                    runtime.weights[l][k] = origW[l][k] + alpha * deltaDirs[l][k] + beta * etaDirs[l][k];
                }
            }

            // Run prediction feedforward via neurex-runtime
            const preds = await runtime.predict(trainX);

            if (!preds) {
                console.error(`Prediction failed at grid position [${i}, ${j}]`);
                zGrid[i][j] = null;
                continue;
            }

            // Evaluate loss across sample batch
            let batchLoss = 0;
            for (let s = 0; s < preds.length; s++) {
                batchLoss += lossFunc(preds[s], trainY[s]);
            }

            zGrid[i][j] = batchLoss / preds.length;
        }
    }

    // Restore original baseline weights on the runtime
    for (let l = 0; l < origW.length; l++) {
        runtime.weights[l].set(origW[l]);
    }

    return { x: xCoord, y: yCoord, z: zGrid };
}