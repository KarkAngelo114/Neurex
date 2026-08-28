import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FontLoader } from "three/addons/loaders/FontLoader.js";
import { TextGeometry } from "three/addons/geometries/TextGeometry.js";

const ws = new WebSocket(`ws://${location.host}`);
const viewer = document.getElementById('renderer');
const tooltip = document.getElementById('tooltip');
const list = document.getElementById('list');
const gridController = document.getElementById('gridControllerButton');


let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let modelGroup = null;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let hoveredObject = null;
let originalEmissive = null;
let dataFlowGroup = null;
let isEnableGrid = true;
const size = 5000;
const divisions = 500;
const gridHelper = new THREE.GridHelper(size, divisions);
const fontLoader = new FontLoader();
const reshapeFont = fontLoader.loadAsync(
    "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/fonts/helvetiker_regular.typeface.json"
);

let clock = new THREE.Clock();
let animatedLayers = []; // Track objects that have a sliding window

let height = viewer.clientHeight;
let width = viewer.clientWidth;

(() => {
    createScene();
    animate();
    gridDisplayController();
})();

ws.onmessage = async (event) => {
    const wsData = JSON.parse(event.data);

    let data = null;
    let layers = [];
    let weights = [];
    let biases = [];

    if (wsData.type === "HISTORICAL_DATA") {
        data = wsData.payload.at(-1);
    } else {
        data = wsData;
    }

    layers = data.layers;
    weights = data.weights;
    biases = data.biases;

    document.getElementById('inputShape').innerText = `[${data.inputShape}]`;
    document.getElementById('numLayers').innerText = layers.length;

    renderModel(layers, weights, biases);

    listLayers(layers);


}

function renderModel(layer_data, weights, biases) {

    modelGroup.clear();
    animatedLayers = []; // Reset tracked layers
    
    let pointer = 0; // if isParametric ? pointer++ : pointer;
    let currentZ = 0; 
    const baseGap = 15;

    layer_data.forEach((layer, index) => {
        let cube = null;
        let windowWidth = 1;
        let windowHeight = 1;
        let strideX = 1;
        let strideY = 1;
        
        if (layer.layer_name === "Convolutional Layer" || layer.layer_name === "Max Pooling" || layer.layer_name === "Trans Convolution" ) {
            const [iH, iW, iD] = layer.inputShape;

            const visualWidth = Math.min(iW, 224);
            const visualHeight = Math.min(iH, 224);
            const visualDepth = Math.max(Math.min(iD, 512), 4);
            

            const spatialLayerName = layer.layer_name;

            cube = createCube({
                height: visualHeight, 
                width: visualWidth, 
                depth: visualDepth, 
                color: spatialLayerName === "Convolutional Layer" ? 0x4f8cff : spatialLayerName === "Trans Convolution" ? 0x00d52e : 0xFFAE00,
                outlineColor: spatialLayerName === "Convolutional Layer" ? 0x00ffff : spatialLayerName === "Trans Convolution" ? 0x6af488 : 0xFFAE00, 
                opacity: 0.3
            });

            // Determine kernel / pool dimensions & strides
            if (spatialLayerName === "Convolutional Layer" || spatialLayerName === "Trans Convolution") {
                const k = layer.kernel_size || layer.kernelSize || [3, 3];
                windowWidth = Array.isArray(k) ? k[1] : k;
                windowHeight = Array.isArray(k) ? k[0] : k;
            } else {
                const p = layer.poolSize || layer.pool_size || [2, 2];
                windowWidth = Array.isArray(p) ? p[1] : p;
                windowHeight = Array.isArray(p) ? p[0] : p;
            }

            const s = layer.strides; // an integer (eg. 1, 2, 4, and so on)
            strideY = s;
            strideX = s;

            // Positioning Z axis
            if (index > 0) currentZ += visualDepth / 2;
            else currentZ = visualDepth / 2;

            cube.position.set(0, 0, currentZ);
            currentZ += visualDepth / 2 + baseGap;

            // Add sliding window indicator
            const windowMesh = createSlidingWindow(windowWidth, windowHeight, visualDepth, 0xffffff);
            cube.add(windowMesh);

            animatedLayers.push({
                parentCube: cube,
                windowMesh: windowMesh,
                boundsW: visualWidth,
                boundsH: visualHeight,
                winW: windowWidth,
                winH: windowHeight,
                strideX: strideX,
                strideY: strideY
            });

            modelGroup.add(cube);

            const conv = 'Allows you to add convolutional layers <br/> in your model architecture in sequential building.';
            const transconv = "transConv (or transpose convolution) is a <br/> specialized convolutional layer that upsamples incoming tensor <br/>map, which does the opposite of the normal convolution"
            const maxpool = "is use for downsampling operation that reduces the spatial <br/> dimensions of an input tensor by taking the maximum <br/> value over a defined sliding window"; 

            // 1. Store metadata inside the cube mesh when creating it inside renderModel()
            const layerData = {
                name: spatialLayerName,
                inputShape: layer.inputShape || `Size: ${layer.layer_size}`,
                desc: spatialLayerName === "Convolutional Layer" ? conv : spatialLayerName === "Trans Convolution" ? transconv : maxpool,
                outputShape: layer.outputShape,
            };

            if (layer.isParametric) {
                layerData.weight_L1 = L1_Mean(weights[pointer]);
                layerData.bias_L1 = L1_Mean(biases[pointer]);
                pointer++;
            }

            cube.userData = layerData;
        }
        else if (layer.layer_name === "Reshape") {
            const [iH, iW, iD] = layer.inputShape;
            const [oH, oW, oD] = layer.outputShape;
            const rawSize = layer.outputShape.reduce((acc, val) => acc * val , 1);
            const visualWidth = Math.min(Math.max(rawSize / 4, 2), 30); 
            const visualHeight = 1;
            const visualDepth = 1;


            cube = createCube({
                width: 30,
                height: 1, 
                depth: 10,
                color: 0xFFFFFF,
                outlineColor: 0xFFFFFF,
                opacity: 0.5
            });

            

            if (index > 0) currentZ += visualDepth / 2;
            else currentZ = visualDepth / 2;

            cube.position.set(0, 0, currentZ); 
            currentZ += visualDepth / 2 + baseGap;

            modelGroup.add(cube);

            addEngravedLabel(cube, "RESHAPE", 30, 10);

            const layerData = {
                name: "Reshape",
                inputShape: layer.inputShape,
                outputShape: layer.outputShape,
                desc: "The <b>Reshape</b> layer changes the dimensions (shape) of the data passing through <br/> it without changing the data values. This acts as the `connector` to bridge data from different <br/> layers (e.g: from connected layer to convolutional layer)",
            };

            cube.userData = layerData;
        }
        else if (layer.layer_name === "Connected Layer") {
            const rawSize = layer.layer_size;
            const visualWidth = Math.min(Math.max(rawSize / 4, 2), 30); 
            const visualHeight = 1;
            const visualDepth = 1;

            cube = createCube({
                height: visualHeight, 
                width: visualWidth, 
                depth: visualDepth, 
                color: 0xFF3300,
                outlineColor: 0xFF3300, 
                opacity: 0.5
            });

            if (index > 0) currentZ += visualDepth / 2;
            else currentZ = visualDepth / 2;

            cube.position.set(0, 0, currentZ); 
            currentZ += visualDepth / 2 + baseGap;

            // 1x1 sliding window with stride = 1 for dense layers
            const windowMesh = createSlidingWindow(1, 1, visualDepth + 0.1, 0xffffff);
            cube.add(windowMesh);

            animatedLayers.push({
                parentCube: cube,
                windowMesh: windowMesh,
                boundsW: visualWidth,
                boundsH: visualHeight,
                winW: 1,
                winH: 1,
                strideX: 1,
                strideY: 1
            });

            modelGroup.add(cube);

            const layerData = {
                name: layer.layer_name,
                inputShape: layer.inputShape,
                outputShape: layer.outputShape,
                desc: "Allows you to build a layer with number of neurons and the <br>activation function to use in a layer. Stacking more layers <br> will build connected layers or multilayer perceptron",
            };

            if (layer.isParametric) {
                layerData.weight_L1 = L1_Mean(weights[pointer]);
                layerData.bias_L1 = L1_Mean(biases[pointer]);
                pointer++;
            }

            cube.userData = layerData;
        }
        else if (layer.layer_name === "Embedding Layer") {
            // Determine 3D cube dimensions (Width: EmbeddingDim, Height: Sequence Length)
            const visualWidth = Math.min(Math.max(layer.embeddingDim / 2, 2), 30);
            const visualHeight = Math.min(Math.max(layer.maxSequenceLength / 2, 2), 30);
            const visualDepth = 2;

            cube = createCube({
                height: visualHeight,
                width: visualWidth,
                depth: visualDepth,
                color: 0x9B51E0, // Purple for Embeddings
                outlineColor: 0xBB6BD9,
                opacity: 0.6
            });

            if (index > 0) currentZ += visualDepth / 2;
            else currentZ = visualDepth / 2;

            cube.position.set(0, 0, currentZ);
            currentZ += visualDepth / 2 + baseGap;

            const layerData = {
                name: layer.layer_name,
                inputShape: [1, 1, layer.maxSequenceLength],
                outputShape: [1, 1, layer.embeddingDim, layer.maxSequenceLength],
                desc: "Maps discrete token IDs into dense continuous vector representations using a trainable lookup table.",
            };

            if (layer.isParametric) {
                layerData.weight_L1 = L1_Mean(weights[pointer]);
                layerData.bias_L1 = L1_Mean(biases[pointer]);
                pointer++;
            }

            cube.userData = layerData;

            const numRows = Math.min(Math.max(Math.round(layer.maxSequenceLength), 2), 20);
            const rowHeight = visualHeight / numRows;
            const windowMesh = createSlidingWindow(visualWidth, rowHeight, visualDepth + 0.1, 0xffffff);
            cube.add(windowMesh);

            animatedLayers.push({
                parentCube: cube,
                windowMesh: windowMesh,
                boundsW: visualWidth,
                boundsH: visualHeight,
                winW: visualWidth,   // full width -> no horizontal movement
                winH: rowHeight,
                strideX: visualWidth,
                strideY: rowHeight
            });

            modelGroup.add(cube);
        } 
        else if (layer.layer_name === "Recurrent Cell") {
            const visualWidth = Math.min(Math.max(layer.units / 2, 2), 30);
            const visualHeight = 4;
            const visualDepth = 4;
            const zExtent = visualWidth;

            cube = createCube({
                height: visualHeight,
                width: visualWidth,
                depth: visualDepth,
                color: 0xFF69B4,
                outlineColor: 0xFF69B4,
                opacity: 0.6
            });

            if (index > 0) currentZ += zExtent / 2;
            else currentZ = zExtent / 2;
            
            cube.rotation.y = Math.PI / 2;
            cube.position.set(0, 0, currentZ);
            currentZ += zExtent / 2 + baseGap;

            const recurrenceLoop = createRecurrenceLoop(visualWidth, visualHeight, visualDepth);
            cube.add(recurrenceLoop);

            animatedLayers.push({
                type: "rnn",
                parentCube: cube,
                recurrenceLoop: recurrenceLoop,
                boundsW: zExtent,
                boundsH: visualHeight,
                boundsD: visualDepth
            });

            const layerData = {
                name: layer.layer_name,
                inputShape: layer.inputShape,
                outputShape: layer.outputShape,
                desc: "is the fundamental building block of a Recurrent Neural Network (RNN) <br/> designed to process sequential data. It maintains an internal `memory` by <br/> taking its output from the previous time step and feeding <br/> it back into itself alongside the new input.",
            };

            if (layer.isParametric) {
                layerData.weight_L1 = L1_Mean(weights[pointer]);
                layerData.bias_L1 = L1_Mean(biases[pointer]);
                pointer++;
            }


            cube.userData = layerData;

            modelGroup.add(cube);
        }
        else if (layer.layer_name === "Simple Attention" || layer.layer_name === "Multi Head Attention") {
            const visualWidth = Math.min(Math.max(layer.embedDim / 2, 2), 30);
            const visualHeight = Math.min(Math.max(layer.seqLen / 2, 2), 30);
            const numHeads = layer.numHeads;
            const visualDepth = 2;

            cube = createCube({
                height: visualHeight,
                width: visualWidth,
                depth: visualDepth,
                color: 0x6b6d6e,
                outlineColor: 0xf7fbfc,
                opacity: 0.3
            });

            if (index > 0) currentZ += visualDepth / 2;
            else currentZ = visualDepth / 2;
           

            cube.position.set(0, 0, currentZ);
            currentZ += visualDepth / 2 + baseGap;

            const desc = layer.layer_name === "Simple Attention" ? "Is the implementation of an attention layer in its simpliest and basic form. <br/> This layer creates a single-head Scaled Dot-Product Self-Attention layer <br/> inspired/based on the attention mechanism introduced by Vaswani et al. (2017)." : 
            "Is the advance and improved variant of the existing <code>simpleAttention</code>. <br/> It splits Query, Key, and Value projections into multiple independent attention heads."
 
            const layerData = {
                name: layer.layer_name,
                inputShape: [1, 1, layer.embedDim, layer.seqLen],
                outputShape: [1, 1, layer.embedDim, layer.seqLen],
                desc: desc,
            };

            cube.userData = layerData;

            // Create moving query/key scanning markers (the small travelling cubes)
            const seqLen = Math.min(Math.max(Math.round(layer.seqLen), 2), 10);
            const attentionGroup = createAttentionGridMarkers(visualWidth, visualHeight, visualDepth, seqLen);
            cube.add(attentionGroup);


            const animationData = {
                type: "attention",
                parentCube: cube,
                attentionGroup: attentionGroup,
                boundsW: visualWidth,
                boundsH: visualHeight,
                boundsD: visualDepth,
                seqLen: seqLen
            }

            if (layer.layer_name ===  "Multi Head Attention") {
                const headsGroup = createMultiHeadArches(visualWidth, visualHeight, numHeads || 4);
                cube.add(headsGroup);
            }

            animatedLayers.push(animationData);

            if (layer.isParametric) {
                layerData.weight_L1 = L1_Mean(weights[pointer]);
                layerData.bias_L1 = L1_Mean(biases[pointer]);
                pointer++;
            }

            modelGroup.add(cube);
        }
    });

    if (dataFlowGroup) {
        modelGroup.remove(dataFlowGroup);
    }

    let maxHeight = 0;
    layer_data.forEach(l => {
        if (l.inputShape) maxHeight = Math.max(maxHeight, l.inputShape[0] || 0);
    });
    gridHelper.position.y = -Math.max(20, (maxHeight / 2) + 10);

    // Generate new connection particles
    dataFlowGroup = createDataFlowParticles(animatedLayers);
    modelGroup.add(dataFlowGroup);

    const box = new THREE.Box3().setFromObject(modelGroup);
    const center = new THREE.Vector3();
    box.getCenter(center);

    controls.target.copy(center);
    controls.update();
}


function createSlidingWindow(width, height, depth, color) {
    const geom = new THREE.BoxGeometry(width, height, depth);
    const edges = new THREE.EdgesGeometry(geom);
    const lineMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
    const wireframe = new THREE.LineSegments(edges, lineMat);

    const mat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.25 });
    const fill = new THREE.Mesh(geom, mat);
    wireframe.add(fill);

    return wireframe;
}

function createAttentionGridMarkers(width, height, depth, seqLen) {
    const group = new THREE.Group();
    const cellW = width / seqLen;
    const cellH = height / seqLen;
    const cubeGeo = new THREE.BoxGeometry(cellW * 0.5, cellH * 0.5, depth + 0.2);

    // Stepped query column: one vertical strip of markers that jumps
    // one cell to the right every timestep (pixel-y, not eased).
    const queryMat = new THREE.MeshBasicMaterial({ color: 0xf7fbfc });
    const queryMarkers = [];
    for (let i = 0; i < seqLen; i++) {
        const mesh = new THREE.Mesh(cubeGeo, queryMat);
        group.add(mesh);
        queryMarkers.push(mesh);
    }

    // Stepped key row: one horizontal strip of markers that jumps
    // one cell downward every timestep. Fully independent of the
    // query column - its own axis, its own step clock.
    const keyMat = new THREE.MeshBasicMaterial({ color: 0xf7fbfc });
    const keyMarkers = [];
    for (let j = 0; j < seqLen; j++) {
        const mesh = new THREE.Mesh(cubeGeo, keyMat);
        group.add(mesh);
        keyMarkers.push(mesh);
    }

    // Pool of "echo" columns: every time the query column steps, one of
    // these is spawned at the query's current X, then flies off along
    // local +Z (toward the next layer) and fades out independently of
    // the query column's own movement.
    const echoPoolSize = Math.max(seqLen * 2, 6);

    const echoColumns = [];
    for (let e = 0; e < echoPoolSize; e++) {
        const echoGroup = new THREE.Group();
        const markers = [];
        for (let i = 0; i < seqLen; i++) {
            const echoMat = new THREE.MeshBasicMaterial({
                color: 0x00ffff,
                transparent: true,
                opacity: 0
            });
            const mesh = new THREE.Mesh(cubeGeo, echoMat);
            echoGroup.add(mesh);
            markers.push(mesh);
        }
        echoGroup.visible = false;
        echoGroup.userData = { active: false, spawnTime: 0, startX: 0 };
        group.add(echoGroup);
        echoColumns.push(echoGroup);
    }

    group.userData = {
        queryMarkers,
        keyMarkers,
        echoColumns,
        nextEcho: 0,     // round-robin pointer into the echo pool
        lastStep: -1     // last seen step index, to detect a new timestep
    };
    return group;
}

async function addEngravedLabel(parent, label, surfaceWidth, surfaceDepth) {
    const font = await reshapeFont;
    const fontSize = Math.min(surfaceWidth / (label.length * 1.8), surfaceDepth * 0.45);
    const textGeometry = new TextGeometry(label, {
        font,
        size: fontSize,
        depth: 0.04,
        curveSegments: 4,
        bevelEnabled: true,
        bevelThickness: 0.15,
        bevelSize: 0.01,
        bevelSegments: 2
    });

    textGeometry.computeBoundingBox();
    const textWidth = textGeometry.boundingBox.max.x - textGeometry.boundingBox.min.x;
    textGeometry.translate(-textWidth / 2, 0, -fontSize / 2);
    textGeometry.rotateX(-Math.PI / 2);

    const engravedText = new THREE.Mesh(
        textGeometry,
        new THREE.MeshStandardMaterial({
            color: 0xffff0,
            roughness: 0.9,
            metalness: 0.1,
        })
    );
    engravedText.position.y = 1;
    engravedText.position.z =  0;
    engravedText.userData = parent.userData;
    parent.add(engravedText);
}

function createScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, width / height, 0.5, 1000);

    modelGroup = new THREE.Group();
    scene.add(modelGroup);

    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    viewer.appendChild(renderer.domElement);

    camera.position.set(-50, 20, -50);
    camera.lookAt(0, 0, 0);

    const ambient = new THREE.AmbientLight(0xffffff, 1);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xffffff, 2);
    sun.position.set(5, 5, 5);
    scene.add(sun);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 2, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
}

window.addEventListener("resize", () => {
    const width = viewer.clientWidth;
    const height = viewer.clientHeight;

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});

function animate() {
    requestAnimationFrame(animate);

    const elapsedTime = clock.getElapsedTime();
    const speed = 15; // Speed of tensor data flow

    // Update sliding windows
    animatedLayers.forEach(item => {
        if (item.type === "rnn") {
            const {recurrenceLoop, boundsW } = item;

            // Rotate loops around the X axis (spinning vertically around the layer geometry)
            if (recurrenceLoop) {
                recurrenceLoop.children.forEach((loop, i) => {
                    // slow spin
                    loop.rotation.z = elapsedTime * 1.5;

                });
            }
        }
        else if (item.type === "attention") {
            const { attentionGroup, boundsW, boundsH, boundsD, seqLen, recurrenceLoop } = item;
            const { queryMarkers, keyMarkers, echoColumns } = attentionGroup.userData;

            const cellW = boundsW / seqLen;
            const cellH = boundsH / seqLen;

            const stepsPerSecond = 10; // how many timesteps per second (tweak to taste)
            const totalSteps = seqLen;

            // Discrete step index - jumps instantly, no easing/tweening
            const step = Math.floor(elapsedTime * stepsPerSecond) % totalSteps;

            const startX = -boundsW / 2 + cellW / 2;
            const startY = boundsH / 2 - cellH / 2;

            const stepX = startX + step * cellW;

            // 1. Move the query column in discrete, pixel-y jumps (left -> right)
            queryMarkers.forEach((m, idx) => {
                const posY = startY - idx * cellH;
                m.position.set(stepX, posY, 0);
            });

            // 1b. Move the key row in discrete, pixel-y jumps (top -> bottom).
            // Independent axis and independent step clock from the query column.
            const stepY = startY - step * cellH;
            keyMarkers.forEach((m, idx) => {
                const posX = startX + idx * cellW;
                m.position.set(posX, stepY, 0);
            });

            // 2. Spawn a new independent echo column whenever the step changes
            if (step !== attentionGroup.userData.lastStep) {
                attentionGroup.userData.lastStep = step;

                const echoColumns_ = echoColumns;
                const poolIdx = attentionGroup.userData.nextEcho % echoColumns_.length;
                attentionGroup.userData.nextEcho++;

                const echoGroup = echoColumns_[poolIdx];
                echoGroup.visible = true;
                echoGroup.userData.active = true;
                echoGroup.userData.spawnTime = elapsedTime;
                echoGroup.userData.startX = stepX;

                echoGroup.children.forEach((m, idx) => {
                    const posY = startY - idx * cellH;
                    m.position.set(stepX, posY, 0);
                    m.material.opacity = 0.9;
                });
            }

            // 3. Animate all active echo columns: fly along local +Z, fade out.
            // Fully decoupled from the query column's own movement.
            const travelDuration = 1.2;   // seconds to travel + fade
            const travelDistanceZ = boundsD * 6; // how far forward it ejects

            echoColumns.forEach(echoGroup => {
                if (!echoGroup.userData.active) return;

                const age = elapsedTime - echoGroup.userData.spawnTime;
                const t = age / travelDuration;

                if (t >= 1) {
                    echoGroup.visible = false;
                    echoGroup.userData.active = false;
                    return;
                }

                const z = t * travelDistanceZ;
                const fadeOpacity = 0.9 * (1 - t);

                echoGroup.children.forEach(m => {
                    m.position.z = z;
                    m.material.opacity = fadeOpacity;
                });
            });

            if (recurrenceLoop) {
                recurrenceLoop.children.forEach((loop, i) => {
                    // slow spin
                    loop.rotation.z = elapsedTime * 1.5;

                });
            }
        }
        else {
            const { windowMesh, boundsW, boundsH, winW, winH, strideX, strideY } = item;

            const maxStepsX = Math.max(1, Math.floor((boundsW - winW) / strideX) + 1);
            const maxStepsY = Math.max(1, Math.floor((boundsH - winH) / strideY) + 1);
            const totalSteps = maxStepsX * maxStepsY;

            const step = Math.floor(elapsedTime * 10) % totalSteps;
            const stepX = step % maxStepsX;
            const stepY = Math.floor(step / maxStepsX);

            const startX = -boundsW / 2 + winW / 2;
            const startY = boundsH / 2 - winH / 2;

            const posX = startX + stepX * strideX;
            const posY = startY - stepY * strideY;

            windowMesh.position.set(posX, posY, 0);
        }
    });

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(modelGroup.children, false);

    if (intersects.length > 0) {
        const object = intersects[0].object;

        if (hoveredObject !== object) {
            if (hoveredObject && originalEmissive) {
                hoveredObject.material.emissive.setHex(originalEmissive);
            }
            hoveredObject = object;
            originalEmissive = hoveredObject.material.emissive.getHex();
            hoveredObject.material.emissive.setHex(0x555555); 
            document.body.style.cursor = 'pointer'; 

            // Show and set tooltip text
            if (object.userData && object.userData.name) {
                tooltip.innerHTML = `
                    <strong>${object.userData.name}</strong><br/>
                    <p>Description: ${object.userData.desc}</p>
                    <p>Input Shape: ${JSON.stringify(object.userData.inputShape)}</p>
                    <p>Output Shape: ${JSON.stringify(object.userData.outputShape)}</p>
                    <p>Weight L1: ${object.userData?.weight_L1 || "Non-parametric"}</p>
                    <p>Bias L1: ${object.userData?.bias_L1 || "Non-parametric"}</p>
                `;
                tooltip.style.display = 'block';
            }
        }
    } else {
        if (hoveredObject) {
            if (originalEmissive) hoveredObject.material.emissive.setHex(originalEmissive);
            hoveredObject = null;
            document.body.style.cursor = 'default';

            // Hide tooltip
            tooltip.style.display = 'none';
        }
    }

    if (dataFlowGroup) {
        dataFlowGroup.children.forEach(points => {
            const positions = points.geometry.attributes.position.array;
            const { sourceZ, gapDistance, particleCount, offsets } = points.userData;

            for (let i = 0; i < particleCount; i++) {
                // Compute continuous movement along the gap using modulo
                const progress = ((elapsedTime * speed + offsets[i] * gapDistance) % gapDistance) / gapDistance;
                
                // Set current particle Z location along gap
                positions[i * 3 + 2] = sourceZ + (progress * gapDistance);
            }

            // Flag WebGL to update vertex buffer
            points.geometry.attributes.position.needsUpdate = true;
        });
    }

    renderer.render(scene, camera);
    controls.update();
}

// helper function to create cubes, because why not?
const createCube = ({width = 1, height = 1, depth = 1, color = 0x4f8cff, opacity = 0.8, outlineColor = null} = {}) => {
    const maxHeight = height > 2000 ? 2000 : height;
    const maxWidth = width > 2000 ? 2000 : width;
    const maxDepth = depth > 2000 ? 2000 : depth;

    const geometry = new THREE.BoxGeometry(maxWidth, maxHeight, maxDepth);
    const material = new THREE.MeshStandardMaterial({
        color,
        transparent: opacity < 1,
        opacity
    });

    const mesh = new THREE.Mesh(geometry, material);

    if (outlineColor !== null) {
        const edges = new THREE.EdgesGeometry(geometry);
        const lineMaterial = new THREE.LineBasicMaterial({ color: outlineColor });
        const wireframe = new THREE.LineSegments(edges, lineMaterial);
        mesh.add(wireframe);
    }

    return mesh;
};

function createMultiHeadArches(width, height, numHeads) {
    const group = new THREE.Group();
    const spacing = width / (numHeads + 1);
    const archRadius = spacing * 0.4;

    for (let i = 0; i < numHeads; i++) {
        // Semi-circle arc representing an individual head
        const curve = new THREE.EllipseCurve(
            0, 0,
            archRadius, archRadius * 1.5,
            0, Math.PI,
            false, 0
        );

        const points = curve.getPoints(30);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x00ffff });
        const arc = new THREE.Line(geometry, material);

        const posX = -width / 2 + spacing * (i + 1);
        arc.position.set(posX, height / 2, 0);

        group.add(arc);
    }
    return group;
}

function createRecurrenceLoop(width, height, depth) {
    const group = new THREE.Group();

    // Set ring radius based on mesh height/depth
    const radius = Math.max(height, depth) * 0.75;
    const loops = Math.min(Math.max(Math.floor(width / 6), 2), 4);
    const spacing = width / (loops + 1);

    for (let i = 0; i < loops; i++) {
        const loop = new THREE.Group();

        // 1. Create continuous 2D circular path
        const curve = new THREE.EllipseCurve(
            0, 0,
            radius, radius,
            0, Math.PI * 2,
            false, 0
        );

        const points = curve.getPoints(80);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x00ffff });
        const circle = new THREE.LineLoop(geometry, material);

        // 2. Cone arrow placed at top of the loop (0, radius, 0)
        const arrowGeom = new THREE.ConeGeometry(0.35, 1.0, 8);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0x00ffff });
        const arrow = new THREE.Mesh(arrowGeom, arrowMat);

        // Position arrow at top vertex
        arrow.position.set(0, radius, 0);
        // Point arrow horizontally along tangent (facing +X direction)
        arrow.rotation.z = -Math.PI / 2;

        loop.add(circle);
        loop.add(arrow);

        // 3. Offset center so bottom portion clips into mesh top boundary
        // Position X along mesh width, Y slightly above half-height
        loop.position.x = -width / 2 + spacing * (i + 1);
        loop.position.y = height / 2 - radius * 0.35; 

        group.add(loop);
    }

    return group;
}

function createDataFlowParticles(animatedLayers) {
    const particleGroup = new THREE.Group();
    const particleCountPerGap = 30; // Adjust density per gap

    for (let i = 0; i < animatedLayers.length - 1; i++) {
        const sourceCube = animatedLayers[i].parentCube;
        const targetCube = animatedLayers[i + 1].parentCube;

        const sourceZ = sourceCube.position.z;
        const targetZ = targetCube.position.z;
        const gapDistance = targetZ - sourceZ;

        // Shared geometry and material for high performance
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCountPerGap * 3);
        const offsets = new Float32Array(particleCountPerGap); // Stores unique speed/time offsets

        const sourceBounds = animatedLayers[i];
        
        for (let p = 0; p < particleCountPerGap; p++) {
            // Randomize X and Y within the bounds of the source layer
            const x = (Math.random() - 0.5) * sourceBounds.boundsW;
            const y = (Math.random() - 0.5) * sourceBounds.boundsH;
            
            positions[p * 3] = x;
            positions[p * 3 + 1] = y;
            positions[p * 3 + 2] = sourceZ; // Start at source layer Z

            offsets[p] = Math.random(); // Stagger start positions along the gap
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const material = new THREE.PointsMaterial({
            color: 0x00ffff,
            size: 0.9,
            transparent: true,
            opacity: 0.7,
            blending: THREE.AdditiveBlending
        });

        const points = new THREE.Points(geometry, material);
        
        // Attach metadata to animate in the loop
        points.userData = {
            sourceZ,
            gapDistance,
            particleCount: particleCountPerGap,
            offsets
        };

        particleGroup.add(points);
    }

    return particleGroup;
}

viewer.addEventListener("pointermove", (event) => {
    const rect = viewer.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / viewer.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / viewer.clientHeight) * 2 + 1;
});


function gridDisplayController() {
    gridHelper.position.y = -20;
    if (isEnableGrid) {
        scene.add(gridHelper);
        gridController.innerText = "Disable Grid";
        isEnableGrid = false;
    }
    else {
        scene.remove(gridHelper);
        isEnableGrid = true;
        gridController.innerText = "Enable Grid";
    }
}




function listLayers(layers) {
    list.innerHTML = '';

    layers.forEach(layer => {
        const p = document.createElement('p');

        const color = layer.layer_name === "Convolutional Layer" ? "#4f8cff": 
                    layer.layer_name === "Simple Attention" || layer.layer_name === "Multi Head Attention" ? "#6b6d6e": 
                    layer.layer_name === "Max Pooling" ? "#FFAE00" : 
                    layer.layer_name === "Recurrent Cell" ? "#FF69B4" :
                    layer.layer_name === "Embedding Layer" ? "#BB6BD9" :
                    layer.layer_name === "Trans Convolution" ? "#00d52e" :
                    layer.layer_name === "Reshape" ? "#fbfffc" :
                    "red"; // default color: red for connected layer

        p.innerHTML = `
             <span style = "display: flex; justify-content: flex-start; gap: 10px; align-items: center; width: 100%; flex-wrap: no-wrap">
                <p style = "height: 10px; width: 10px; background-color: ${color}"></p>
                <p>${layer.layer_name}</p>
            </span>
        `;

        list.appendChild(p);
    });
}

viewer.addEventListener("pointermove", (event) => {
    const rect = viewer.getBoundingClientRect();

    // 1. Update Raycaster normalized coordinates
    mouse.x = ((event.clientX - rect.left) / viewer.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / viewer.clientHeight) * 2 + 1;

    // 2. Position tooltip next to cursor
    tooltip.style.left = `${event.clientX + 12}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
});

window.gridDisplayController = gridDisplayController;