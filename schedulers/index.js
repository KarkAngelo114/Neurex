// module.exports = {
//     stepDecay: (dropFactor = 0.5, dropEvery = 10) => (epoch, currentLR, prevLoss, initialLR) => initialLR * Math.pow(dropFactor, Math.floor(epoch / dropEvery)),
//     exponentialDecay: (decayRate = 0.96) => (epoch, currentLR, prevLoss, initialLR) => initialLR * Math.pow(decayRate, epoch),
//     cosineAnnealing: (totalEpochs, minLR = 0) => (epoch, currentLR, prevLoss, initialLR) => minLR + 0.5 * (initialLR - minLR) * (1 + Math.cos(Math.PI * epoch / totalEpochs)),
//     reduceOnPlateau: ({ factor = 0.5, patience = 5, minLR = 1e-6 } = {}) => {
//         let best = Infinity, wait = 0;
//         return (epoch, currentLR, prevLoss) => {
//             if (prevLoss < best - 1e-4) { 
//                 best = prevLoss; wait = 0; return currentLR; 
//             }
//             wait++;
//             if (wait >= patience) { 
//                 wait = 0; return Math.max(currentLR * factor, minLR); 
//             }
//             return currentLR;
//         };
//     }
// }


function stepDecay(dropFactor = 0.5, dropEvery = 10) {
    return function (data) {

        const { current_epoch, initial_learning_rate } = data; // destructure data

        const floorOutput = Math.floor(current_epoch / dropEvery);

        return initial_learning_rate * Math.pow(dropFactor, floorOutput);
    }
}

function exponentialDecay(decayRate = 0.96) {
    return function (data) {
        const { current_epoch, initial_learning_rate } = data; // destructure data

        return initial_learning_rate * Math.pow(decayRate, current_epoch);
    }
}

function cosineAnnealing(totalEpochs = 100, minLR = 0) {
    return function (data) {
        const {initial_learning_rate, current_epoch} = data;

        return minLR + 0.5 * (initial_learning_rate - minLR) * (1 + Math.cos(Math.PI * current_epoch / totalEpochs));
    }
}

function reduceOnPlateau({ factor = 0.5, patience = 5, minLR = 1e-6 } = {}) {
    let best = Infinity, wait = 0;

    return function (data) {
        const {previousEpochLoss, current_epoch, learning_rate} = data;

        if (previousEpochLoss < best - 1e-4) { 
            best = previousEpochLoss; wait = 0; 
            return learning_rate; 
        }
        wait++;

        if (wait > patience) {
            wait = 0; 
            return Math.max(learning_rate * factor, minLR); 
        }

        return learning_rate;
        

    }

}


module.exports = {
    stepDecay,
    exponentialDecay,
    cosineAnnealing,
    reduceOnPlateau
}