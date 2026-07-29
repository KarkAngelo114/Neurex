module.exports = {
    stepDecay: (dropFactor = 0.5, dropEvery = 10) => (epoch, currentLR, prevLoss, initialLR) => initialLR * Math.pow(dropFactor, Math.floor(epoch / dropEvery)),
    exponentialDecay: (decayRate = 0.96) => (epoch, currentLR, prevLoss, initialLR) => initialLR * Math.pow(decayRate, epoch),
    cosineAnnealing: (totalEpochs, minLR = 0) => (epoch, currentLR, prevLoss, initialLR) => minLR + 0.5 * (initialLR - minLR) * (1 + Math.cos(Math.PI * epoch / totalEpochs)),
    reduceOnPlateau: ({ factor = 0.5, patience = 5, minLR = 1e-6 } = {}) => {
        let best = Infinity, wait = 0;
        return (epoch, currentLR, prevLoss) => {
            if (prevLoss < best - 1e-4) { 
                best = prevLoss; wait = 0; return currentLR; 
            }
            wait++;
            if (wait >= patience) { 
                wait = 0; return Math.max(currentLR * factor, minLR); 
            }
            return currentLR;
        };
    }
}