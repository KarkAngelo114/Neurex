
/**
 * 
 * @param {Object} shapeConfig shape configuration object to set the size and shape the network is expecting
 * @returns {Object} input layer config 
 */
const inputConfig = (shapeConfig) => {
    try {
        if (shapeConfig.featureSize && shapeConfig.sequenceLength) {
            const { featureSize, sequenceLength } = shapeConfig;
            return {
                layer_name: "input_layer",
                layer_size: featureSize * sequenceLength,
                input_shape: [1, 1, featureSize, sequenceLength]
            };
        }
        else if (shapeConfig.features) {
            const features = shapeConfig.features;
            return {
                layer_name: "input_layer",
                layer_size: features,
                input_shape: [1, 1, features, 1]
            };
        }
        else if (shapeConfig.height && shapeConfig.width && shapeConfig.depth) {
            const { height, width, depth } = shapeConfig;

            return {
                layer_name: "input_layer",
                layer_size: height * width * depth,
                input_shape: [height, width, depth]
            };
        } 
        else {
            throw new Error(`[ERROR]------- Invalid input shape config`);
        }
    } 
    catch (error) {
        console.error(error.message);
    }
}

module.exports = inputConfig;