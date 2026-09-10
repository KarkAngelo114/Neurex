// each gradient normalizer function must have an exposed `normalize()` function to be called by the training engine


const { gradientClipping } = require("../core/bindings/entry")

const clipGradient = (clip_norm_value_threshold = 5.0) => {
    return function normalize(grads) {
        return gradientClipping(grads, clip_norm_value_threshold);
    }
}


module.exports = {
    clipGradient
}