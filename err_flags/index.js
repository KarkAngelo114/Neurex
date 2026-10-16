

const NAN_ERR = "ERR_NAN_DETECTED"; // error flag means there are NaNs (Not a number) detected in an input or an output of an operator
const BUILD_ERR = "ERR_BUILD_ERROR"; // error flag triggered if there are problem building or reloading a model
const UNSUPPORTED_MODEL_VER = "ERR_UNSUPPORTED_NRX_VERSION"; // error flag triggered if loading an unsupported model version
const UNSUPPORTED_MODEL_TYPE = "ERR_MODEL_TYPE_UNSUPPORTED"; // error flag for loading unsupported model type
const SHAPE_ERR = "ERR_SHAPE_MISMATCH"; // error flag if shape input mismatch with the expected shape input
const MODEL_SAVE_ERR = "ERR_MODEL_SAVING"; // error flag if model saving has a problem
const UNEQUAL_SIZE = "ERR_SIZE_MISMATCH"; // error flag if inputs are unequal to each other (usually for comparing two or more arrays which must have equal size)
const ARGUMENT_NULL_OR_UNDEFINED = "ERR_ARG_NULLED_UNDEFINED"; // error flag if passing an argument to a function with nulled or undefined values
const NEGATIVE_NUMBER_ARGUMENT = "ERR_NEGATIVE_NUMBER"; // error flag for passing a negative number as an argument to a function
const OPENCL_ERR = "ERR_OPENCL_ERROR"; // error flag for OpenCL initialization
const PLUGIN_ERR = "ERR_PLUGIN_ERROR"; // error flag for plugin-related functions
const MODEL_PARAMETER_ERR = "ERR_MODEL_PARAMETER"; // error flag if there is now model parameter initialized upon model creation or reloading
const ACTIVATION_TYPE_ERR = "ERR_ACTIVATION_TYPE"; // error flag for unknown activation function used in a layer
const LOAD_MODEL_ERR = "ERR_LOAD_MODEL"; // error flag for loading model related functions
const UNSUPPORTED_LOSS_ERR = "ERR_UNSUPPORTED_LOSS_FUNC"; // error flag for using unsupported loss function
const NO_LAYER_UNIQUE_ID_ERR = "ERR_LAYER_ID_UNDEFINED"; // error flag if a layer type has no layerID
const VALUE_TYPE_ERR = "ERR_INVALID_VALUE_DATA_TYPE"; // error flag for value type error
const COMPUTE_BACKEND_ERR = "ERR_COMPUTE_BACKEND_SETTING"; // error flag for setting compute backend
const LABEL_MODE_ERR = "ERR_LABEL_MODE_ERR"; // error flag used when error triggered during data preprocessing
const ERR_UNRESOLVED_DIR_PATH = "ERR_UNRESOLVED_PATH_TO_DIR"; // error flag used if an error triggered navigating to a filesystem path

module.exports = {
    NAN_ERR,
    BUILD_ERR,
    UNSUPPORTED_MODEL_TYPE,
    UNSUPPORTED_MODEL_VER,
    SHAPE_ERR,
    MODEL_SAVE_ERR,
    UNEQUAL_SIZE,
    ARGUMENT_NULL_OR_UNDEFINED,
    NEGATIVE_NUMBER_ARGUMENT,
    OPENCL_ERR,
    PLUGIN_ERR,
    MODEL_PARAMETER_ERR,
    ACTIVATION_TYPE_ERR,
    LOAD_MODEL_ERR,
    UNSUPPORTED_LOSS_ERR,
    NO_LAYER_UNIQUE_ID_ERR,
    VALUE_TYPE_ERR,
    COMPUTE_BACKEND_ERR,
    LABEL_MODE_ERR,
    ERR_UNRESOLVED_DIR_PATH
}


