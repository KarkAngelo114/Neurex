const Neurex = require('../core');
const { red, reset, green, lime } = require('../color-code');
const { load_multiple_images } = require('./imagery');
const fs = require('fs').promises;
const path = require('path');

class Annotator {
    constructor () {
        this.model_path = null;
        this.target_directory_path = null;
        this.CSV_file_name = null; // for CSV
        this.task = null; // loaded by the model
        this.neurex_instance = null;
        this.shape = [];
        this.size = [];

        this.datasets = null;
        this.image_paths = null;
        this.classes = null;
        this.parent_folder = "annotator_dumps";
    }


    configure(options) {
        this.model_path = options.model_path ?? null;
        this.target_directory_path = options.target_directory_path ?? null;
        this.classes = options.classes ?? null;
        this.CSV_file_name = options.CSV_file_name ?? null;
    }

    init() {
        console.log(`${green}\n[Task]------- Spinning up a new instance ${reset}`);
        

        if (!this.model_path) {
            throw new Error(`
${red}[ERROR]------- Failed to spin up a new instance.${reset}
Reason: No loaded model yet. Ensure you have specified it in the configure() first:

configure({
    model_path: 'model-name.nrx'
})

            `);
        }

        this.neurex_instance = new Neurex();

        this.neurex_instance.loadSavedModel(this.model_path);
    }

    // prepare for classification
    async imageClassifier() {

        if (!this.neurex_instance) throw new Error(`${red}[ERROR]-------- Neurex instance is null. Please call "init()" first. ${reset}`); 

        this.task = this.neurex_instance.get_task_type();
        this.shape = this.neurex_instance.getTensorShape();
        this.size = this.neurex_instance.getInputSize();

        // check for task type and shape
        // for image, H > 1 and W > 1. If H (this.shape[0]) and W (this.shape[1]) are equal to one, then the model is not trained on data that are not in proper tensor shape (like 1D array of features from tabular datasets) 
        if ((this.task !== "binary_classification" && this.task !== "multi_class_classification") && this.shape[0] == 1 && this.shape[1] == 1) {
            throw new Error(`
${red}[ERROR]------- Model is trained on image classification only${reset}
    - Load models that are trained on Image classification having valid saved input shape to avoid this error
            `);
        }

        console.log(`${green}[TASK]------- Reading image directory from "${this.target_directory_path}/" ${reset}`);

        if (!this.target_directory_path) {
throw new Error(`
${red}[ERROR]------- No image directory has been set ${reset}
    - Did you forget to set it in the configuration? Use:

        configure({
            target_directory_path: 'path/to/images'
        })
            `);
        }

        // The loaded model saves the shape of the input. So, no need for users to enter shape and pixel format as this method will automatically set it
        const {datasets, paths} = await load_multiple_images(this.target_directory_path, this.shape, this.shape[2] == 1 ? "grayscale" : this.shape[2] == 3 ? "rgb" : "rgba");
        this.datasets = datasets;
        this.image_paths = paths;

        // create a dump folder for the annotator to put the images. The created folder must have subfolders named after the class names set in the configure();
        console.log(`\n${green}[TASK]------- Creating dump folder... ${reset}`);
        if (!this.classes) {
            throw new Error(`
${red}[ERROR]------ An error occurred, please specify class names in an array in the configure():${reset}

configure({
    classes: ["class1", "class2", ....]
});
            `);
        }
        
        await fs.mkdir(this.parent_folder, {recursive:true});

        for (const folder of this.classes) {
            const folder_path = path.join(this.parent_folder, folder);
            await fs.mkdir(folder_path, {recursive:true});
        }

        console.log(`\n${lime}[SUCCESS]------- Dump folder has been created ${reset}`);

    }

    async image_classify() {
        if ((this.task !== "binary_classification" && this.task !== "multi_class_classification") 
            && this.shape[0] == 1 && this.shape[1] == 1) {
            throw new Error(`
    ${red}[ERROR]------- Model is trained on image classification only${reset}
        - Load models that are trained on Image classification having valid saved input shape to avoid this error
            `);
        }

        console.log(`\n${green}[TASK]------- Annotation in progress...${reset}`);
        const predictions = this.neurex_instance.predict(this.datasets);

        for (let i = 0; i < predictions.length; i++) {

            const file_path = this.image_paths[i];
            const pred_values = predictions[i];

            const max_value = Math.max(...pred_values);
            const idx = pred_values.indexOf(max_value);

            const predicted_class = this.classes[idx];

            if (!predicted_class) {
                console.warn(`${red}[WARNING] No class found for index ${idx}${reset}`);
                continue;
            }

            // Extract file name
            const file_name = path.basename(file_path);

            // Destination path
            const destination_path = path.join(
                this.parent_folder,
                predicted_class,
                file_name
            );

            try {
                await fs.rename(file_path, destination_path);
                
            } catch (err) {
                console.error(`${red}[ERROR] Failed to move ${file_name}:${reset}`, err);
            }
        }

        console.log(`\n${lime}[SUCCESS]------- Annotation completed.${reset}`);
    }
}

module.exports = {Annotator}