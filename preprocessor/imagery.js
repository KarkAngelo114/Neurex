
const { toTensor } = require('./reshaper');
const { red, reset, green } = require('../prettify')
const fs = require('fs').promises;
const sharp = require('sharp');

/**
 * @function load_images_from_directory
 * @param {String} targetDir - target directory of your image datasets. The folders inside the target directory will represents as class names for the images inside. The first class being read will be the first class among all classes. Therefore, assign your data to it's correct class.
 * @param {Array<Number>} resize - an array containing the values for resizing [H, W].
 * @param {String} pixelFormat - grayscale, rgb, or rgba. "grayscale" - 1 channel, "rgb" - 3 channel, and "rgba" - 4 channels.
 * @param {Number} limit_per_class - limit the number of items per class
 * @returns an object that contains the datasets, labels, and classes
 */
const load_images_from_directory = async (targetDir, resize = [28, 28], pixelFormat = "grayscale", limit_per_class = 0) => {

    const subdirs = []; // only subdirectories (class names)
    const datasets = []; // all images converted to tensors and already normalized.
    const labels = []; // folder names where the image belongs to.

    try {
        console.log(`\n${green}[Task]------- Loading datasets from "${targetDir}/" ${reset}`)
        const items = await fs.readdir(targetDir);

        // Only collect subdirectories (class names)
        for (let item of items) {
            const itemPath = `${targetDir}/${item}`;
            const stat = await fs.stat(itemPath);
            if (stat.isDirectory()) {
                subdirs.push(item);
            }
        }

        for (let className of subdirs) {
            const classDir = `${targetDir}/${className}`;
            const imgItems = await fs.readdir(classDir);

            let loadedCount = 0;
            for (let imgName of imgItems) {
                
                if (loadedCount >= limit_per_class) {
                    break;
                }

                const imgPath = `${classDir}/${imgName}`;
                const stat = await fs.stat(imgPath);
                if (stat.isFile()) {
                    try {

                        let image = sharp(imgPath).resize(resize[1], resize[0]);

                        if (pixelFormat === 'grayscale') {
                            image = image.grayscale(); 
                        } else if (pixelFormat === 'rgb') {
                            image = image.removeAlpha();
                        } else if (pixelFormat === 'rgba') {
                            image = image.ensureAlpha();
                        }

                        const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
                        // Normalize pixel values to [0, 1] and reshape to [height][width][channels]
                        const normalized = Array.from(data).map(v => v / 255);
                        const { width, height, channels } = info;
                        
                        datasets.push(toTensor(normalized, [height, width, channels]));
                        labels.push([className]);
                        loadedCount++;
                    } catch (imgErr) {
                        console.error(`${red} Error processing image: ${imgPath}\n`, imgErr, `${reset}`);
                    }
                }
            }
        }


        console.log(`${green}[/]------- Successfully loaded datasets from "${targetDir}/"${reset}\n`);
        console.log(`- Found ${subdirs.length} classes`);
        console.log(`- Found ${datasets.length} items in total`);
        return {
            datasets: datasets,
            labels: labels,
            classes: subdirs
        }
    } catch (error) {
        console.error(`${red} Error occured in loading dataset: \n`, error, `${reset}`);
        process.exit(1);
    }
}

/**
 * 
 * @param {String} file_path - path to the image file (can be nested anywhere)
 * @param {Array<Number>} resize - resize the image to [H, W]
 * @param {String} pixelFormat - grayscale, rgb, or rgba.
 * @returns a normalized tensor map
 */
const load_single_image = async (file_path, resize = [28, 28], pixelFormat = "grayscale") => {

    try {
        console.log(`\n${green}[Task]------- Loading image "${file_path}" ${reset}`);

        const stat = await fs.stat(file_path);

        if (!stat.isFile()) {
            throw new Error("Provided path is not a valid file.");
        }

        let image = sharp(file_path).resize(resize[1], resize[0]);

        // Apply pixel format
        if (pixelFormat === 'grayscale') {
            image = image.grayscale();
        } else if (pixelFormat === 'rgb') {
            image = image.removeAlpha();
        } else if (pixelFormat === 'rgba') {
            image = image.ensureAlpha();
        }

        const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });

        const normalized = Array.from(data).map(v => v / 255);
        const { width, height, channels } = info;

        const tensor = toTensor(normalized, [height, width, channels]);

        console.log(`${green}[/]------- Successfully loaded image "${file_path}"${reset}\n`);

        return {
            datasets: [tensor],
            shape: [height, width, channels]
        };

    } catch (error) {
        console.error(`${red} Error occurred while loading image:\n`, error, `${reset}`);
        process.exit(1);
    }
};


/**
 * 
 * @param {String} file_path - path to the image file (can be nested anywhere)
 * @param {Array<Number>} resize - resize the image to [H, W]
 * @param {String} pixelFormat - grayscale, rgb, or rgba.
 * @returns an object containing an array of tensor normalized tensor maps and their file path
 */
const load_multiple_images = async (file_path, resize = [28, 28], pixelFormat = "grayscale") => {

    const datasets = [];
    const paths = [];

    try {
        console.log(`\n${green}[Task]------- Loading image "${file_path}" ${reset}`);

        const items = await fs.readdir(file_path);

        for (const img of items) {
            const imagePath = `${file_path}/${img}`;

            paths.push(imagePath);

            let image = sharp(imagePath).resize(resize[1], resize[0]);

            if (pixelFormat === "grayscale") {
                image = image.grayscale();
            }
            else if (pixelFormat === "rgb") {
                image = image.removeAlpha();
            }
            else if (pixelFormat === "rgba") {
                image = image.ensureAlpha();
            }

            const {data, info} = await image.raw().toBuffer({ resolveWithObject:true});
            const normalized = Array.from(data).map(v => v / 255);
            const { width, height, channels } = info;

            datasets.push(toTensor(normalized, [height, width, channels]));
            

        }

        return {
            datasets: datasets,
            paths: paths
        }
    }
    catch (error) {
        console.error(`${red} Error occurred while loading image:\n`, error, `${reset}`);
        process.exit(1);
    }
}

module.exports = {
    load_images_from_directory,
    load_single_image,
    load_multiple_images
}