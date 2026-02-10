
const { red, reset, green } = require('../prettify')
const fs = require('fs').promises;
const sharp = require('sharp');

/**
 * @function load_images_from_directory
 * @param {String} targetDir - target directory of your image datasets. The folders inside the target directory will represents as class names for the images inside. The first class being read will be the first class among all classes. Therefore, assign your data to it's correct class.
 * @param {Array<Number>} resize - an array containing the values for resizing [H, W].
 * @param {String} pixelFormat - grayscale, rgb, or rgba. "grayscale" - 1 channel, "rgb" - 3 channel, and "rgba" - 4 channels.
 * @returns an object that contains the datasets, labels, and classes
 */
const load_images_from_directory = async (targetDir, resize = [28, 28], pixelFormat = "grayscale") => {

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
            for (let imgName of imgItems) {
                const imgPath = `${classDir}/${imgName}`;
                const stat = await fs.stat(imgPath);
                if (stat.isFile()) {
                    try {
                        // Set sharp options for pixel format
                        let toFormat = 'greyscale';
                        if (pixelFormat === 'rgb') {
                            channels = 3;
                            toFormat = 'rgb';
                        } else if (pixelFormat === 'rgba') {
                            channels = 4;
                            toFormat = 'rgba';
                        }

                        let image = sharp(imgPath).resize(resize[1], resize[0]);
                        if (pixelFormat === 'grayscale') {
                            image = image.grayscale();
                        } else if (pixelFormat === 'rgb') {
                            image = image.toColourspace('rgb');
                        } else if (pixelFormat === 'rgba') {
                            image = image.toColourspace('rgba');
                        }

                        const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
                        // Normalize pixel values to [0, 1] and reshape to [height][width][channels]
                        const normalized = Array.from(data).map(v => v / 255);
                        const { width, height, channels } = info;
                        const tensor = [];
                        let idx = 0;
                        for (let h = 0; h < height; h++) {
                            const row = [];
                            for (let w = 0; w < width; w++) {
                                const pixel = [];
                                for (let c = 0; c < channels; c++) {
                                    pixel.push(normalized[idx++]);
                                }
                                row.push(pixel);
                            }
                            tensor.push(row);
                        }
                        datasets.push(tensor);
                        labels.push([className]);
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

module.exports = {
    load_images_from_directory
}