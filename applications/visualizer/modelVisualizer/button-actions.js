let topNav = document.getElementById("top-nav");
let bottomNav = document.getElementById("bottom-nav");
let hidden = document.getElementById('hidden');

function viewFullScreen(isTakingScreenShot = false) {
    topNav.style.display = "none";
    bottomNav.style.display = "none";

    if (!isTakingScreenShot) {
        hidden.style.display = "block";
    }
    
}

function exitFullScreen() {
    topNav.style.display = "flex";
    bottomNav.style.display = "grid";
    hidden.style.display = "none";
    
}


function takeScreenshot() {
    // Hide UI overlay if desired
    viewFullScreen(true);

    // Allow the DOM to render before capturing
    requestAnimationFrame(() => {
        // 1. Get canvas element from your #renderer container
        const canvas = document.querySelector("#renderer canvas");
        if (!canvas) return;

        // 2. Convert canvas content to base64 image data URL
        const imageData = canvas.toDataURL("image/png");

        // 3. Create a temporary <a> element to trigger download
        const link = document.createElement("a");
        link.download = "model-visualization.png";
        link.href = imageData;
        link.click();

        // Restore UI
        exitFullScreen();
    });
}