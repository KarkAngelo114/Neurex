const { execSync } = require('child_process')


const runCommand = (command) => {
    try {
        execSync(command, { stdio: 'inherit' });
    } catch (e) {
        console.error(`\n[FAIL] Command failed: "${command}"`);
        process.exit(1); 
    }
};

(async () => {
    runCommand("cls");
    runCommand('git checkout main');
    runCommand('git pull origin experimental');
    runCommand('git push origin main');
    runCommand('git checkout experimental');

    console.log('\nDone...');
})();