const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

function findCargo() {
  const ext = process.platform === 'win32' ? '.exe' : '';
  const pathDirs = (process.env.PATH || '').split(path.delimiter);
  
  for (const dir of pathDirs) {
    const full = path.join(dir, `cargo${ext}`);
    if (fs.existsSync(full)) return `cargo${ext}`;
  }

  const home = process.env.USERPROFILE || process.env.HOME || '';
  const cargoHome = path.join(home, '.cargo', 'bin', `cargo${ext}`);
  if (fs.existsSync(cargoHome)) return cargoHome;

  return 'cargo';
}

const args = process.argv.slice(2);

// Fast-path: If running the backend binary and it is already compiled, run it directly
if (args.includes('run') && (args.includes('web_ml_backend') || args.includes('-p'))) {
  const exePath = path.join(__dirname, '..', 'target', 'debug', process.platform === 'win32' ? 'web_ml_backend.exe' : 'web_ml_backend');
  if (fs.existsSync(exePath)) {
    const child = spawn(exePath, [], { stdio: 'inherit', env: process.env });
    child.on('error', (err) => {
      console.error(`Failed to launch backend binary: ${err.message}`);
      process.exit(1);
    });
    child.on('exit', (code, signal) => {
      if (code !== null) process.exit(code);
      if (signal) process.kill(process.pid, signal);
    });
    return;
  }
}

const cargoBin = findCargo();

const home = process.env.USERPROFILE || process.env.HOME || '';
const cargoDir = path.join(home, '.cargo', 'bin');
const env = { ...process.env };
if (fs.existsSync(cargoDir)) {
  env.PATH = `${cargoDir}${path.delimiter}${env.PATH || ''}`;
}

const child = spawn(cargoBin, args, { stdio: 'inherit', env });

child.on('error', (err) => {
  console.error(`\nFailed to start cargo: ${err.message}`);
  console.error('Please ensure Rust is installed and C:\\Users\\<username>\\.cargo\\bin is added to your PATH.\n');
  process.exit(1);
});

child.on('exit', (code, signal) => {
  if (code !== null) process.exit(code);
  if (signal) process.kill(process.pid, signal);
});
