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

const cargoBin = findCargo();
const args = process.argv.slice(2);

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
