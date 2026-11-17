#!/usr/bin/env node
// Builds or tests the Rust ML engine.
//
//   node scripts/engine.mjs build   -> compiles engine/ to WebAssembly and writes the
//                                      browser bindings to frontend/src/engine/pkg/
//   node scripts/engine.mjs test    -> runs the engine's tests natively (cargo test)
//
// Needs Rust with the wasm32-unknown-unknown target and wasm-bindgen-cli at the same
// version as the wasm-bindgen crate (the script tells you the exact install command).

import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
import { delimiter, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const exe = process.platform === 'win32' ? '.exe' : '';
const cargoBin = join(homedir(), '.cargo', 'bin');

// Make ~/.cargo/bin visible even when the shell PATH doesn't include it
const env = { ...process.env };
if (existsSync(cargoBin)) env.PATH = `${cargoBin}${delimiter}${env.PATH ?? ''}`;

function run(cmd, args, options = {}) {
  const result = spawnSync(cmd + exe, args, { cwd: root, env, stdio: 'inherit', ...options });
  if (result.error) {
    console.error(`\nCould not start ${cmd}: ${result.error.message}`);
    process.exit(1);
  }
  if (result.status !== 0) process.exit(result.status ?? 1);
  return result;
}

function capture(cmd, args) {
  const result = spawnSync(cmd + exe, args, { cwd: root, env, encoding: 'utf8' });
  return result.status === 0 ? result.stdout.trim() : null;
}

/** wasm-bindgen version pinned in Cargo.lock; the CLI must match it exactly. */
function lockedBindgenVersion() {
  const lock = readFileSync(join(root, 'Cargo.lock'), 'utf8');
  const match = lock.match(/name = "wasm-bindgen"\r?\nversion = "([^"]+)"/);
  if (!match) throw new Error('wasm-bindgen not found in Cargo.lock');
  return match[1];
}

function build() {
  // Resolve the lock file (and the wasm-bindgen version) before checking the CLI
  run('cargo', ['build', '-p', 'ml_engine', '--target', 'wasm32-unknown-unknown', '--release']);

  const wanted = lockedBindgenVersion();
  const installed = capture('wasm-bindgen', ['--version'])?.split(/\s+/)[1];
  if (installed !== wanted) {
    console.error(
      `\nwasm-bindgen-cli ${wanted} is required (found ${installed ?? 'none'}). Install it with:\n` +
        `  cargo install wasm-bindgen-cli --version ${wanted} --locked\n`,
    );
    process.exit(1);
  }

  const metadata = JSON.parse(capture('cargo', ['metadata', '--format-version', '1', '--no-deps']));
  const wasm = join(metadata.target_directory, 'wasm32-unknown-unknown', 'release', 'ml_engine.wasm');
  const outDir = join(root, 'frontend', 'src', 'engine', 'pkg');
  rmSync(outDir, { recursive: true, force: true });
  run('wasm-bindgen', ['--target', 'web', '--out-dir', outDir, '--out-name', 'ml_engine', wasm]);
  console.log(`\nEngine built: ${outDir}`);
}

const command = process.argv[2];
if (command === 'build') build();
else if (command === 'test') run('cargo', ['test', '--workspace']);
else {
  console.error('Usage: node scripts/engine.mjs <build|test>');
  process.exit(1);
}
