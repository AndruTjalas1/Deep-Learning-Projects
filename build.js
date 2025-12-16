#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const projects = [
  { name: 'hub', path: '.', buildCmd: 'npx vite build' },
  { name: 'DNN', path: '../Deep Neural Network/frontend', buildCmd: 'npm run build' },
  { name: 'GAN', path: '../GAN/Frontend', buildCmd: 'npm run build' },
  { name: 'RNN', path: '../RNN/frontend', buildCmd: 'npm run build' },
];

console.log('🚀 Building monorepo...\n');

for (const project of projects) {
  console.log(`📦 Building ${project.name}...`);
  
  try {
    // Install dependencies
    console.log(`  → Installing dependencies for ${project.name}...`);
    execSync(`cd "${project.path}" && npm install`, { stdio: 'inherit' });
    
    // Build with direct command (not npm run build to avoid recursion)
    console.log(`  → Building ${project.name}...`);
    execSync(`cd "${project.path}" && ${project.buildCmd}`, { stdio: 'inherit' });
    
    console.log(`✅ ${project.name} built successfully\n`);
  } catch (error) {
    console.error(`❌ Failed to build ${project.name}`);
    process.exit(1);
  }
}

// Copy all dist folders into my-app/dist for Vercel deployment
console.log('📦 Consolidating builds for deployment...\n');

const cwd = process.cwd();
const projectPaths = [
  { dist: 'Deep Neural Network/frontend/dist', dest: 'dist/dnp', name: 'DNN' },
  { dist: 'GAN/Frontend/dist', dest: 'dist/gan', name: 'GAN' },
  { dist: 'RNN/frontend/dist', dest: 'dist/rnn', name: 'RNN' },
];

for (const { dist, dest, name } of projectPaths) {
  const srcPath = path.resolve(cwd, '..', dist);
  const destPath = path.resolve(cwd, dest);
  
  console.log(`  Checking ${name}: ${srcPath}`);
  
  if (fs.existsSync(srcPath)) {
    console.log(`  → Copying ${name} from ${srcPath} to ${destPath}...`);
    
    // Remove existing destination
    if (fs.existsSync(destPath)) {
      fs.rmSync(destPath, { recursive: true, force: true });
    }
    
    // Create dest parent if needed
    fs.mkdirSync(path.dirname(destPath), { recursive: true });
    
    // Copy directory
    fs.cpSync(srcPath, destPath, { recursive: true });
    console.log(`  ✅ Copied ${name}`);
  } else {
    console.warn(`  ⚠️  Source not found: ${srcPath}`);
  }
}

console.log('\n✅ All projects built and consolidated successfully!');
