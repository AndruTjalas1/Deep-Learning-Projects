#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');

const projects = [
  { name: 'hub', path: '.' },
  { name: 'DNN', path: '../Deep Neural Network/frontend' },
  { name: 'GAN', path: '../GAN/Frontend' },
  { name: 'RNN', path: '../RNN/frontend' },
];

console.log('🚀 Building monorepo...\n');

for (const project of projects) {
  console.log(`📦 Building ${project.name}...`);
  
  try {
    // Install dependencies
    console.log(`  → Installing dependencies for ${project.name}...`);
    execSync(`cd "${project.path}" && npm install`, { stdio: 'inherit' });
    
    // Build
    console.log(`  → Building ${project.name}...`);
    execSync(`cd "${project.path}" && npm run build`, { stdio: 'inherit' });
    
    console.log(`✅ ${project.name} built successfully\n`);
  } catch (error) {
    console.error(`❌ Failed to build ${project.name}`);
    process.exit(1);
  }
}

console.log('✅ All projects built successfully!');
