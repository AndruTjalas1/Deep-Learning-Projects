#!/usr/bin/env node

const { execSync } = require('child_process');

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

console.log('✅ All projects built successfully!');
