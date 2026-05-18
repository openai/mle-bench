import { Dolos } from "@dodona/dolos-lib";
import fs from 'fs';
import path from 'path';

// Function to parse command line arguments
function parseArgs(args) {
  const parsed = { files: [] };
  for (let i = 2; i < args.length; i++) {
    if (args[i] === '--files' || args[i] === '-f') {
      while (++i < args.length && !args[i].startsWith('-')) {
        parsed.files.push(args[i]);
      }
      i--;
    }
  }
  return parsed;
}

// Parse command line arguments
const args = parseArgs(process.argv);

// Use command line arguments
const files = args.files;

// Filter out non-existent files
const existingFiles = files.filter(file => fs.existsSync(file) && fs.statSync(file).isFile());

const dolos = new Dolos();
const report = await dolos.analyzePaths(existingFiles);

const pairs = report.allPairs();
const simplifiedPairs = pairs.map(pair => ({
  leftFile: pair.leftFile.path,
  rightFile: pair.rightFile.path,
  similarity: pair.similarity
}));

const jsonOutput = JSON.stringify(simplifiedPairs, null, 2);
console.log(jsonOutput); // Print the JSON output to stdout