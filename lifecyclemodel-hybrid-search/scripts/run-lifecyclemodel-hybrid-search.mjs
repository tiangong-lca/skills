#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runSimpleInputWrapper } from '../../shared/run-tiangong-cli-wrapper.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const SKILL_DIR = path.resolve(SCRIPT_DIR, '..');

runSimpleInputWrapper({
  importMetaUrl: import.meta.url,
  argv: process.argv.slice(2),
  command: ['search', 'lifecyclemodel'],
  defaultInputFile: path.join(SKILL_DIR, 'assets', 'example-request.json'),
});
