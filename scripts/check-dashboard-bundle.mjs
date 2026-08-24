import { readFile, readdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const outputDirectory = resolve(scriptDirectory, "..", "static", "dashboard");
const indexHtml = await readFile(resolve(outputDirectory, "index.html"), "utf8");
const entryMatch = indexHtml.match(/src="\/assets\/([^"/]+\.js)"/);

if (!entryMatch) {
  throw new Error("Dashboard build does not declare a JavaScript entry in index.html");
}

const javascriptFiles = (await readdir(outputDirectory)).filter((name) => name.endsWith(".js"));
const assets = await Promise.all(
  javascriptFiles.map(async (name) => ({ name, bytes: (await stat(resolve(outputDirectory, name))).size })),
);
const entry = assets.find(({ name }) => name === entryMatch[1]);
const largest = assets.reduce((current, asset) => (asset.bytes > current.bytes ? asset : current));

if (!entry) {
  throw new Error(`Dashboard entry ${entryMatch[1]} is missing from the build output`);
}

const maxEntryBytes = 300 * 1024;
const maxChunkBytes = 300 * 1024;
if (entry.bytes > maxEntryBytes) {
  throw new Error(`Dashboard entry ${entry.name} is ${(entry.bytes / 1024).toFixed(1)}KB; budget is 300KB`);
}
if (largest.bytes > maxChunkBytes) {
  throw new Error(`Dashboard chunk ${largest.name} is ${(largest.bytes / 1024).toFixed(1)}KB; budget is 300KB`);
}

console.log(
  `Dashboard bundle budget passed: entry ${(entry.bytes / 1024).toFixed(1)}KB, largest chunk ${(largest.bytes / 1024).toFixed(1)}KB`,
);
