const fs = require("graceful-fs");
const path = require("node:path");
const convert = require("gd-level-to-json");

console.log(convert(fs.readFileSync(path.resolve(__dirname, "level.txt")).toString()));