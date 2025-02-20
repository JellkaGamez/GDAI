const fs = require("graceful-fs");
const path = require("node:path");
const convert = require("gd-level-to-json");

// scan the levels folder
fs.readdirSync(path.join(__dirname, "levels")).forEach((file) => {
    if (file.endsWith(".gmd")) {
        const name = file.replace(".gmd", "");
        const level = convert(path.join(__dirname, "levels", file));
        fs.writeFileSync(path.join(__dirname, "levels", name + ".json"), JSON.stringify(level));
    }
});