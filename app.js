const fs = require("graceful-fs");
const path = require("node:path");
const convert = require("gd-level-to-json");

// text after
function after(text, after) {
    return text.toString().substring(text.toString().indexOf(after) + after.length);
}

function before(text, before) {
    return text.toString().substring(0, text.toString().indexOf(before));
}

// take raw level data from levels folder and put it in levels_data folder
fs.readdirSync(path.join(__dirname, "levels")).forEach((file) => {
    if (file.endsWith(".gmd")) {
        const name = file.replace(".gmd", "");
        // read file data
        let level = fs.readFileSync(path.join(__dirname, "levels", file));
        level = after(level, "<s>");
        level = after(level, "<s>");
        level = before(level, "</s>");
        fs.writeFileSync(path.join(__dirname, "levels_data", name + ".json"), level.toString());
    }
});

// scan the levels_data folder
fs.readdirSync(path.join(__dirname, "levels_data")).forEach((file) => {
    if (file.endsWith(".gmd")) {
        const name = file.replace(".gmd", "");
        const level = convert(path.join(__dirname, "levels", file));
        fs.writeFileSync(path.join(__dirname, "levels", name + ".json"), JSON.stringify(level));
    }
});