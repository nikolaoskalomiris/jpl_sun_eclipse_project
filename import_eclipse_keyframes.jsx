/**
 * import_eclipse_keyframes.jsx (updated)
 *
 * Robust importer that aligns the CSV row closest to time_s_center==0.0
 * exactly to the AE center frame (metadata ae_center_frame).
 *
 * Places results in eclipse_keyframe_mapping.json next to the CSV for inspection.
 *
 * Note: AE indexing assumption: mapping produced here is 0-based frame indices.
 * If you need 1-based AE frames, tweak the ae_frame_base_1 flag below.
 */
(function () {
    function readFileText(path) {
        var f = new File(path);
        if (!f.exists) {
            alert("File not found: " + path);
            return null;
        }
        f.open("r");
        var txt = f.read();
        f.close();
        return txt;
    }

    function parseCSV(text) {
        var lines = text.split(/\r\n|\n/);
        while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
        if (lines.length < 2) return [];
        var header = lines[0].split(",");
        var rows = [];
        for (var i = 1; i < lines.length; i++) {
            var parts = lines[i].split(",");
            while (parts.length < header.length) parts.push("");
            var obj = {};
            for (var j = 0; j < header.length; j++) {
                obj[header[j].trim()] = parts[j] !== undefined ? parts[j].trim() : "";
            }
            rows.push(obj);
        }
        return rows;
    }

    function parseJSONSafe(text) {
        try { return JSON.parse(text); } catch (e) { return null; }
    }

    var csvFile = File.openDialog("Select eclipse_keyframes_full.csv", "*.csv");
    if (!csvFile) {
        alert("No CSV chosen. Exiting.");
        return;
    }
    var csvText = readFileText(csvFile.fsName);
    if (!csvText) { alert("Failed to read CSV."); return; }
    var rows = parseCSV(csvText);
    if (!rows || rows.length === 0) { alert("CSV parsing failed or empty CSV"); return; }

    var folderPath = csvFile.parent.fsName;
    var metaPath = folderPath + "/" + "center_metadata.json";
    var metaText = readFileText(metaPath);
    var meta = parseJSONSafe(metaText);

    var frames = meta && meta.frames ? meta.frames : rows.length;
    var fps = meta && meta.fps ? meta.fps : 25;
    var real_duration_s = meta && meta.real_duration_s ? meta.real_duration_s : (frames / fps) * 135;
    var frames_per_real_second = frames / real_duration_s;
    var ae_center_frame = (meta && meta.ae_center_frame) ? meta.ae_center_frame : 1000;
    var ae_frame_base_1 = false; // set true if you want AE 1-based frames

    // provisional mapping
    var mapping = [];
    for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        var ts_center = parseFloat(r["time_s_center"]);
        if (isNaN(ts_center)) {
            var ts_from_start = parseFloat(r["time_s_from_start"]);
            if (!isNaN(ts_from_start)) ts_center = ts_from_start - (real_duration_s / 2.0);
            else ts_center = 0.0;
        }
        var frame_offset = Math.round(ts_center * frames_per_real_second);
        var target_frame = ae_center_frame + frame_offset;
        mapping.push({
            csv_index: i,
            csv_frame: parseInt(r["frame"] || i),
            time_s_center: ts_center,
            time_s_from_start: parseFloat(r["time_s_from_start"] || (ts_center + (real_duration_s/2.0))),
            provisional_target_frame: target_frame,
            screen_x_px: parseFloat(r["screen_x_px"] || 0),
            screen_y_px: parseFloat(r["screen_y_px"] || 0),
            alpha_deg: parseFloat(r["alpha_deg"] || 0),
            beta_deg: parseFloat(r["beta_deg"] || 0)
        });
    }

    // Find row whose |time_s_center| is minimal (closest to event center)
    var minAbs = Number.POSITIVE_INFINITY;
    var idxMin = 0;
    for (var k = 0; k < mapping.length; k++) {
        var a = Math.abs(mapping[k].time_s_center);
        if (a < minAbs) { minAbs = a; idxMin = k; }
    }
    // provisional mapped frame for that row:
    var provisional_center_mapped = mapping[idxMin].provisional_target_frame;
    // compute delta to align it to ae_center_frame
    var delta = ae_center_frame - provisional_center_mapped;

    // apply alignment delta to all mappings
    for (var k = 0; k < mapping.length; k++) {
        mapping[k].target_frame = mapping[k].provisional_target_frame + delta;
        if (ae_frame_base_1) mapping[k].target_frame = mapping[k].target_frame + 1;
    }

    // write mapping JSON
    var mappingFile = new File(folderPath + "/" + "eclipse_keyframe_mapping.json");
    mappingFile.encoding = "UTF-8";
    mappingFile.open("w");
    mappingFile.write(JSON.stringify(mapping, null, 2));
    mappingFile.close();
    alert("Wrote mapping to: " + mappingFile.fsName);

    // print diagnostic samples
    function dbg(msg) { $.writeln(msg); }
    dbg("Sample mappings (after alignment):");
    var sampleIndices = [0, 1, Math.max(0, Math.floor(frames/2) - 1), Math.max(0, Math.floor(frames/2)), mapping.length - 1];
    for (var s = 0; s < sampleIndices.length; s++) {
        var si = sampleIndices[s];
        if (si >= 0 && si < mapping.length) {
            var m = mapping[si];
            dbg("csv_row=" + m.csv_index + " csv_frame=" + m.csv_frame + " time_s_center=" + m.time_s_center.toFixed(3) + " => AE_frame=" + m.target_frame);
        }
    }

    // extra diagnostic: verify the aligned center
    var aligned_center_mapped = mapping[idxMin].target_frame;
    dbg("Closest row index to center:", idxMin, "time_s_center:", mapping[idxMin].time_s_center, "maps to AE_frame:", aligned_center_mapped, " (should be ae_center_frame:", ae_center_frame, ")");

    // Example: set keyframes on selected layer (commented)
    /*
    if (app.project && app.project.activeItem && app.project.activeItem.selectedLayers.length > 0) {
        app.beginUndoGroup("Apply eclipse keyframes");
        var comp = app.project.activeItem;
        var layer = comp.selectedLayers[0];
        var prop = layer.property("Transform").property("Position");
        for (var i = 0; i < mapping.length; i++) {
            var m = mapping[i];
            var t = m.target_frame / fps; // seconds in comp
            var x = comp.width/2 + m.screen_x_px;
            var y = comp.height/2 - m.screen_y_px;
            prop.setValueAtTime(t, [x, y]);
        }
        app.endUndoGroup();
    } else {
        alert("To apply keyframes: select a composition and a layer, then re-run and uncomment the keyframe block.");
    }
    */
})();
