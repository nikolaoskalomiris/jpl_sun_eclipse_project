/* import_eclipse_3d_from_csv_timecompress_centered_fixed_parentscale_patch.jsx
   FULL PATCH:
   - parent-scale-aware sizing helpers (handles RG Geo parent-null scaling)
   - robust Sun scale computed via explicit sun_texture_px and computeScalePercentForCompPx()
   - keeps all previous behaviour (time-compression, POI/rot bake, quantize-to-frame, etc.)
   - assumes moon_texture_px = 500.0 and sun_texture_px = 500.0 (as requested)
*/

(function () {
    // -------- USER OPTIONS (edit these) --------
    var sensor_pixels = 4096.0;   // must match Python's sensor_pixels and AE camera/comp width
    var sensor_width_mm = 21.44;  // diagonal measure of camera sensor in mm (keep same as Python)
    var f_mm = 35.0;			  // camera focal length
    var moon_texture_px = 500.0;  // measured in AE (you requested 500)
    var sun_texture_px = 500.0;   // explicit sun texture pixel width (requested 500)

    var useCompCenterAsAnchor = true;
    var extraOffsetX = 0;
    var extraOffsetY = 0;

    // --- Time compression options ---
    var enableTimeCompression = true;
    var desiredDurationSeconds = 2000.0 / 25.0; // 2000 frames @25fps = 80s
    var timeOffsetSeconds = 0.0;
    var centerTargetFraction = 0.5; // put min angular separation at middle

    var zMode = "relative";
    var zScale = 0.02;
    var zBase = 0;
    var invertZ = true;

    var useTimeSeconds = true;
    var addNodeMarkers = true;

    var posMultiplier = 16.488758;            // 0 = auto-calc
    var desiredScreenCoverage = 0.6;
    var scaleCorrection = 0;
    var desiredScalePercent = 500;
    var zMultiplier = 0;

    var createOrUseSunNull = true;
    var sunLayerName = "Sun Null";
    var sunUseSameScaleMapping = true;

    // camera baking mode: "rot" = bake Camera Track Null X/Y rotations
    //                    "poi" = bake camera Point of Interest (if you want camera POI baked directly)
    //                    "off" = don't bake camera
    var bakeCameraMode = "off"; // "rot" or "poi" or "off"
    var cameraTrackName = "Camera Track Null";

    // toggles if pan/tilt are mirrored in your setup
    var invertCameraX = false; // invert X rotation sign (pitch)
    var invertCameraY = true;  // invert Y rotation sign (yaw)
    var quantizeToFrame = true;
    // ------------------------------------------

    // ---------- Utility functions ----------
    function safeTrim(s) { if (s === null || typeof s === "undefined") return ""; s = String(s); return s.replace(/^\s+|\s+$/g, ""); }
    function isNumericString(s) { if (s === null || typeof s === "undefined") return false; var ss = String(s).replace(/^\s+|\s+$/g, ""); if (ss.length === 0) return false; var n = Number(ss); return !isNaN(n) && isFinite(n); }
    function splitCSVLine(line) { var res = []; var cur = ""; var inQuotes = false; for (var i = 0; i < line.length; i++) { var ch = line.charAt(i); if (ch === '"') { var next = (i + 1 < line.length) ? line.charAt(i + 1) : null; if (inQuotes && next === '"') { cur += '"'; i++; } else { inQuotes = !inQuotes; } } else if (ch === ',' && !inQuotes) { res.push(cur); cur = ""; } else { cur += ch; } } res.push(cur); return res; }
    function parseCSVText(text) {
        if (!text || text.length === 0) return { headers: [], H: {}, rows: [] };
        if (text.charCodeAt && text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
        text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
        var lines = text.split("\n");
        while (lines.length > 0) {
            var last = lines[lines.length - 1];
            if (typeof last !== "string" || !last.replace(/\s/g, "")) lines.pop(); else break;
        }
        if (lines.length === 0) return { headers: [], H: {}, rows: [] };
        var rawHeaderParts = splitCSVLine(lines[0]);
        var headers = [];
        for (var hi = 0; hi < rawHeaderParts.length; hi++) headers.push(safeTrim(rawHeaderParts[hi]));
        var H = {};
        for (var i = 0; i < headers.length; i++) {
            var name = headers[i];
            if (name === "") name = "__empty_" + i;
            if (typeof H[name] === "undefined") H[name] = i;
            else { var uniq = name + "_" + i; H[uniq] = i; headers[i] = uniq; }
        }
        var rows = [];
        for (var r = 1; r < lines.length; r++) {
            var rawLine = lines[r];
            if (typeof rawLine !== "string") continue;
            if (!rawLine.replace(/\s/g, "")) continue;
            var parts = splitCSVLine(rawLine);
            if (parts.length < headers.length) { for (var p = 0; p < headers.length - parts.length; p++) parts.push(""); }
            var obj = {};
            for (var c = 0; c < headers.length; c++) {
                var key = headers[c];
                var rawVal = (c < parts.length) ? parts[c] : "";
                var valStr = safeTrim(rawVal);
                obj[key] = isNumericString(valStr) ? Number(valStr) : valStr;
            }
            rows.push(obj);
        }
        return { headers: headers, H: H, rows: rows };
    }
    function computeMean(arr) { var s = 0; for (var i = 0; i < arr.length; i++) s += arr[i]; return arr.length ? (s / arr.length) : 0; }
    function computeMin(arr) { if (!arr || !arr.length) return 0; var m = arr[0]; for (var i =1;i<arr.length;i++) if (arr[i] < m) m = arr[i]; return m; }
    function computeMax(arr) { if (!arr || !arr.length) return 0; var M = arr[0]; for (var i =1;i<arr.length;i++) if (arr[i] > M) M = arr[i]; return M; }

    // ---- Script body ----
    try {
        var proj = app.project || app.newProject();
        var comp = proj.activeItem;
        if (!(comp && comp instanceof CompItem)) { alert("Open a composition and select the Moon Null layer."); return; }
        if (comp.selectedLayers.length !== 1) { alert("Select exactly ONE target layer (the Moon layer) first."); return; }
        var layer = comp.selectedLayers[0];

        var csvFile = File.openDialog("Select eclipse_keyframes_full.csv");
        if (!csvFile) { alert("No file selected."); return; }
        if (!csvFile.open("r")) { alert("Unable to open CSV."); return; }
        var csvText = csvFile.read();
        csvFile.close();

        var parsed = parseCSVText(csvText);
        var data = parsed.rows;
        if (!data || data.length === 0) { alert("CSV contained no data rows."); return; }

        if (!layer.threeDLayer) { if (confirm("Selected layer is not 3D. Enable 3D for this layer?")) layer.threeDLayer = true; else return; }

        // helper: find first child in comp parented to targetLayer that has a source.width
        function findFirstChildWithSourceWidth(compItem, targetLayer) {
            try {
                for (var li = 1; li <= compItem.numLayers; li++) {
                    try {
                        var L = compItem.layer(li);
                        if (L && L.parent && L.parent === targetLayer) {
                            if (L.source && typeof L.source.width === "number" && L.source.width > 0) return L;
                        }
                    } catch (e) {}
                }
            } catch (e) {}
            return null;
        }

        // helper: cumulative parent scale factor (multiplicative). Walks up parents and multiplies X-scale fractions.
        function getCumulativeParentScaleFactor(targetLayer) {
            try {
                var s = 1.0;
                var L = targetLayer;
                while (L && L.parent) {
                    L = L.parent;
                    try {
                        var tgrp = L.property("ADBE Transform Group") || L.property("Transform");
                        if (tgrp) {
                            var scProp = tgrp.property("Scale");
                            if (scProp) {
                                var scVal = scProp.valueAtTime(0, false);
                                var sx = (scVal && scVal.length) ? scVal[0] : scVal;
                                if (typeof sx === "number") {
                                    s *= (sx / 100.0);
                                }
                            }
                        }
                    } catch (e) {
                        // ignore and continue up the chain
                    }
                }
                return s;
            } catch (e) {
                return 1.0;
            }
        }

        // helper: compute percent scale (AE Scale property percent) required to make an object occupy desiredCompPx in composition pixels.
        // - Tries targetLayer.source.width * cumulative parent scale
        // - Falls back to first child with source.width (if present)
        // - Final fallback uses textureFallbackPx as if that image corresponds to 100% scale of object
        function computeScalePercentForCompPx(desiredCompPx, targetLayer, textureFallbackPx) {
            try {
                if (targetLayer && targetLayer.source && typeof targetLayer.source.width === "number" && targetLayer.source.width > 0) {
                    var srcW = targetLayer.source.width;
                    var cumScale = getCumulativeParentScaleFactor(targetLayer);
                    var effectiveSrcW = srcW * cumScale;
                    if (effectiveSrcW > 0) {
                        return (desiredCompPx / effectiveSrcW) * 100.0;
                    }
                }
            } catch (e) {}

            // try first child with source width
            try {
                var child = findFirstChildWithSourceWidth(comp, targetLayer);
                if (child && child.source && typeof child.source.width === "number" && child.source.width > 0) {
                    var srcW2 = child.source.width;
                    var cumScaleChild = getCumulativeParentScaleFactor(child);
                    var effectiveSrcW2 = srcW2 * cumScaleChild;
                    if (effectiveSrcW2 > 0) return (desiredCompPx / effectiveSrcW2) * 100.0;
                }
            } catch (e) {}

            // fallback to textureFallbackPx
            if (textureFallbackPx && textureFallbackPx > 0) {
                var percent = (desiredCompPx / textureFallbackPx) * 100.0;
                return percent;
            }

            return NaN;
        }

        // find or create Sun Null if requested
        var sunLayer = null;
        if (createOrUseSunNull) {
            for (var li = 1; li <= comp.numLayers; li++) {
                try { var L = comp.layer(li); if (L.name === sunLayerName) { sunLayer = L; break; } } catch (e) {}
            }
            if (!sunLayer) { sunLayer = comp.layers.addNull(); sunLayer.name = sunLayerName; }
            if (!sunLayer.threeDLayer) sunLayer.threeDLayer = true;
        }

        // collect arrays for auto-scaling & find min sep row
        var sxArr = [], syArr = [], scArr = [], mdArr = [], sunScArr = [];
        var minSep = Number.POSITIVE_INFINITY, minSepIndex = -1;
        var minFrame = Number.POSITIVE_INFINITY, maxFrame = Number.NEGATIVE_INFINITY;
        var minTime = Number.POSITIVE_INFINITY, maxTime = Number.NEGATIVE_INFINITY;
        for (var i = 0; i < data.length; i++) {
            var row = data[i];
            var sx = row["screen_x_px"], sy = row["screen_y_px"], sc = row["scale_pct"], sd = row["sun_px"], md = row["moon_distance_km"];
            var sep = (typeof row["angular_sep_deg"] === "number") ? row["angular_sep_deg"] : Number.POSITIVE_INFINITY;
            if (sep < minSep) { minSep = sep; minSepIndex = i; }
            if (typeof sx === "number") sxArr.push(sx);
            if (typeof sy === "number") syArr.push(sy);
            if (typeof sc === "number") scArr.push(sc);
            if (typeof sd === "number") sunScArr.push(sd);
            if (typeof md === "number") mdArr.push(md);
            if (typeof row["frame"] === "number") {
                if (row["frame"] < minFrame) minFrame = row["frame"];
                if (row["frame"] > maxFrame) maxFrame = row["frame"];
            }
            if (typeof row["time_s"] === "number") {
                if (row["time_s"] < minTime) minTime = row["time_s"];
                if (row["time_s"] > maxTime) maxTime = row["time_s"];
            }
        }
        if (sxArr.length === 0) { alert("No numeric screen_x_px found."); return; }

        var csvHasFrame = isFinite(minFrame) && isFinite(maxFrame) && (maxFrame > minFrame);
        var csvHasTime = isFinite(minTime) && isFinite(maxTime) && (maxTime > minTime);

        // compute meanDist for z mapping
        var meanDist = computeMean(mdArr);
        var zSign = invertZ ? -1 : 1;
        var zArr = [];
        for (var i = 0; i < data.length; i++) {
            var md = data[i]["moon_distance_km"];
            var z = (zMode === "absolute") ? zSign * (md * zScale) + zBase : zSign * ((md - meanDist) * zScale) + zBase;
            zArr.push(z);
        }

        // mm per pixel (sensor model)
        var mm_per_pixel = sensor_width_mm / sensor_pixels;

        // Map CSV sensor pixels to comp pixels: baseline mapping (no crazy auto-stretch)
        var sensorToComp = comp.width / sensor_pixels;

        // Force posMultiplier to baseline mapping (1 sensor px => sensorToComp comp px).
        posMultiplier = (posMultiplier > 0) ? posMultiplier : sensorToComp;

        // scaleCorrection
        var meanCSVScale = computeMean(scArr);
        if (scaleCorrection <= 0) {
            scaleCorrection = (meanCSVScale > 0) ? (desiredScalePercent / meanCSVScale) : 1.0;
        }

        // zMultiplier
        var minZ = computeMin(zArr), maxZ = computeMax(zArr);
        var computedZRange = Math.abs(maxZ - minZ);
        if (zMultiplier <= 0) {
            var desiredZRange = Math.max(10, comp.width * 0.02);
            zMultiplier = (computedZRange > 0) ? (desiredZRange / computedZRange) : 1.0;
        }

        // anchor
        var anchorX = useCompCenterAsAnchor ? (comp.width / 2) : layer.property("Transform").property("Position").valueAtTime(0, false)[0];
        var anchorY = useCompCenterAsAnchor ? (comp.height / 2) : layer.property("Transform").property("Position").valueAtTime(0, false)[1];

        // create/find Camera Track Null
        var camTrack = null;
        for (var li = 1; li <= comp.numLayers; li++) {
            try { var L = comp.layer(li); if (L.name === cameraTrackName) { camTrack = L; break; } } catch (e) {}
        }
        if (!camTrack) { camTrack = comp.layers.addNull(); camTrack.name = cameraTrackName; }
        if (!camTrack.threeDLayer) camTrack.threeDLayer = true;

        // summary to user
        var summary = "Auto values:\n sensorToComp=" + (Math.round(sensorToComp*1000)/1000) + " posMultiplier=" + (Math.round(posMultiplier*100)/100) +
                      " scaleCorrection=" + (Math.round(scaleCorrection*100)/100) + " zMultiplier=" + (Math.round(zMultiplier*100)/100) +
                      "\nTime compression: " + (enableTimeCompression ? ("ENABLED -> " + desiredDurationSeconds + "s, centerTargetFraction=" + centerTargetFraction) : "OFF");
        alert(summary);

        app.beginUndoGroup("Import Eclipse CSV (timecompress centered fixed + parent-scale)");

        // clear keys on Moon layer
        var posProp = layer.property("Transform").property("Position");
        var sclProp = layer.property("Transform").property("Scale");
        var markerProp = layer.property("Marker");
        try { while (posProp.numKeys) posProp.removeKey(1); } catch(e) {}
        try { while (sclProp.numKeys) sclProp.removeKey(1); } catch(e) {}
        if (markerProp) { try { while (markerProp.numKeys) markerProp.removeKey(1); } catch(e) {} }

        // prepare Sun properties
        var sunPosProp = null, sunSclProp = null;
        if (sunLayer) {
            try { while (sunLayer.property("Transform").property("Position").numKeys) sunLayer.property("Transform").property("Position").removeKey(1); } catch(e) {}
            try { while (sunLayer.property("Transform").property("Scale").numKeys) sunLayer.property("Transform").property("Scale").removeKey(1); } catch(e) {}
            sunPosProp = sunLayer.property("Transform").property("Position");
            sunSclProp = sunLayer.property("Transform").property("Scale");
        }

        // rotation props on camTrack (for "rot" mode)
        var rotXProp = camTrack.property("Transform").property("X Rotation");
        var rotYProp = camTrack.property("Transform").property("Y Rotation");
        try { while (rotXProp.numKeys) rotXProp.removeKey(1); } catch(e){}
        try { while (rotYProp.numKeys) rotYProp.removeKey(1); } catch(e){}

        // camera POI discovery
        var activeCam = null;
        try { activeCam = comp.activeCamera; } catch(e) { activeCam = null; }
        function findPropertyByMatchOrName(parentProp, matchCandidates, nameContainsLower) {
            for (var mi = 0; mi < matchCandidates.length; mi++) {
                try { var p = parentProp.property(matchCandidates[mi]); if (p) return p; } catch(e){}
            }
            try {
                for (var i = 1; i <= parentProp.numProperties; i++) {
                    try {
                        var p = parentProp.property(i);
                        var mn = (p.matchName || "").toString().toLowerCase();
                        var nm = (p.name || "").toString().toLowerCase();
                        if (mn && mn.indexOf("point") !== -1) return p;
                        if (nm && nameContainsLower && nm.indexOf(nameContainsLower) !== -1) return p;
                    } catch(e){}
                }
            } catch(e){}
            return null;
        }

        var camPOIProp = null;
        if (activeCam && activeCam instanceof Layer) {
            var tgroup = activeCam.property("ADBE Transform Group") || activeCam.property("Transform");
            if (tgroup) camPOIProp = findPropertyByMatchOrName(tgroup, ["ADBE Point of Interest", "Point of Interest", "Point of Interest"], "point");
        }

        // if poi mode requested and no POI found, create fallback camera (Two-node) and get its POI
        var createdFallbackCam = null;
        if (bakeCameraMode === "poi" && !camPOIProp) {
            var originalPos = null;
            try { if (activeCam && activeCam instanceof Layer) { originalPos = (activeCam.property("ADBE Transform Group") || activeCam.property("Transform")).property("Position").valueAtTime(0,false); } } catch(e){ originalPos = null; }
            var camPos2 = (originalPos && originalPos.length >= 2) ? [originalPos[0], originalPos[1]] : [comp.width/2, comp.height/2];
            try {
                createdFallbackCam = comp.layers.addCamera("ECLIPSE_Camera_POI", camPos2);
                if (originalPos && originalPos.length === 3) createdFallbackCam.property("Transform").property("Position").setValue(originalPos);
                createdFallbackCam.threeDLayer = true;
                var tg = createdFallbackCam.property("ADBE Transform Group") || createdFallbackCam.property("Transform");
                camPOIProp = findPropertyByMatchOrName(tg, ["ADBE Point of Interest","Point of Interest"], "point");
                if (activeCam && activeCam instanceof Layer) activeCam.enabled = false;
            } catch(e) {
                // ignore
            }
        }

        var fps = comp.frameRate;

        // compute remap scaling & center shift
        var orig_min = csvHasFrame ? minFrame : (csvHasTime ? minTime : 0);
        var orig_max = csvHasFrame ? maxFrame : (csvHasTime ? maxTime : (data.length - 1));
        var orig_span = (orig_max - orig_min);
        if (orig_span <= 0) orig_span = Math.max(1, data.length - 1);
        var scale = desiredDurationSeconds / orig_span;

        // compute center_orig (minSep) as numeric value of frame/time/index
        var center_orig;
        if (minSepIndex >= 0) {
            if (csvHasFrame && typeof data[minSepIndex]["frame"] === "number") center_orig = data[minSepIndex]["frame"];
            else if (csvHasTime && typeof data[minSepIndex]["time_s"] === "number") center_orig = data[minSepIndex]["time_s"];
            else center_orig = minSepIndex;
        } else {
            if (csvHasFrame) center_orig = (minFrame + maxFrame) / 2.0;
            else if (csvHasTime) center_orig = (minTime + maxTime) / 2.0;
            else center_orig = (data.length - 1) / 2.0;
        }
        var center_landed = timeOffsetSeconds + (center_orig - orig_min) * scale;
        var desired_center_time = timeOffsetSeconds + centerTargetFraction * desiredDurationSeconds;
        var center_shift = desired_center_time - center_landed;

        // base az/alt for rotations (frame 0 row)
        var baseAz = (typeof data[0]["sun_az_deg"] === "number") ? data[0]["sun_az_deg"] : 0;
        var baseAlt = (typeof data[0]["sun_alt_deg"] === "number") ? data[0]["sun_alt_deg"] : 0;

        // iterate rows and set keys (snap to frame grid if requested)
        for (var k = 0; k < data.length; k++) {
            var row = data[k];

            // original numeric coordinate (frame or time or index)
            var orig_val;
            if (csvHasFrame && typeof row["frame"] === "number") orig_val = row["frame"];
            else if (csvHasTime && typeof row["time_s"] === "number") orig_val = row["time_s"];
            else orig_val = k;

            // remapped time (seconds)
            var t;
            if (enableTimeCompression) {
                t = timeOffsetSeconds + (orig_val - orig_min) * scale + center_shift;
                var tmin = timeOffsetSeconds;
                var tmax = timeOffsetSeconds + desiredDurationSeconds;
                if (t < tmin) t = tmin;
                if (t > tmax) t = tmax;
            } else {
                if (useTimeSeconds && typeof row["time_s"] === "number") t = row["time_s"];
                else if (typeof row["frame"] === "number") t = row["frame"] / fps;
                else t = k / fps;
            }

            // quantize time to nearest frame if requested
            if (quantizeToFrame) {
                var frameNum = Math.round(t * fps);
                t = frameNum / fps;
            }

            var sx = row["screen_x_px"], sy = row["screen_y_px"];
            var md = row["moon_distance_km"], scpct = row["scale_pct"];
            var sun_px = row["sun_px"];
            var nodeVal = row["node"];

            var zRaw = (zMode === "absolute") ? zSign * (md * zScale) + zBase : zSign * ((md - meanDist) * zScale) + zBase;
            var z = zRaw * zMultiplier;

            // map CSV sensor pixels -> comp pixels
            var px = anchorX + (sx + extraOffsetX) * posMultiplier;
            var py = anchorY - (sy + extraOffsetY) * posMultiplier;

            // set Moon keys (one key per quantized t)
            try { posProp.setValueAtTime(t, [px, py, z]); } catch(e){}
            try { sclProp.setValueAtTime(t, [scpct * scaleCorrection, scpct * scaleCorrection, scpct * scaleCorrection]); } catch(e){}

            // Sun layer: set at comp center (z aligned with moon for consistent visual overlap)
            if (sunLayer && sunPosProp && sunSclProp) {
                var sunPx = anchorX + (0 + extraOffsetX) * posMultiplier;
                var sunPy = anchorY - (0 + extraOffsetY) * posMultiplier;
                try { sunPosProp.setValueAtTime(t, [sunPx, sunPy, z]); } catch(e){}

                if (typeof sun_px === "number" && sunUseSameScaleMapping) {

                    // === robust Sun scaling: prefer moon-relative mapping (avoids parent/source-size surprises) ===
                    // Try to preserve the Moon:Sun pixel ratio from CSV using the same applied Moon scale first.
                    var moon_px_val = (typeof row["moon_px"] === "number") ? row["moon_px"] : null;
                    // moonScaleApplied = applied AE Scale percent (not percent/100) — but since both are percentages we keep percent units.
                    var moonScaleApplied = scpct * scaleCorrection;

                    var sunScalePct = NaN;

                    if (moon_px_val && moon_px_val > 0) {
                        // keep exact Moon:Sun pixel ratio from CSV
                        sunScalePct = moonScaleApplied * (sun_px / moon_px_val);
                    }

                    // fallback if the moon_px info isn't present or produced NaN:
                    if (!isFinite(sunScalePct) || isNaN(sunScalePct)) {
                        // desired visible width in comp pixels
                        var desiredCompPx = sun_px * posMultiplier;

                        // try parent-aware computation (computeScalePercentForCompPx was defined earlier)
                        var computed = computeScalePercentForCompPx(desiredCompPx, sunLayer, sun_texture_px);
                        if (isFinite(computed) && !isNaN(computed)) {
                            // apply global scaleCorrection to keep parity with Moon mapping
                            sunScalePct = computed * scaleCorrection;
                        } else {
                            // ultimate fallback: texture-relative mapping (legacy behavior)
                            sunScalePct = (sun_px / moon_texture_px) * 100.0 * scaleCorrection;
                        }
                    }

                    // clamp to sane range to avoid runaway percents due to weird layer data
                    sunScalePct = Math.max(0.01, Math.min(10000, sunScalePct));

                    if (!isNaN(sunScalePct) && isFinite(sunScalePct)) {
                        try { sunSclProp.setValueAtTime(t, [sunScalePct, sunScalePct, sunScalePct]); } catch(e){}
                    }
                }
            }

            // camera baking
            if (bakeCameraMode === "rot") {
                // bake cameraTrack rotations using sun_az_deg/sun_alt_deg deltas
                if (typeof row["sun_az_deg"] === "number" && typeof row["sun_alt_deg"] === "number") {
                    var az = row["sun_az_deg"]; var alt = row["sun_alt_deg"];
                    var dAz = az - baseAz; var dAlt = alt - baseAlt;
                    var yaw = (invertCameraY ? -1 : 1) * -dAz;
                    var pitch = (invertCameraX ? -1 : 1) * dAlt;
                    try { rotYProp.setValueAtTime(t, yaw); } catch(e){}
                    try { rotXProp.setValueAtTime(t, pitch); } catch(e){}
                }
            } else if (bakeCameraMode === "poi") {
                if (camPOIProp && typeof row["sun_az_deg"] === "number" && typeof row["sun_alt_deg"] === "number") {
                    var az = row["sun_az_deg"]; var alt = row["sun_alt_deg"];
                    var dAz_deg = az - baseAz; var dAlt_deg = alt - baseAlt;
                    var dAz_rad = dAz_deg * Math.PI / 180.0;
                    var dAlt_rad = dAlt_deg * Math.PI / 180.0;
                    var poi_offset_x = (f_mm * Math.tan(dAz_rad)) / mm_per_pixel;
                    var poi_offset_y = (f_mm * Math.tan(dAlt_rad)) / mm_per_pixel;
                    var poi_x_raw = anchorX + (poi_offset_x + extraOffsetX) * posMultiplier;
                    var poi_y_raw = anchorY - (poi_offset_y + extraOffsetY) * posMultiplier;
                    var maxBound = comp.width * 20;
                    var minBoundX = anchorX - maxBound, maxBoundX = anchorX + maxBound;
                    var minBoundY = anchorY - maxBound, maxBoundY = anchorY + maxBound;
                    var poi_x = Math.max(minBoundX, Math.min(maxBoundX, poi_x_raw));
                    var poi_y = Math.max(minBoundY, Math.min(maxBoundY, poi_y_raw));
                    var poi_z = z;
                    try { camPOIProp.setValueAtTime(t, [poi_x, poi_y, poi_z]); } catch(e) { try { camPOIProp.setValueAtTime(t, [poi_x, poi_y]); } catch(e){} }
                }
            }

            if (addNodeMarkers && (nodeVal === 1 || nodeVal === -1)) {
                var label = (nodeVal === 1) ? "Ascending Node" : "Descending Node";
                var m = new MarkerValue(label);
                try { markerProp.setValueAtTime(t, m); } catch(e){}
            }
        }

        // attempt to auto-parent active camera to camTrack (works in many AE builds)
        try {
            var tryCam = comp.activeCamera;
            if (tryCam && tryCam instanceof Layer) {
                tryCam.parent = camTrack;
            }
        } catch (e) {}

        app.endUndoGroup();

        var finishMsg = "Import complete. Keys created: approx " + data.length + " (one per CSV row quantized to frames).\n";
        finishMsg += (bakeCameraMode === "rot") ? "Camera Track Null rotations baked." : (bakeCameraMode === "poi" ? "Camera POI baked." : "Camera baking skipped.");
        if (createdFallbackCam) finishMsg += "\nCreated fallback camera: " + createdFallbackCam.name;
        alert(finishMsg);

    } catch (err) {
        alert("Error: " + err.message);
        throw err;
    }

})();
