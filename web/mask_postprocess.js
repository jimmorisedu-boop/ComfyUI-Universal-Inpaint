// Pure-JS real-time Grow (2D Euclidean distance transform) and Feather
// (edge-clamped separable Gaussian / 3-pass box blur) matching
// mask_core.postprocess_mask() in O(W * H) time.

function edt1d(f, n, d, v, z) {
    let k = 0;
    v[0] = 0;
    z[0] = -Infinity;
    z[1] = Infinity;
    for (let q = 1; q < n; q += 1) {
        const fqPlusQ2 = f[q] + q * q;
        let s = (fqPlusQ2 - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k]));
        while (s <= z[k]) {
            k -= 1;
            s = (fqPlusQ2 - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k]));
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q += 1) {
        while (z[k + 1] < q) k += 1;
        const dx = q - v[k];
        d[q] = dx * dx + f[v[k]];
    }
}

export function computeSquaredDistanceField(alpha, width, height) {
    const w = Math.max(0, Math.round(Number(width) || 0));
    const h = Math.max(0, Math.round(Number(height) || 0));
    const len = w * h;
    if (!alpha || len <= 0 || alpha.length < len) return null;

    let maxVal = 0;
    for (let i = 0; i < len; i += 1) {
        if (alpha[i] > maxVal) maxVal = alpha[i];
    }
    if (maxVal === 0) return null;

    const thr = maxVal >= 128 ? 128 : Math.max(1, Math.floor(maxVal * 0.5));
    const fillVal = maxVal >= 128 ? 255 : maxVal;
    const inf = (w + h) * (w + h) * 4;
    const maxDim = Math.max(w, h);
    const f = new Float64Array(maxDim);
    const d = new Float64Array(maxDim);
    const v = new Int32Array(maxDim);
    const z = new Float64Array(maxDim + 1);
    const distSq = new Float32Array(len);

    for (let x = 0; x < w; x += 1) {
        for (let y = 0; y < h; y += 1) {
            f[y] = alpha[y * w + x] >= thr ? 0 : inf;
        }
        edt1d(f, h, d, v, z);
        for (let y = 0; y < h; y += 1) {
            const val = d[y];
            distSq[y * w + x] = val < inf ? val : inf;
        }
    }

    for (let y = 0; y < h; y += 1) {
        const row = y * w;
        for (let x = 0; x < w; x += 1) {
            f[x] = distSq[row + x];
        }
        edt1d(f, w, d, v, z);
        for (let x = 0; x < w; x += 1) {
            const val = d[x];
            distSq[row + x] = val < inf ? val : inf;
        }
    }

    return { distSq, fillVal };
}

export function applyGrowFromDistanceField(alpha, distField, radius, out = new Uint8Array(alpha.length)) {
    const len = alpha.length;
    if (!distField || !(radius > 0)) {
        out.set(alpha);
        return out;
    }
    const r = Number(radius);
    const r2 = r * r;
    const outer = r + 1;
    const outer2 = outer * outer;
    const { distSq, fillVal } = distField;
    for (let i = 0; i < len; i += 1) {
        const src = alpha[i];
        const d2 = distSq[i];
        if (d2 <= r2) {
            out[i] = src >= fillVal ? src : fillVal;
        } else if (d2 < outer2) {
            const soft = Math.round(fillVal * (outer - Math.sqrt(d2)));
            out[i] = src >= soft ? src : soft;
        } else {
            out[i] = src;
        }
    }
    return out;
}

function boxesForGauss(sigma, count = 3) {
    const wIdeal = Math.sqrt((12 * sigma * sigma) / count + 1);
    let wl = Math.floor(wIdeal);
    if (wl % 2 === 0) wl -= 1;
    wl = Math.max(1, wl);
    const wu = wl + 2;
    const mIdeal = (12 * sigma * sigma - count * wl * wl - 4 * count * wl - 3 * count) / (-4 * wl - 4);
    const m = Math.round(mIdeal);
    const sizes = [];
    for (let i = 0; i < count; i += 1) sizes.push(i < m ? wl : wu);
    return sizes;
}

function boxBlurHorizontalFloat(src, dst, width, height, radius) {
    if (radius <= 0) {
        dst.set(src);
        return;
    }
    const inv = 1 / (2 * radius + 1);
    const maxX = width - 1;
    for (let y = 0; y < height; y += 1) {
        const row = y * width;
        let acc = (radius + 1) * src[row];
        for (let k = 1; k <= radius; k += 1) {
            acc += src[row + (k < maxX ? k : maxX)];
        }
        for (let x = 0; x < width; x += 1) {
            dst[row + x] = acc * inv;
            if (x + 1 < width) {
                const exitIdx = x - radius > 0 ? x - radius : 0;
                const enterIdx = x + 1 + radius < maxX ? x + 1 + radius : maxX;
                acc += src[row + enterIdx] - src[row + exitIdx];
            }
        }
    }
}

function boxBlurVerticalFloat(src, dst, width, height, radius, colAcc) {
    if (radius <= 0) {
        dst.set(src);
        return;
    }
    const inv = 1 / (2 * radius + 1);
    const maxY = height - 1;
    for (let x = 0; x < width; x += 1) {
        colAcc[x] = (radius + 1) * src[x];
    }
    for (let k = 1; k <= radius; k += 1) {
        const row = (k < maxY ? k : maxY) * width;
        for (let x = 0; x < width; x += 1) {
            colAcc[x] += src[row + x];
        }
    }
    for (let y = 0; y < height; y += 1) {
        const outRow = y * width;
        const exitRow = (y - radius > 0 ? y - radius : 0) * width;
        const enterRow = (y + 1 + radius < maxY ? y + 1 + radius : maxY) * width;
        for (let x = 0; x < width; x += 1) {
            dst[outRow + x] = colAcc[x] * inv;
            if (y + 1 < height) {
                colAcc[x] += src[enterRow + x] - src[exitRow + x];
            }
        }
    }
}

export function applyFeatherBlur(src, width, height, featherRadius, out = new Uint8Array(src.length), scratch = null) {
    const w = Math.max(0, Math.round(Number(width) || 0));
    const h = Math.max(0, Math.round(Number(height) || 0));
    const len = w * h;
    if (!src || len <= 0 || !(featherRadius > 0.05)) {
        if (src && out && out !== src) out.set(src);
        return out;
    }

    const r = Number(featherRadius);
    const sigma = Math.max(0.1, r / 2.0);
    const bufA = scratch?.bufA?.length >= len ? scratch.bufA : new Float32Array(len);

    if (sigma <= 6.0) {
        const kRadius = Math.max(1, Math.min(32, Math.ceil(r)));
        const kernel = new Float32Array(kRadius + 1);
        const denom = 2 * sigma * sigma;
        let sum = 1;
        kernel[0] = 1;
        for (let d = 1; d <= kRadius; d += 1) {
            const weight = Math.exp(-(d * d) / denom);
            kernel[d] = weight;
            sum += 2 * weight;
        }
        for (let d = 0; d <= kRadius; d += 1) kernel[d] /= sum;

        const maxX = w - 1;
        for (let y = 0; y < h; y += 1) {
            const row = y * w;
            for (let x = 0; x < w; x += 1) {
                let acc = src[row + x] * kernel[0];
                for (let d = 1; d <= kRadius; d += 1) {
                    const lx = x - d > 0 ? x - d : 0;
                    const rx = x + d < maxX ? x + d : maxX;
                    acc += (src[row + lx] + src[row + rx]) * kernel[d];
                }
                bufA[row + x] = acc;
            }
        }

        const upRows = new Int32Array(kRadius + 1);
        const downRows = new Int32Array(kRadius + 1);
        const maxY = h - 1;
        for (let y = 0; y < h; y += 1) {
            const row = y * w;
            for (let d = 1; d <= kRadius; d += 1) {
                upRows[d] = (y - d > 0 ? y - d : 0) * w;
                downRows[d] = (y + d < maxY ? y + d : maxY) * w;
            }
            for (let x = 0; x < w; x += 1) {
                let acc = bufA[row + x] * kernel[0];
                for (let d = 1; d <= kRadius; d += 1) {
                    acc += (bufA[upRows[d] + x] + bufA[downRows[d] + x]) * kernel[d];
                }
                const rounded = Math.round(acc);
                out[row + x] = rounded <= 0 ? 0 : rounded >= 255 ? 255 : rounded;
            }
        }
        return out;
    }

    const bufB = scratch?.bufB?.length >= len ? scratch.bufB : new Float32Array(len);
    const colAcc = scratch?.colAcc?.length >= w ? scratch.colAcc : new Float32Array(w);
    for (let i = 0; i < len; i += 1) bufA[i] = src[i];
    const sizes = boxesForGauss(sigma, 3);
    for (const size of sizes) {
        const radius = (size - 1) >> 1;
        boxBlurHorizontalFloat(bufA, bufB, w, h, radius);
        boxBlurVerticalFloat(bufB, bufA, w, h, radius, colAcc);
    }
    for (let i = 0; i < len; i += 1) {
        const rounded = Math.round(bufA[i]);
        out[i] = rounded <= 0 ? 0 : rounded >= 255 ? 255 : rounded;
    }
    return out;
}

export function postprocessMaskAlpha(alpha, width, height, { growPx = 0, featherPx = 0, distField = null, out = null } = {}) {
    const len = width * height;
    const dst = out && out.length >= len ? out : new Uint8Array(len);
    if (!alpha || len <= 0) return dst;
    const grow = Math.max(0, Number(growPx) || 0);
    const feather = Math.max(0, Number(featherPx) || 0);
    if (!(grow > 0) && !(feather > 0.05)) {
        dst.set(alpha);
        return dst;
    }
    let grown = alpha;
    if (grow > 0) {
        const field = distField || computeSquaredDistanceField(alpha, width, height);
        grown = applyGrowFromDistanceField(alpha, field, grow, feather > 0.05 ? new Uint8Array(len) : dst);
    }
    if (feather > 0.05) {
        applyFeatherBlur(grown, width, height, feather, dst);
    }
    return dst;
}

export function effectiveCropExtent(meta, grow, feather, source) {
    const serverSettings = meta?.mask_postprocess;
    const g = Math.max(0, Math.round(Number(grow) || 0));
    const f = Math.max(0, Math.round(Number(feather) || 0));
    if (
        meta?.crop_extent?.frames?.length &&
        Array.isArray(meta?.crop_extent?.box) &&
        serverSettings &&
        Math.round(Number(serverSettings.grow)) === g &&
        Math.round(Number(serverSettings.feather)) === f
    ) {
        return meta.crop_extent;
    }
    const raw = meta?.raw_crop_extent;
    const width = Math.max(0, Math.round(Number(source?.width) || 0));
    const height = Math.max(0, Math.round(Number(source?.height) || 0));
    if (raw?.frames?.length && Array.isArray(raw.box) && raw.box.length === 4 && width > 0 && height > 0) {
        const pad = g + f;
        const [rx0, ry0, rx1, ry1] = raw.box.map((val) => Math.round(Number(val) || 0));
        return {
            frames: raw.frames,
            box: [
                Math.max(0, rx0 - pad),
                Math.max(0, ry0 - pad),
                Math.min(width, rx1 + pad),
                Math.min(height, ry1 + pad),
            ],
        };
    }
    return meta?.crop_extent || null;
}

export function createMaskPostprocessor() {
    let rawAlpha = null;
    let grownAlpha = null;
    let finalAlpha = null;
    let blurScratch = null;
    let blurScratch2 = null;
    let colAcc = null;
    let cachedWidth = 0;
    let cachedHeight = 0;
    let cachedLen = 0;
    let cachedRawKey = "";
    let cachedGrowKey = "";
    let cachedFinalKey = "";
    let distField = null;

    function ensureBuffers(width, height) {
        const len = width * height;
        if (len !== cachedLen) {
            rawAlpha = new Uint8Array(len);
            grownAlpha = new Uint8Array(len);
            finalAlpha = new Uint8Array(len);
            blurScratch = new Float32Array(len);
            blurScratch2 = new Float32Array(len);
            cachedLen = len;
        }
        if (!colAcc || colAcc.length < width) {
            colAcc = new Float32Array(width);
        }
    }

    function invalidate() {
        cachedRawKey = "";
        cachedGrowKey = "";
        cachedFinalKey = "";
        distField = null;
    }

    function processImageData(
        imageData,
        { rawKey = "", growPx = 0, featherPx = 0, maskOnly = false, alpha = 0.38, applyPostprocess = true } = {},
    ) {
        const pixels = imageData?.data;
        const width = Math.max(0, Math.round(Number(imageData?.width) || 0));
        const height = Math.max(0, Math.round(Number(imageData?.height) || 0));
        const len = width * height;
        if (!pixels || len <= 0 || pixels.length < len * 4) return false;

        ensureBuffers(width, height);
        const key = String(rawKey || "");
        if (!key || key !== cachedRawKey || width !== cachedWidth || height !== cachedHeight) {
            for (let i = 0, p = 0; i < len; i += 1, p += 4) {
                const r = pixels[p];
                const g = pixels[p + 1];
                const b = pixels[p + 2];
                const a = pixels[p + 3];
                if (a === 0) {
                    rawAlpha[i] = 0;
                    continue;
                }
                const lum = r === g && g === b ? r : 0.2126 * r + 0.7152 * g + 0.0722 * b;
                rawAlpha[i] = a === 255 ? Math.round(lum) : Math.round((lum * a) / 255);
            }
            cachedRawKey = key;
            cachedWidth = width;
            cachedHeight = height;
            cachedGrowKey = "";
            cachedFinalKey = "";
            distField = null;
        }

        const gPx = Math.max(0, Number(growPx) || 0);
        const fPx = Math.max(0, Number(featherPx) || 0);
        let outputAlpha = rawAlpha;

        if (applyPostprocess && (gPx > 0 || fPx > 0.05)) {
            const gKey = key ? `${key}|g:${gPx.toFixed(2)}` : "";
            if (!gKey || gKey !== cachedGrowKey) {
                if (gPx > 0) {
                    if (!distField) distField = computeSquaredDistanceField(rawAlpha, width, height);
                    applyGrowFromDistanceField(rawAlpha, distField, gPx, grownAlpha);
                } else {
                    grownAlpha.set(rawAlpha);
                }
                cachedGrowKey = gKey;
                cachedFinalKey = "";
            }

            const fKey = gKey ? `${gKey}|f:${fPx.toFixed(2)}` : "";
            if (!fKey || fKey !== cachedFinalKey) {
                if (fPx > 0.05) {
                    applyFeatherBlur(grownAlpha, width, height, fPx, finalAlpha, {
                        bufA: blurScratch,
                        bufB: blurScratch2,
                        colAcc,
                    });
                } else {
                    finalAlpha.set(grownAlpha);
                }
                cachedFinalKey = fKey;
            }
            outputAlpha = finalAlpha;
        }

        if (maskOnly) {
            for (let i = 0, p = 0; i < len; i += 1, p += 4) {
                const v = outputAlpha[i];
                pixels[p] = v;
                pixels[p + 1] = v;
                pixels[p + 2] = v;
                pixels[p + 3] = 255;
            }
        } else {
            const overlayAlpha = Math.max(0, Math.min(1, Number(alpha) || 0));
            for (let i = 0, p = 0; i < len; i += 1, p += 4) {
                pixels[p] = 79;
                pixels[p + 1] = 227;
                pixels[p + 2] = 139;
                pixels[p + 3] = Math.round(outputAlpha[i] * overlayAlpha);
            }
        }
        return true;
    }

    return {
        invalidate,
        processImageData,
    };
}
