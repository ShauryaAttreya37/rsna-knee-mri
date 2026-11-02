// Renders the page's charts from window.FIGURE_DATA as theme-aware inline SVG.
// Labels go through textContent only; no chart builds markup from strings.
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const AUC_DOMAIN = [0.5, 1];
  const AUC_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const COVERAGE_TICKS = [0, 0.25, 0.5, 0.75, 1];
  const NARROW_WIDTH = 520;
  const MINUS = "−";

  const formatAuc = (value) => value.toFixed(3);
  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
  const formatDelta = (value) =>
    `${value < 0 ? MINUS : "+"}${Math.abs(value).toFixed(3)}`;
  const scale = (value, [d0, d1], [r0, r1]) =>
    r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

  function svgEl(tag, attrs, parent, text) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function htmlEl(tag, className, parent, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  const tooltip = htmlEl("div", "viz-tooltip", document.body);
  // Each focusable mark carries a full aria-label, so the tooltip is visual only.
  tooltip.setAttribute("aria-hidden", "true");
  tooltip.hidden = true;
  let deactivateCurrent = null;

  function placeTooltip(x, y) {
    const margin = 12;
    const { width, height } = tooltip.getBoundingClientRect();
    const left = Math.min(x + margin, window.innerWidth - width - margin);
    const fitsBelow = y + margin + height < window.innerHeight;
    tooltip.style.left = `${Math.max(margin, left)}px`;
    tooltip.style.top = `${fitsBelow ? y + margin : y - height - margin}px`;
  }

  function showTooltip(detail, x, y) {
    tooltip.replaceChildren();
    htmlEl("div", "viz-tooltip-title", tooltip, detail.title);
    detail.rows.forEach((row) => {
      const line = htmlEl("div", "viz-tooltip-row", tooltip);
      const key = htmlEl("span", `viz-key viz-key-${row.key || "none"}`, line);
      key.setAttribute("aria-hidden", "true");
      htmlEl("strong", "", line, row.value);
      htmlEl("span", "viz-tooltip-name", line, row.name);
    });
    tooltip.hidden = false;
    placeTooltip(x, y);
  }

  const hideTooltip = () => {
    tooltip.hidden = true;
  };

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && deactivateCurrent) deactivateCurrent();
  });

  document.addEventListener("pointerdown", (event) => {
    if (deactivateCurrent && !event.target.closest(".viz-hit")) deactivateCurrent();
  });

  function bindHit(hit, detail, onActive) {
    hit.setAttribute("tabindex", "0");
    hit.setAttribute("role", "img");
    hit.setAttribute(
      "aria-label",
      `${detail.title}: ${detail.rows.map((r) => `${r.name} ${r.value}`).join(", ")}`
    );
    const deactivate = () => {
      if (onActive) onActive(false);
      hideTooltip();
      deactivateCurrent = null;
    };
    const activate = (x, y) => {
      if (deactivateCurrent && deactivateCurrent !== deactivate) deactivateCurrent();
      deactivateCurrent = deactivate;
      if (onActive) onActive(true);
      showTooltip(detail, x, y);
    };
    hit.addEventListener("pointermove", (e) => activate(e.clientX, e.clientY));
    // A finger lifting fires pointerleave; keep the tooltip until a tap elsewhere.
    hit.addEventListener("pointerleave", (e) => {
      if (e.pointerType !== "touch") deactivate();
    });
    hit.addEventListener("blur", deactivate);
    hit.addEventListener("focus", () => {
      const box = hit.getBoundingClientRect();
      activate(box.left + box.width / 2, box.top + box.height / 2);
    });
  }

  function renderLegend(mount, entries) {
    const legend = htmlEl("div", "viz-legend", mount);
    entries.forEach(({ key, label }) => {
      const item = htmlEl("span", "viz-legend-item", legend);
      const swatch = htmlEl("span", `viz-key viz-key-${key}`, item);
      swatch.setAttribute("aria-hidden", "true");
      htmlEl("span", "", item, label);
    });
  }

  function newSvg(mount, width, height, label) {
    return svgEl(
      "svg",
      { width, height, viewBox: `0 0 ${width} ${height}`, role: "group", "aria-label": label },
      mount
    );
  }

  function drawAucAxis(svg, xOf, top, bottom) {
    AUC_TICKS.forEach((tick) => {
      const x = xOf(tick);
      svgEl("line", { x1: x, x2: x, y1: top, y2: bottom, class: "viz-grid" }, svg);
      svgEl(
        "text",
        { x, y: bottom + 18, class: "viz-tick", "text-anchor": "middle" },
        svg,
        tick.toFixed(1)
      );
    });
  }

  function selectLabelledRows(rows, labelAll) {
    if (labelAll) return new Set(rows);
    const gains = rows.filter((row) => row.after >= row.before);
    const topGain = gains.reduce(
      (best, row) => (row.after - row.before > best.after - best.before ? row : best),
      gains[0]
    );
    return new Set([topGain, ...rows.filter((row) => row.after < row.before)]);
  }

  // Narrow screens stack each row's text above its mark, so the plot keeps full width.
  function rowLayout(width) {
    return width < NARROW_WIDTH
      ? { stacked: true, plotLeft: 12, plotRightMargin: 14, rowHeight: 46, textY: 14, markY: 32 }
      : { stacked: false, plotLeft: 158, plotRightMargin: 80, rowHeight: 30, textY: 19, markY: 15 };
  }

  function rowTextAttrs(layout, rowTop, width, slot) {
    const y = rowTop + layout.textY;
    if (slot === "label") {
      return layout.stacked
        ? { x: 0, y, class: "viz-label" }
        : { x: layout.plotLeft - 12, y, class: "viz-label", "text-anchor": "end" };
    }
    return layout.stacked
      ? { x: width, y, class: "viz-value", "text-anchor": "end" }
      : { x: width - layout.plotRightMargin + 10, y, class: "viz-value" };
  }

  function drawDumbbellRow(svg, row, geometry, config, showDelta) {
    const { xOf, width, layout } = geometry;
    const y = geometry.y + layout.markY;
    const delta = row.after - row.before;
    const declined = delta < 0;
    const group = svgEl("g", {}, svg);
    const hit = svgEl(
      "rect",
      { x: 0, y: geometry.y, width, height: layout.rowHeight, class: "viz-hit" },
      group
    );
    svgEl("text", rowTextAttrs(layout, geometry.y, width, "label"), group, row.label);
    svgEl(
      "line",
      {
        x1: xOf(row.before),
        x2: xOf(row.after),
        y1: y,
        y2: y,
        class: declined ? "viz-link viz-link-drop" : "viz-link",
      },
      group
    );
    svgEl("circle", { cx: xOf(row.before), cy: y, r: 4.5, class: "viz-dot-before" }, group);
    svgEl("circle", { cx: xOf(row.after), cy: y, r: 5, class: "viz-dot-after" }, group);
    if (showDelta) {
      svgEl(
        "text",
        rowTextAttrs(layout, geometry.y, width, "value"),
        group,
        `${declined ? "▼ " : ""}${formatDelta(delta)}`
      );
    }
    bindHit(hit, {
      title: row.studies ? `${row.label} (${row.studies} studies)` : row.label,
      rows: [
        { key: "after", value: formatAuc(row.after), name: config.afterLabel },
        { key: "before", value: formatAuc(row.before), name: config.beforeLabel },
        { key: declined ? "drop" : "link", value: formatDelta(delta), name: "Change" },
      ],
    });
  }

  function renderDumbbell(mount, config) {
    const width = mount.clientWidth;
    const layout = rowLayout(width);
    const top = 6;
    const plotBottom = top + config.rows.length * layout.rowHeight;
    const hasDrop = config.rows.some((row) => row.after < row.before);
    renderLegend(mount, [
      { key: "before", label: config.beforeLabel },
      { key: "after", label: config.afterLabel },
      ...(hasDrop ? [{ key: "drop", label: "AUC dropped" }] : []),
    ]);
    const svg = newSvg(mount, width, plotBottom + 26, config.summary);
    const xOf = (value) =>
      scale(value, AUC_DOMAIN, [layout.plotLeft, width - layout.plotRightMargin]);
    drawAucAxis(svg, xOf, top, plotBottom);
    const labelled = selectLabelledRows(config.rows, config.labelAll);
    config.rows.forEach((row, index) => {
      const geometry = { xOf, width, layout, y: top + index * layout.rowHeight };
      drawDumbbellRow(svg, row, geometry, config, labelled.has(row));
    });
  }

  function barPath(x0, y, length, thickness, radius) {
    const end = x0 + length;
    const r = Math.min(radius, length);
    return (
      `M${x0},${y}H${end - r}Q${end},${y} ${end},${y + r}` +
      `V${y + thickness - r}Q${end},${y + thickness} ${end - r},${y + thickness}` +
      `H${x0}Z`
    );
  }

  function renderBars(mount, config) {
    const width = mount.clientWidth;
    const layout = rowLayout(width);
    const baseline = layout.stacked ? 0 : layout.plotLeft;
    const thickness = layout.stacked ? 14 : 16;
    const barOffset = layout.stacked ? 22 : (layout.rowHeight - thickness) / 2;
    const height = config.rows.length * layout.rowHeight + 4;
    const svg = newSvg(mount, width, height, config.summary);
    const plotWidth = width - baseline - (layout.stacked ? 0 : layout.plotRightMargin);
    const lengthOf = (row) => (row.positives / row.total) * plotWidth;
    svgEl("line", { x1: baseline, x2: baseline, y1: 0, y2: height, class: "viz-axis" }, svg);
    config.rows.forEach((row, index) => {
      const y = index * layout.rowHeight;
      const hit = svgEl("rect", { x: 0, y, width, height: layout.rowHeight, class: "viz-hit" }, svg);
      const labelAttrs = rowTextAttrs(layout, y, width, "label");
      svgEl("text", layout.stacked ? { ...labelAttrs, x: 6 } : labelAttrs, svg, row.label);
      const length = lengthOf(row);
      svgEl("path", { d: barPath(baseline, y + barOffset, length, thickness, 4), class: "viz-bar" }, svg);
      const valueAttrs = layout.stacked
        ? rowTextAttrs(layout, y, width, "value")
        : { x: baseline + length + 8, y: y + layout.textY, class: "viz-value" };
      svgEl("text", valueAttrs, svg, `${row.positives} of ${row.total}`);
      bindHit(hit, {
        title: row.label,
        rows: [
          { key: "after", value: `${row.positives} of ${row.total}`, name: "Positive gold studies" },
          { value: formatPercent(row.positives / row.total), name: "Share of gold studies" },
        ],
      });
    });
  }

  function linePath(rows, field, xOf, yOf) {
    return rows
      .map((row, i) => `${i ? "L" : "M"}${xOf(row.crop_mm)},${yOf(row[field])}`)
      .join("");
  }

  function drawCoverageAxes(svg, geometry, data) {
    const { xOf, yOf, left, right, bottom } = geometry;
    COVERAGE_TICKS.forEach((tick) => {
      const y = yOf(tick);
      svgEl("line", { x1: left, x2: right, y1: y, y2: y, class: "viz-grid" }, svg);
      svgEl(
        "text",
        { x: left - 8, y: y + 4, class: "viz-tick", "text-anchor": "end" },
        svg,
        `${tick * 100}%`
      );
    });
    data.rows.forEach((row) => {
      svgEl(
        "text",
        { x: xOf(row.crop_mm), y: bottom + 22, class: "viz-tick", "text-anchor": "middle" },
        svg,
        row.crop_mm
      );
    });
    svgEl(
      "text",
      { x: (left + right) / 2, y: bottom + 42, class: "viz-tick", "text-anchor": "middle" },
      svg,
      "Square crop size (mm)"
    );
    const limitY = yOf(data.minimum_coverage);
    svgEl("line", { x1: left, x2: right, y1: limitY, y2: limitY, class: "viz-limit" }, svg);
    svgEl(
      "text",
      { x: right, y: limitY - 6, class: "viz-note", "text-anchor": "end" },
      svg,
      `${data.minimum_coverage * 100}% minimum`
    );
  }

  function drawSelectedCrop(svg, selected, geometry) {
    const { xOf, yOf, bottom } = geometry;
    const x = xOf(selected.crop_mm);
    svgEl("line", { x1: x, x2: x, y1: yOf(selected.worst_plane), y2: bottom, class: "viz-axis" }, svg);
    const labelY = yOf(0.25);
    svgEl("text", { x: x + 8, y: labelY, class: "viz-value" }, svg, `${selected.crop_mm} mm selected`);
    svgEl(
      "text",
      { x: x + 8, y: labelY + 17, class: "viz-note" },
      svg,
      `${formatPercent(selected.overall)} overall`
    );
  }

  function drawCoverageColumns(svg, data, geometry) {
    const { xOf, yOf, top, bottom, left, right } = geometry;
    const step = (right - left) / (data.rows.length - 1);
    data.rows.forEach((row) => {
      const x = xOf(row.crop_mm);
      const hit = svgEl(
        "rect",
        { x: x - step / 2, y: top, width: step, height: bottom - top, class: "viz-hit" },
        svg
      );
      const crosshair = svgEl(
        "line",
        { x1: x, x2: x, y1: top, y2: bottom, class: "viz-crosshair", visibility: "hidden" },
        svg
      );
      bindHit(
        hit,
        {
          title: `${row.crop_mm} mm crop`,
          rows: [
            { key: "line-after", value: formatPercent(row.overall), name: "All series" },
            { key: "line-before", value: formatPercent(row.worst_plane), name: "Worst plane" },
            { value: formatPercent(row.axial), name: "Axial" },
            { value: formatPercent(row.coronal), name: "Coronal" },
            { value: formatPercent(row.sagittal), name: "Sagittal" },
          ],
        },
        (active) => crosshair.setAttribute("visibility", active ? "visible" : "hidden")
      );
      [
        ["worst_plane", "viz-point-before"],
        ["overall", "viz-point-after"],
      ].forEach(([field, className]) => {
        svgEl("circle", { cx: x, cy: yOf(row[field]), r: 4, class: className }, svg);
      });
    });
  }

  function renderCoverage(mount, data, summary) {
    const width = mount.clientWidth;
    const left = 48;
    const right = width - 20;
    const top = 16;
    const bottom = 256;
    renderLegend(mount, [
      { key: "line-after", label: "All series" },
      { key: "line-before", label: "Worst anatomical plane" },
    ]);
    const svg = newSvg(mount, width, bottom + 50, summary);
    const crops = data.rows.map((row) => row.crop_mm);
    const xOf = (mm) => scale(mm, [crops[0], crops[crops.length - 1]], [left, right]);
    const yOf = (share) => scale(share, [0, 1], [bottom, top]);
    const geometry = { xOf, yOf, left, right, top, bottom };
    drawCoverageAxes(svg, geometry, data);
    drawSelectedCrop(
      svg,
      data.rows.find((row) => row.crop_mm === data.selected_crop_mm),
      geometry
    );
    svgEl("path", { d: linePath(data.rows, "worst_plane", xOf, yOf), class: "viz-line-before" }, svg);
    svgEl("path", { d: linePath(data.rows, "overall", xOf, yOf), class: "viz-line-after" }, svg);
    drawCoverageColumns(svg, data, geometry);
  }

  function renderTable(details, headers, rows) {
    htmlEl("summary", "", details, "Table view");
    const wrap = htmlEl("div", "tablewrap", details);
    const table = htmlEl("table", "", wrap);
    const headRow = htmlEl("tr", "", htmlEl("thead", "", table));
    headers.forEach((header, index) => {
      htmlEl("th", index ? "num" : "", headRow, header);
    });
    const body = htmlEl("tbody", "", table);
    rows.forEach((cells) => {
      const row = htmlEl("tr", "", body);
      cells.forEach((cell, index) => htmlEl("td", index ? "num" : "", row, cell));
    });
  }

  const pairedCells = (row) => [
    row.label,
    formatAuc(row.before),
    formatAuc(row.after),
    formatDelta(row.after - row.before),
  ];

  function fillTables(data) {
    const tables = {
      prevalence: [
        ["Finding", "Positive", "Share"],
        data.prevalence.map((r) => [r.label, `${r.positives} of ${r.total}`, formatPercent(r.positives / r.total)]),
      ],
      crop_coverage: [
        ["Crop (mm)", "All series", "Worst plane", "Axial", "Coronal", "Sagittal"],
        data.crop_coverage.rows.map((r) => [
          `${r.crop_mm}`,
          ...[r.overall, r.worst_plane, r.axial, r.coronal, r.sagittal].map(formatPercent),
        ]),
      ],
      label_quality: [
        ["Finding", "Qwen", "Ensemble", "Change"],
        [...data.label_quality.rows.map(pairedCells), pairedCells({ label: "Macro AUC", ...data.label_quality.macro })],
      ],
      fold_auc: [
        ["Fold", "v1", "v2", "Change"],
        data.fold_auc.map(pairedCells),
      ],
    };
    document.querySelectorAll("details[data-table]").forEach((details) => {
      const [headers, rows] = tables[details.dataset.table];
      renderTable(details, headers, rows);
    });
  }

  function chartRenderers(data) {
    return {
      prevalence: (mount) =>
        renderBars(mount, {
          rows: data.prevalence,
          summary: "Bar chart of positive gold studies for each of twelve findings.",
        }),
      crop_coverage: (mount) =>
        renderCoverage(
          mount,
          data.crop_coverage,
          "Line chart of series covered without padding at each candidate crop size."
        ),
      label_quality: (mount) =>
        renderDumbbell(mount, {
          rows: data.label_quality.rows,
          beforeLabel: "Qwen labels",
          afterLabel: "Label ensemble",
          summary: "Dumbbell chart of label AUC against gold, Qwen versus ensemble, per finding.",
        }),
      finding_auc: (mount) =>
        renderDumbbell(mount, {
          rows: data.finding_auc.rows,
          beforeLabel: "v1: Qwen labels",
          afterLabel: "v2: label ensemble",
          summary: "Dumbbell chart of out-of-fold model AUC, v1 versus v2, per finding.",
        }),
      fold_auc: (mount) =>
        renderDumbbell(mount, {
          rows: data.fold_auc,
          beforeLabel: "v1: Qwen labels",
          afterLabel: "v2: label ensemble",
          labelAll: true,
          summary: "Dumbbell chart of macro AUC, v1 versus v2, for each of five folds.",
        }),
    };
  }

  function start() {
    const data = window.FIGURE_DATA;
    if (!data) throw new Error("figure-data.js did not load");
    fillTables(data);
    const renderers = chartRenderers(data);
    const widths = new WeakMap();
    const observer = new ResizeObserver((entries) => {
      entries.forEach(({ target }) => {
        if (widths.get(target) === target.clientWidth) return;
        widths.set(target, target.clientWidth);
        hideTooltip();
        target.replaceChildren();
        renderers[target.dataset.chart](target);
      });
    });
    document.querySelectorAll("[data-chart]").forEach((mount) => observer.observe(mount));
  }

  start();
})();
