(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const COLORS = {
    ucf: '#7557f6',
    dff: '#20ad82',
    effort: '#f4a340',
    clip: '#2d9cff',
    dino: '#e85d6a',
    spsl: '#8a98aa',
    recce: '#a06ee1',
    xception: '#7f8a9a'
  };

  const datasets = {
    inDomain: {
      labels: ['CDF', 'DFFD', 'DFDCP', 'FF++', 'DF40', 'MFFI'],
      min: 0,
      series: [
        { name: 'UCF-Net', color: COLORS.ucf, values: [99.74, 99.99, 97.88, 90.55, 92.90, 90.92], mean: 95.33 },
        { name: 'DFF-Adapter', color: COLORS.dff, values: [99.85, 99.96, 97.31, 92.10, 91.60, 88.31], mean: 94.86 },
        { name: 'Effort', color: COLORS.effort, values: [99.52, 99.95, 96.51, 85.14, 92.56, 85.60], mean: 93.21 },
        { name: 'SPSL', color: COLORS.spsl, values: [98.97, 98.87, 93.49, 82.32, 81.92, 81.43], mean: 89.50 },
        { name: 'RECCE', color: COLORS.recce, values: [99.38, 99.69, 89.04, 78.70, 88.60, 75.29], mean: 88.45 },
        { name: 'Xception', color: COLORS.xception, values: [99.26, 98.48, 90.93, 76.03, 83.89, 78.94], mean: 87.92 },
        { name: 'CLIP', color: COLORS.clip, values: [94.83, 99.39, 86.78, 74.61, 87.45, 82.60], mean: 87.61 },
        { name: 'DINOv2', color: COLORS.dino, values: [80.28, 98.11, 83.50, 66.08, 83.75, 71.39], mean: 80.52 }
      ],
      ranking: [
        ['UCF-Net', 95.33, COLORS.ucf], ['DFF-Adapter', 94.86, COLORS.dff], ['Effort', 93.21, COLORS.effort],
        ['SPSL', 89.50, COLORS.spsl], ['RECCE', 88.45, COLORS.recce], ['Xception', 87.92, COLORS.xception],
        ['CLIP', 87.61, COLORS.clip], ['DINOv2', 80.52, COLORS.dino]
      ]
    },
    crossDomain: {
      labels: ['UADFV', 'DFF', 'DFDC', 'DF40-FS', 'DF40-FR', 'DF40-EFS', 'DF40-FE'],
      min: 0,
      series: [
        { name: 'UCF-Net', color: COLORS.ucf, values: [97.49, 89.72, 90.01, 97.85, 93.38, 78.10, 98.52], mean: 92.15 },
        { name: 'DFF-Adapter', color: COLORS.dff, values: [99.08, 81.97, 87.63, 92.45, 90.11, 77.78, 95.37], mean: 89.20 },
        { name: 'Effort', color: COLORS.effort, values: [98.24, 88.91, 88.61, 95.51, 88.39, 69.04, 93.67], mean: 88.91 },
        { name: 'CLIP', color: COLORS.clip, values: [95.46, 85.52, 77.06, 95.21, 89.04, 74.65, 93.62], mean: 87.22 },
        { name: 'DINOv2', color: COLORS.dino, values: [85.67, 58.12, 72.71, 94.35, 74.62, 61.51, 95.04], mean: 77.43 },
        { name: 'SPSL', color: COLORS.spsl, values: [94.29, 56.72, 73.02, 60.65, 74.26, 61.89, 49.72], mean: 67.22 },
        { name: 'Xception', color: COLORS.xception, values: [91.37, 53.14, 75.16, 66.00, 89.90, 45.96, 46.91], mean: 66.92 },
        { name: 'RECCE', color: COLORS.recce, values: [85.59, 72.62, 73.37, 76.27, 64.50, 42.89, 48.04], mean: 66.18 }
      ],
      branches: [
        { name: 'UCF-Net', color: COLORS.ucf, values: [97.49, 89.72, 90.01, 97.85, 93.38, 78.10, 98.52], mean: 92.15 },
        { name: 'CLIP', color: COLORS.clip, values: [95.46, 85.52, 77.06, 95.21, 89.04, 74.65, 93.62], mean: 87.22 },
        { name: 'DINOv2', color: COLORS.dino, values: [85.67, 58.12, 72.71, 94.35, 74.62, 61.51, 95.04], mean: 77.43 }
      ],
      ranking: [
        ['UCF-Net', 92.15, COLORS.ucf], ['DFF-Adapter', 89.20, COLORS.dff], ['Effort', 88.91, COLORS.effort],
        ['CLIP', 87.22, COLORS.clip], ['DINOv2', 77.43, COLORS.dino], ['SPSL', 67.22, COLORS.spsl],
        ['Xception', 66.92, COLORS.xception], ['RECCE', 66.18, COLORS.recce]
      ]
    },
    crossGenerator: {
      labels: ['0-shot', '5-shot', '50-shot', '100-shot'],
      series: [
        { name: 'UCF-Net', color: COLORS.ucf, values: [40.90, 91.36, 98.24, 98.81] },
        { name: 'DFF-Adapter', color: COLORS.dff, values: [43.21, 89.23, 97.07, 97.27] },
        { name: 'Effort', color: COLORS.effort, values: [37.23, 83.03, 97.10, 98.49] },
        { name: 'SPSL', color: COLORS.spsl, values: [60.02, 82.72, 90.96, 96.11] },
        { name: 'Xception', color: COLORS.xception, values: [46.72, 77.41, 91.85, 95.66] },
        { name: 'RECCE', color: COLORS.recce, values: [32.80, 77.06, 94.08, 91.45] }
      ]
    },
    trainingScale: {
      labels: ['10K', '1M', '2M'],
      series: [
        { name: 'UCF-Net', color: COLORS.ucf, values: [85.58, 91.46, 92.15] },
        { name: 'DFF-Adapter', color: COLORS.dff, values: [85.12, 89.70, 89.20] },
        { name: 'Effort', color: COLORS.effort, values: [83.37, 88.79, 88.91] },
        { name: 'SPSL', color: COLORS.spsl, values: [65.88, 70.34, 67.22] },
        { name: 'Xception', color: COLORS.xception, values: [69.81, 68.14, 66.92] },
        { name: 'RECCE', color: COLORS.recce, values: [63.18, 63.01, 66.18] }
      ]
    }
  };

  const miniData = {
    encoders: [
      ['CLIP', 88.86, COLORS.clip], ['DINO', 87.41, COLORS.dino], ['CLIP + CLIP', 87.97, '#72b9ef'],
      ['DINO + DINO', 86.94, '#e79b7b'], ['CLIP + DINO', 92.15, COLORS.ucf]
    ],
    lea: [['Without LEA', 91.85, '#99a5b5'], ['With LEA', 92.15, COLORS.ucf]],
    fusion: [['Concat', 90.02, '#8b98aa'], ['Sum (average)', 90.12, '#7fa3bb'], ['Cross-attn.', 89.19, '#c58c79'], ['UAF', 92.15, COLORS.ucf]],
    lora: [['Frozen', 89.25, '#9ba6b5'], ['Rank 1', 91.75, '#6ea0ca'], ['Rank 4', 92.15, COLORS.ucf], ['Rank 16', 90.47, '#a678be'], ['Rank 64', 91.15, '#c17f99']]
  };

  function initTheme() {
    // The project page intentionally uses one consistent soft academic palette.
    // This avoids abrupt black/white section switching and keeps figures readable.
    document.documentElement.dataset.theme = 'light';
  }

  function initNavigation() {
    const header = $('.site-header');
    const toggle = $('.nav-toggle');
    const nav = $('.primary-nav');
    const links = $$('.primary-nav a');

    const closeNav = () => {
      nav?.classList.remove('open');
      toggle?.setAttribute('aria-expanded', 'false');
    };

    if (toggle && nav) toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
    links.forEach(link => link.addEventListener('click', closeNav));
    window.addEventListener('scroll', () => header?.classList.toggle('scrolled', window.scrollY > 16), { passive: true });

    const sectionLinks = links.map(link => {
      const href = link.getAttribute('href');
      if (!href?.startsWith('#') || href.length < 2) return null;
      let id;
      try {
        id = decodeURIComponent(href.slice(1));
      } catch {
        return null;
      }
      const section = document.getElementById(id);
      return section ? { link, section } : null;
    }).filter(Boolean);

    if (!sectionLinks.length || typeof IntersectionObserver !== 'function') return;
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        links.forEach(link => link.classList.remove('active'));
        sectionLinks.find(item => item.section === entry.target)?.link.classList.add('active');
      });
    }, { rootMargin: '-35% 0px -55%', threshold: 0 });
    sectionLinks.forEach(({ section }) => observer.observe(section));
  }

  function initReveal() {
    const revealItems = $$('.reveal');
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      revealItems.forEach(item => item.classList.add('visible'));
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: .12 });
    revealItems.forEach(item => observer.observe(item));
  }

  function polarPoint(cx, cy, radius, angle) {
    return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius];
  }

  const SERIES_DASHES = {
    'DFF-Adapter': '7 4',
    Effort: '5 4',
    SPSL: '3 4',
    RECCE: '2 4',
    Xception: '9 4',
    CLIP: '6 3',
    DINOv2: '4 3'
  };

  const CHART_TARGETS = {
    'in-domain': '#radar-in-domain',
    'cross-domain': '#radar-cross-domain',
    'cross-generator': '#line-cross-generator',
    'training-scale': '#line-training-scale'
  };

  function appendChartAccessibility(svg, target, titleText, descriptionText, series) {
    const svgNS = 'http://www.w3.org/2000/svg';
    const idBase = target.id || `chart-${Math.random().toString(36).slice(2)}`;
    const titleId = `${idBase}-title`;
    const descriptionId = `${idBase}-description`;
    const title = document.createElementNS(svgNS, 'title');
    const description = document.createElementNS(svgNS, 'desc');
    title.id = titleId;
    description.id = descriptionId;
    title.textContent = titleText;
    description.textContent = `${descriptionText} Visible methods: ${series.map(item => item.name).join(', ')}.`;
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-labelledby', `${titleId} ${descriptionId}`);
    svg.append(title, description);
  }

  function setPointAccessibility(point, method, dataset, value) {
    const label = `${method}, ${dataset}, AUC ${value.toFixed(2)} percent`;
    point.setAttribute('tabindex', '0');
    point.setAttribute('role', 'img');
    point.setAttribute('aria-label', label);
    point.dataset.tooltip = label;
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = label;
    point.appendChild(title);
  }

  function attachChartTooltip(target, svg) {
    const tooltip = document.createElement('div');
    tooltip.className = 'chart-tooltip';
    tooltip.setAttribute('role', 'status');
    tooltip.hidden = true;

    const show = point => {
      const targetRect = target.getBoundingClientRect();
      const pointRect = point.getBoundingClientRect();
      tooltip.textContent = point.dataset.tooltip || '';
      tooltip.hidden = false;
      const preferredLeft = pointRect.left - targetRect.left + pointRect.width / 2;
      const preferredTop = pointRect.top - targetRect.top - 12;
      tooltip.style.left = `${Math.max(18, Math.min(targetRect.width - 18, preferredLeft))}px`;
      tooltip.style.top = `${Math.max(8, preferredTop)}px`;
    };
    const hide = () => { tooltip.hidden = true; };

    $$('[data-tooltip]', svg).forEach(point => {
      point.addEventListener('mouseenter', () => show(point));
      point.addEventListener('mouseleave', hide);
      point.addEventListener('focus', () => show(point));
      point.addEventListener('blur', hide);
    });
    target.appendChild(tooltip);
  }

  function focusChartSeries(key, name = '') {
    const target = $(CHART_TARGETS[key]);
    const legend = $(`[data-legend="${key}"]`);
    target?.querySelectorAll('[data-series]').forEach(node => {
      const focused = Boolean(name) && node.dataset.series === name;
      node.classList.toggle('is-focused', focused);
      node.classList.toggle('is-muted', Boolean(name) && !focused);
    });
    legend?.querySelectorAll('[data-series]').forEach(button => {
      button.classList.toggle('is-focused', Boolean(name) && button.dataset.series === name);
      button.classList.toggle('is-muted', Boolean(name) && button.dataset.series !== name);
    });
  }

  function orderedSeries(series) {
    return series.slice().sort((a, b) => Number(a.name === 'UCF-Net') - Number(b.name === 'UCF-Net'));
  }

  function createRadarChart(target, labels, series, minValue, accessibility) {
    if (!target) return;
    const size = 420;
    const cx = size / 2;
    const cy = size / 2;
    const radius = 142;
    const count = labels.length;
    const start = -Math.PI / 2;
    const maxValue = 100;
    const rings = 4;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    appendChartAccessibility(svg, target, accessibility.title, accessibility.description, series);

    for (let ring = 1; ring <= rings; ring++) {
      const rr = radius * ring / rings;
      const points = labels.map((_, i) => polarPoint(cx, cy, rr, start + (Math.PI * 2 * i / count)).join(',')).join(' ');
      const polygon = document.createElementNS(svgNS, 'polygon');
      polygon.setAttribute('points', points);
      polygon.setAttribute('class', 'radar-grid');
      svg.appendChild(polygon);
    }

    labels.forEach((label, i) => {
      const angle = start + Math.PI * 2 * i / count;
      const [x, y] = polarPoint(cx, cy, radius, angle);
      const line = document.createElementNS(svgNS, 'line');
      line.setAttribute('x1', cx); line.setAttribute('y1', cy); line.setAttribute('x2', x); line.setAttribute('y2', y); line.setAttribute('class', 'radar-axis');
      svg.appendChild(line);

      const [lx, ly] = polarPoint(cx, cy, radius + 28, angle);
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', lx); text.setAttribute('y', ly);
      text.setAttribute('text-anchor', Math.abs(Math.cos(angle)) < .15 ? 'middle' : Math.cos(angle) > 0 ? 'start' : 'end');
      text.setAttribute('dominant-baseline', 'middle');
      text.setAttribute('class', 'radar-label');
      text.textContent = label;
      svg.appendChild(text);
    });

    orderedSeries(series).forEach(item => {
      const emphasized = item.name === 'UCF-Net';
      const pointPairs = item.values.map((value, i) => {
        const scaled = (value - minValue) / (maxValue - minValue);
        return polarPoint(cx, cy, radius * Math.max(0, scaled), start + Math.PI * 2 * i / count);
      });
      const polygon = document.createElementNS(svgNS, 'polygon');
      polygon.setAttribute('points', pointPairs.map(p => p.join(',')).join(' '));
      polygon.setAttribute('class', 'radar-shape');
      polygon.setAttribute('fill', item.color);
      polygon.setAttribute('stroke', item.color);
      polygon.dataset.series = item.name;
      polygon.style.setProperty('fill-opacity', emphasized ? '.16' : '.012', 'important');
      polygon.style.setProperty('stroke-opacity', emphasized ? '1' : '.62', 'important');
      polygon.style.setProperty('stroke-width', emphasized ? '4' : '1.9', 'important');
      if (SERIES_DASHES[item.name]) polygon.setAttribute('stroke-dasharray', SERIES_DASHES[item.name]);
      svg.appendChild(polygon);
      pointPairs.forEach(([x, y], index) => {
        const dot = document.createElementNS(svgNS, 'circle');
        dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('r', emphasized ? 4.5 : 3.1);
        dot.setAttribute('fill', item.color); dot.setAttribute('class', 'radar-dot');
        dot.dataset.series = item.name;
        dot.style.setProperty('opacity', emphasized ? '1' : '.68', 'important');
        setPointAccessibility(dot, item.name, labels[index], item.values[index]);
        svg.appendChild(dot);
      });
    });

    target.replaceChildren(svg);
    attachChartTooltip(target, svg);
  }

  function renderLegend(key, series, visibleNames, onToggle, focusName = '') {
    const container = $(`[data-legend="${key}"]`);
    if (!container) return;
    const fragment = document.createDocumentFragment();
    let focusTarget = null;
    series.forEach(item => {
      const pressed = visibleNames.has(item.name);
      const button = document.createElement('button');
      const swatch = document.createElement('i');
      const label = document.createElement('span');
      button.type = 'button';
      button.className = 'legend-toggle';
      button.dataset.series = item.name;
      button.setAttribute('aria-pressed', String(pressed));
      button.setAttribute('aria-label', `${pressed ? 'Hide' : 'Show'} ${item.name} chart series`);
      button.title = `${pressed ? 'Hide' : 'Show'} ${item.name}`;
      button.style.cssText = `display:inline-flex;align-items:center;gap:5px;padding:3px 5px;border:0;border-radius:6px;background:transparent;color:inherit;font:inherit;cursor:pointer;opacity:${pressed ? '1' : '.38'};`;
      if (item.name === 'UCF-Net') button.style.fontWeight = '800';
      swatch.setAttribute('aria-hidden', 'true');
      swatch.style.cssText = `width:18px;height:0;border-radius:0;background:transparent;border-top:2px ${SERIES_DASHES[item.name] ? 'dashed' : 'solid'} ${item.color};`;
      label.textContent = item.name;
      button.append(swatch, label);
      button.addEventListener('click', () => onToggle(item.name));
      button.addEventListener('mouseenter', () => focusChartSeries(key, pressed ? item.name : ''));
      button.addEventListener('mouseleave', () => focusChartSeries(key));
      button.addEventListener('focus', () => focusChartSeries(key, pressed ? item.name : ''));
      button.addEventListener('blur', () => focusChartSeries(key));
      if (item.name === focusName) focusTarget = button;
      fragment.appendChild(button);
    });
    container.replaceChildren(fragment);
    focusTarget?.focus();
  }

  function createChartController({ key, series, draw }) {
    let availableSeries = series.slice();
    let visibleNames = new Set(availableSeries.map(item => item.name));

    const render = (focusName = '') => {
      const visibleSeries = availableSeries.filter(item => visibleNames.has(item.name));
      draw(visibleSeries);
      renderLegend(key, availableSeries, visibleNames, toggle, focusName);
    };

    const toggle = name => {
      if (!visibleNames.has(name)) {
        visibleNames.add(name);
      } else if (visibleNames.size > 1) {
        visibleNames.delete(name);
      } else {
        return;
      }
      render(name);
    };

    render();
    return {
      setSeries(nextSeries) {
        availableSeries = nextSeries.slice();
        visibleNames = new Set(availableSeries.map(item => item.name));
        render();
      }
    };
  }

  function renderRanking(targetId, ranking) {
    const target = $(targetId);
    if (!target) return;
    const min = Math.min(...ranking.map(item => item[1])) - 2;
    const max = Math.max(...ranking.map(item => item[1]));
    target.innerHTML = ranking.map(([name, score, color], index) => {
      const width = 35 + ((score - min) / (max - min)) * 65;
      return `<div class="rank-row"><span>${String(index + 1).padStart(2, '0')}</span><div class="rank-name"><strong>${name}</strong><div class="rank-bar"><i style="--w:${width.toFixed(1)}%;--c:${color}"></i></div></div><span class="rank-score">${score.toFixed(2)}</span></div>`;
    }).join('');
  }

  function createLineChart(target, labels, series, accessibility) {
    if (!target) return;
    const width = 760;
    const height = 330;
    const pad = { left: 64, right: 26, top: 28, bottom: 48 };
    const isTrainingScale = target.id === 'line-training-scale';
    const isCrossGenerator = target.id === 'line-cross-generator';
    const minY = isTrainingScale ? 60 : isCrossGenerator ? 30 : 0;
    const maxY = isTrainingScale ? 95 : 100;
    const ticks = isTrainingScale ? [60, 70, 80, 90, 95] : isCrossGenerator ? [40, 60, 80, 100] : [0, 20, 40, 60, 80, 100];
    const yAxisTitle = isTrainingScale ? 'mAUC (%)' : 'AUC (%)';
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    appendChartAccessibility(svg, target, accessibility.title, accessibility.description, series);
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const x = i => pad.left + plotW * i / Math.max(1, labels.length - 1);
    const y = value => pad.top + (maxY - value) / (maxY - minY) * plotH;

    ticks.forEach(tick => {
      const line = document.createElementNS(svgNS, 'line');
      line.setAttribute('x1', pad.left); line.setAttribute('x2', width - pad.right); line.setAttribute('y1', y(tick)); line.setAttribute('y2', y(tick)); line.setAttribute('class', 'line-grid');
      svg.appendChild(line);
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', pad.left - 11); text.setAttribute('y', y(tick)); text.setAttribute('text-anchor', 'end'); text.setAttribute('dominant-baseline', 'middle'); text.setAttribute('class', 'line-axis-label'); text.textContent = tick;
      svg.appendChild(text);
    });

    const yLabel = document.createElementNS(svgNS, 'text');
    yLabel.setAttribute('x', 18);
    yLabel.setAttribute('y', pad.top + plotH / 2);
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('class', 'line-axis-title');
    yLabel.setAttribute('transform', `rotate(-90 18 ${pad.top + plotH / 2})`);
    yLabel.textContent = yAxisTitle;
    svg.appendChild(yLabel);

    labels.forEach((label, i) => {
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', x(i)); text.setAttribute('y', height - 16); text.setAttribute('text-anchor', 'middle'); text.setAttribute('class', 'line-axis-label'); text.textContent = label;
      svg.appendChild(text);
    });

    orderedSeries(series).forEach(item => {
      const emphasized = item.name === 'UCF-Net';
      const points = item.values.map((value, i) => [x(i), y(value)]);
      const path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d', `M ${points.map(point => point.join(' ')).join(' L ')}`);
      path.setAttribute('stroke', item.color);
      path.setAttribute('class', 'line-path');
      path.dataset.series = item.name;
      path.style.setProperty('stroke-width', emphasized ? '4.2' : '2.1', 'important');
      path.style.setProperty('stroke-opacity', emphasized ? '1' : '.68', 'important');
      if (!emphasized && SERIES_DASHES[item.name]) path.setAttribute('stroke-dasharray', SERIES_DASHES[item.name]);
      svg.appendChild(path);
      points.forEach(([px, py], i) => {
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('cx', px); circle.setAttribute('cy', py); circle.setAttribute('r', emphasized ? 4.8 : 3.4); circle.setAttribute('fill', item.color); circle.setAttribute('class', 'line-point');
        circle.dataset.series = item.name;
        circle.style.setProperty('opacity', emphasized ? '1' : '.76', 'important');
        setPointAccessibility(circle, item.name, labels[i], item.values[i]);
        svg.appendChild(circle);
      });
    });
    target.replaceChildren(svg);
    attachChartTooltip(target, svg);
  }

  function initTabSet(tabSelector, panelSelector, tabKey, panelKey) {
    const tabs = $$(tabSelector);
    const panels = $$(panelSelector);
    if (!tabs.length) return;

    const activate = (index, moveFocus = false) => {
      const activeTab = tabs[index];
      const key = tabKey(activeTab);
      tabs.forEach(tab => {
        const active = tab === activeTab;
        tab.classList.toggle('active', active);
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
      });
      panels.forEach(panel => {
        const active = panelKey(panel) === key;
        panel.classList.toggle('active', active);
        panel.hidden = !active;
      });
      if (moveFocus) activeTab.focus();
    };

    let initialIndex = tabs.findIndex(tab => tab.getAttribute('aria-selected') === 'true' || tab.classList.contains('active'));
    if (initialIndex < 0) initialIndex = 0;

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => activate(index));
      tab.addEventListener('keydown', event => {
        let nextIndex = null;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') nextIndex = 0;
        if (event.key === 'End') nextIndex = tabs.length - 1;
        if (nextIndex === null) return;
        event.preventDefault();
        activate(nextIndex, true);
      });
    });

    activate(initialIndex);
  }

  function initResults() {
    createChartController({
      key: 'in-domain',
      series: datasets.inDomain.series,
      draw: series => createRadarChart(
        $('#radar-in-domain'),
        datasets.inDomain.labels,
        series,
        datasets.inDomain.min,
        {
          title: 'In-domain AUC comparison',
          description: 'Radar chart comparing detector AUC across six in-domain datasets on a zero-to-one-hundred scale.'
        }
      )
    });
    renderRanking('#ranking-in-domain', datasets.inDomain.ranking);

    createChartController({
      key: 'cross-domain',
      series: datasets.crossDomain.series,
      draw: series => createRadarChart(
        $('#radar-cross-domain'),
        datasets.crossDomain.labels,
        series,
        datasets.crossDomain.min,
        {
          title: 'Cross-domain AUC comparison',
          description: 'Radar chart comparing detector AUC across seven held-out cross-domain datasets on a zero-to-one-hundred scale.'
        }
      )
    });
    renderRanking('#ranking-cross-domain', datasets.crossDomain.ranking);

    createChartController({
      key: 'cross-generator',
      series: datasets.crossGenerator.series,
      draw: series => createLineChart(
        $('#line-cross-generator'),
        datasets.crossGenerator.labels,
        series,
        {
          title: 'Cross-generator few-shot AUC comparison',
          description: 'Line chart comparing detector AUC from zero-shot through one-hundred-shot adaptation on the cross-generator evaluation set.'
        }
      )
    });

    createChartController({
      key: 'training-scale',
      series: datasets.trainingScale.series,
      draw: series => createLineChart(
        $('#line-training-scale'),
        datasets.trainingScale.labels,
        series,
        {
          title: 'Cross-domain AUC by training scale',
          description: 'Line chart comparing detector mean cross-domain AUC when trained on ten-thousand, one-million, and two-million images.'
        }
      )
    });

    initTabSet('.result-tab', '.result-panel', tab => tab.dataset.result, panel => panel.dataset.resultPanel);

    const tableToggle = $('[data-table-toggle]');
    tableToggle?.addEventListener('click', () => {
      const tables = $('.exact-tables');
      if (!tables) return;
      const open = tables.hasAttribute('hidden');
      tables.toggleAttribute('hidden', !open);
      tableToggle.setAttribute('aria-expanded', String(open));
      const label = $('.table-toggle-label', tableToggle);
      if (label) label.textContent = open ? 'Hide exact result tables' : 'View exact result tables';
      const iconPath = tableToggle.querySelector('.table-toggle-icon path');
      if (iconPath) iconPath.setAttribute('d', open ? 'M5 12h14' : 'M12 5v14M5 12h14');
    });
  }

  function initMiniCharts() {
    $$('[data-mini-chart]').forEach(target => {
      const rows = miniData[target.dataset.miniChart];
      const min = Math.min(...rows.map(row => row[1])) - .6;
      const max = Math.max(...rows.map(row => row[1]));
      target.innerHTML = rows.map(([label, value, color]) => {
        const width = 28 + ((value - min) / (max - min)) * 72;
        return `<div class="mini-row"><label title="${label}">${label}</label><div class="mini-track"><i style="--w:${width.toFixed(1)}%;--c:${color}"></i></div><strong>${value.toFixed(2)}</strong></div>`;
      }).join('');
    });
  }

  function initVisualTabs() {
    initTabSet('.visual-tab', '.visual-panel', tab => tab.dataset.visual, panel => panel.dataset.visualPanel);
  }

  function initLightbox() {
    const dialog = $('.lightbox');
    const image = $('.lightbox img');
    const caption = $('.lightbox figcaption');
    if (!dialog || typeof dialog.showModal !== 'function') return;
    $$('[data-lightbox]').forEach(button => button.addEventListener('click', () => {
      image.src = button.dataset.lightbox;
      image.alt = button.dataset.caption || 'Expanded figure';
      caption.textContent = button.dataset.caption || '';
      dialog.showModal();
      document.body.classList.add('lock-scroll');
    }));
    $('.lightbox-close')?.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      const rect = dialog.getBoundingClientRect();
      const outside = event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom;
      if (outside) dialog.close();
    });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('lock-scroll');
      image.removeAttribute('src');
    });
  }

  function initCopyCitation() {
    $('[data-copy-bibtex]')?.addEventListener('click', async () => {
      const text = $('#bibtex')?.textContent || '';
      const status = $('.copy-status');
      try {
        await navigator.clipboard.writeText(text);
        status.textContent = 'BibTeX copied to clipboard.';
      } catch {
        const range = document.createRange();
        range.selectNodeContents($('#bibtex'));
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'BibTeX selected. Press Ctrl/Cmd+C to copy.';
      }
      window.setTimeout(() => { status.textContent = ''; }, 2600);
    });
  }

  function init() {
    initTheme();
    initNavigation();
    initReveal();
    initResults();
    initMiniCharts();
    initVisualTabs();
    initLightbox();
    initCopyCitation();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
