const controls = Object.fromEntries(['model', 'project', 'tool', 'metric', 'granularity', 'range'].map(id => [id, document.getElementById(id)]));
const messages = {
  zh: {
    title: 'Splitrail · 使用趋势', usageTrends: '使用趋势', languageLabel: '语言', localData: '本地数据',
    headline: '使用变化，一目了然。', intro: '按时间、模型、项目和工具查看 token 使用变化。', refresh: '刷新数据',
    filters: '筛选条件', model: '模型', allModels: '所有模型', project: '项目', allProjects: '所有项目',
    tool: '工具', allTools: '所有工具', metric: '指标', granularity: '统计粒度', hour: '小时', day: '日', week: '周', month: '月', year: '年',
    range: '时间范围', last1: '今天', last7: '近 7 天', last30: '近 30 天', last90: '近 90 天', allTime: '全部',
    total: 'Token 总量', cachedTokens: '缓存', outputTokens: '输出', inputTokens: '输入', reasoningTokens: '推理', cost: '成本',
    loading: '正在读取本地统计数据…', loaded: count => `已读取 ${number.format(count)} 条小时模型 / 项目记录`, noData: '本机暂无可用的使用记录。',
    loadError: detail => `读取失败：${detail}`, summary: '统计摘要', rangeTotal: '所选范围总量',
    trend: (unit, metric) => `按${unit}统计 ${metric}`, details: unit => `按${unit}明细`, averageLabel: unit => `每${unit}平均`,
    peakLabel: unit => `最高单${unit}`, activeLabel: '活跃时段', periodHeading: '时段', periodCount: count => `共 ${count} 个时段`,
    perUnit: (unit, cost) => `${cost ? 'USD' : 'tokens'} / ${unit}`, clearAll: '清除全部', removeFilter: label => `移除 ${label}`,
    scrollHint: '滚动可查看更早时段', legendHint: '点击图例或曲线只看一条，再点恢复全部。', legendAria: '折线图图例', seriesTotal: '总计',
    chartAria: '每日使用趋势图', metricNote: 'Token 总量 = 输入 + 输出 + 缓存；推理 token 单独列示，与现有 TUI 口径一致。',
    empty: '当前筛选条件下没有使用记录。', unknownModel: '未知模型', unknownProject: '未知项目', noMatches: '没有匹配的选项',
  },
  en: {
    title: 'Splitrail · Usage Trends', usageTrends: 'Usage Trends', languageLabel: 'Language', localData: 'Local data',
    headline: 'Your usage over time, at a glance.', intro: 'Explore token usage by time, model, project, and tool.', refresh: 'Refresh data',
    filters: 'Filters', model: 'Model', allModels: 'All models', project: 'Project', allProjects: 'All projects',
    tool: 'Tool', allTools: 'All tools', metric: 'Metric', granularity: 'Group by', hour: 'Hour', day: 'Day', week: 'Week', month: 'Month', year: 'Year',
    range: 'Time range', last1: 'Today', last7: 'Last 7 days', last30: 'Last 30 days', last90: 'Last 90 days', allTime: 'All time',
    total: 'Total tokens', cachedTokens: 'Cached tokens', outputTokens: 'Output tokens', inputTokens: 'Input tokens', reasoningTokens: 'Reasoning tokens', cost: 'Cost',
    loading: 'Loading local usage data…', loaded: count => `Loaded ${number.format(count)} hourly model/project records`, noData: 'No local usage records found.',
    loadError: detail => `Could not load data: ${detail}`, summary: 'Summary', rangeTotal: 'Total in range',
    trend: (unit, metric) => `${metric} by ${unit.toLowerCase()}`, details: unit => `${unit} details`, averageLabel: unit => `Average per ${unit.toLowerCase()}`,
    peakLabel: unit => `Peak ${unit.toLowerCase()}`, activeLabel: 'Active periods', periodHeading: 'Period', periodCount: count => `${count} ${count === 1 ? 'period' : 'periods'}`,
    perUnit: (unit, cost) => `${cost ? 'USD' : 'tokens'} / ${unit.toLowerCase()}`, clearAll: 'Clear all', removeFilter: label => `Remove ${label}`,
    scrollHint: 'Scroll to see earlier periods', legendHint: 'Click a legend item or line to show only that series; click again to show all.', legendAria: 'Line chart legend', seriesTotal: 'Total',
    chartAria: 'Daily usage trend chart', metricNote: 'Total tokens = input + output + cached. Reasoning tokens are shown separately, matching the TUI.',
    empty: 'No usage records match these filters.', unknownModel: 'Unknown model', unknownProject: 'Unknown project', noMatches: 'No matching options',
  },
};
const savedLanguage = (() => { try { return localStorage.getItem('splitrail-web-language'); } catch { return null; } })();
let language = savedLanguage === 'zh' || savedLanguage === 'en' ? savedLanguage : (navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en');
let number;
let money;
let fullDate;
let shortDate;
let monthDate;
const svgNS = 'http://www.w3.org/2000/svg';
let rows = [];
let loaded = false;
let statusState = { kind: 'loading' };
const choices = { model: [], project: [], tool: [] };
const selectedFilters = { model: new Set(), project: new Set(), tool: new Set() };
const menuState = { key: null, values: [], active: 0 };
let isolatedSeriesId = null;
let chartState = null;

function t(key) { return messages[language][key]; }

function formatDate(date, short = false) {
  return (short ? shortDate : fullDate).format(new Date(`${date}T12:00:00`));
}

function renderStatus() {
  const status = document.getElementById('status');
  if (statusState.kind === 'loading') status.textContent = t('loading');
  else if (statusState.kind === 'error') status.textContent = t('loadError')(statusState.detail);
  else status.textContent = rows.length ? t('loaded')(rows.length) : t('noData');
}

function setLanguage(next, persist = false) {
  language = next;
  if (menuState.key) hideMenu(menuState.key);
  const locale = language === 'zh' ? 'zh-CN' : 'en-US';
  number = new Intl.NumberFormat(locale, { maximumFractionDigits: 0 });
  money = new Intl.NumberFormat(locale, { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 4 });
  fullDate = new Intl.DateTimeFormat(locale, { year: 'numeric', month: 'short', day: 'numeric' });
  shortDate = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric' });
  monthDate = new Intl.DateTimeFormat(locale, { year: 'numeric', month: 'short' });
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.title = t('title');
  document.getElementById('language').value = language;
  document.getElementById('language').setAttribute('aria-label', t('languageLabel'));
  for (const element of document.querySelectorAll('[data-i18n]')) element.textContent = t(element.dataset.i18n);
  for (const element of document.querySelectorAll('[data-i18n-aria]')) element.setAttribute('aria-label', t(element.dataset.i18nAria));
  for (const element of document.querySelectorAll('[data-i18n-placeholder]')) element.placeholder = t(element.dataset.i18nPlaceholder);
  document.getElementById('chart-legend').setAttribute('aria-label', t('legendAria'));
  renderLabels();
  for (const key of ['model', 'project', 'tool']) updateChoices(key);
  renderStatus();
  if (loaded) render();
  if (persist) { try { localStorage.setItem('splitrail-web-language', language); } catch { /* The page still works without storage. */ } }
}

function metricValue(day, metric) {
  return metric === 'total' ? day.inputTokens + day.outputTokens + day.cachedTokens : day[metric];
}

function format(value, metric) {
  return metric === 'cost' ? money.format(value) : number.format(value);
}

function choiceLabel(key, value) {
  if (key === 'model' && value === 'Unknown model') return t('unknownModel');
  if (key === 'project' && value === '') return t('unknownProject');
  return value;
}

function renderChips(key) {
  const container = document.getElementById(`${key}-selected`);
  container.hidden = selectedFilters[key].size === 0;
  container.replaceChildren();
  for (const value of selectedFilters[key]) {
    const label = choiceLabel(key, value);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'selected-chip';
    chip.title = label;
    chip.setAttribute('aria-label', t('removeFilter')(label));
    const name = document.createElement('span');
    name.className = 'chip-name';
    name.textContent = label;
    const remove = document.createElement('span');
    remove.setAttribute('aria-hidden', 'true');
    remove.textContent = '×';
    chip.append(name, remove);
    chip.addEventListener('click', () => {
      selectedFilters[key].delete(value);
      renderChips(key);
      if (menuState.key === key) showMenu(key, controls[key].value);
      if (loaded) render();
    });
    container.append(chip);
  }
}

function updateChoices(key, values) {
  const openQuery = menuState.key === key ? controls[key].value : null;
  if (values) choices[key] = [...values].sort((a, b) => a.localeCompare(b, language === 'zh' ? 'zh-CN' : 'en-US'));
  for (const value of selectedFilters[key]) if (!choices[key].includes(value)) selectedFilters[key].delete(value);
  renderChips(key);
  if (openQuery === null) controls[key].value = '';
  else showMenu(key, openQuery);
}

function hideMenu(key) {
  document.getElementById(`${key}-options`).hidden = true;
  controls[key].setAttribute('aria-expanded', 'false');
  controls[key].removeAttribute('aria-activedescendant');
  controls[key].value = '';
  if (menuState.key === key) menuState.key = null;
}

function setActiveOption(key, index) {
  const menu = document.getElementById(`${key}-options`);
  const options = [...menu.querySelectorAll('[role="option"]')];
  if (!options.length) return;
  menuState.active = (index + options.length) % options.length;
  options.forEach((option, i) => option.classList.toggle('active', i === menuState.active));
  controls[key].setAttribute('aria-activedescendant', options[menuState.active].id);
  options[menuState.active].scrollIntoView({ block: 'nearest' });
}

function showMenu(key, query = '') {
  if (menuState.key && menuState.key !== key) hideMenu(menuState.key);
  const menu = document.getElementById(`${key}-options`);
  const needle = query.toLocaleLowerCase(language === 'zh' ? 'zh-CN' : 'en-US');
  menuState.values = query
    ? choices[key].filter(value => choiceLabel(key, value).toLocaleLowerCase(language === 'zh' ? 'zh-CN' : 'en-US').includes(needle))
    : [null, ...choices[key]];
  menuState.key = key;
  menu.replaceChildren();
  if (!menuState.values.length) {
    const empty = document.createElement('div');
    empty.className = 'choice-empty';
    empty.textContent = t('noMatches');
    menu.append(empty);
  }
  menuState.values.forEach((value, index) => {
    const option = document.createElement('div');
    option.id = `${key}-option-${index}`;
    option.className = 'choice-option';
    option.setAttribute('role', 'option');
    option.setAttribute('aria-selected', String(value !== null && selectedFilters[key].has(value)));
    option.textContent = value === null ? t('clearAll') : choiceLabel(key, value);
    option.addEventListener('pointerdown', event => { event.preventDefault(); toggleChoice(key, value); });
    option.addEventListener('click', event => { if (event.detail === 0) toggleChoice(key, value); });
    menu.append(option);
  });
  menu.hidden = false;
  controls[key].setAttribute('aria-expanded', 'true');
  setActiveOption(key, 0);
}

function toggleChoice(key, value) {
  if (value === null) selectedFilters[key].clear();
  else if (selectedFilters[key].has(value)) selectedFilters[key].delete(value);
  else selectedFilters[key].add(value);
  renderChips(key);
  showMenu(key, controls[key].value);
  if (loaded) render();
}

function handleChoiceInput(key) {
  showMenu(key, controls[key].value);
}

function localDateKey(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function rangeStart() {
  if (controls.range.value === 'all') return null;
  const today = new Date();
  const cutoff = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  cutoff.setDate(cutoff.getDate() - Number(controls.range.value) + 1);
  return localDateKey(cutoff);
}

function refreshChoices() {
  const first = rangeStart();
  const last = localDateKey(new Date());
  const available = rows.filter(row =>
    (!first || row.date >= first) && row.date <= last && metricValue(row, controls.metric.value) > 0);
  for (const key of ['model', 'project', 'tool']) updateChoices(key, new Set(available.map(row => row[key])));
}

function periodKey(date, hour, granularity) {
  if (granularity === 'hour') return `${date}T${String(hour).padStart(2, '0')}`;
  if (granularity === 'day') return date;
  if (granularity === 'month') return date.slice(0, 7);
  if (granularity === 'year') return date.slice(0, 4);
  const monday = new Date(`${date}T12:00:00`);
  monday.setDate(monday.getDate() - (monday.getDay() + 6) % 7);
  return localDateKey(monday);
}

function periodLabel(key, granularity, short = false) {
  if (granularity === 'hour') {
    const [date, hour] = key.split('T');
    return `${formatDate(date, short)} ${hour}:00`;
  }
  if (granularity === 'day') return formatDate(key, short);
  if (granularity === 'week') {
    if (short) return formatDate(key, true);
    const sunday = new Date(`${key}T12:00:00`);
    sunday.setDate(sunday.getDate() + 6);
    return `${formatDate(key)} – ${formatDate(localDateKey(sunday))}`;
  }
  if (granularity === 'month') return monthDate.format(new Date(`${key}-01T12:00:00`));
  return key;
}

function emptyPeriod(key) {
  return { key, inputTokens: 0, outputTokens: 0, cachedTokens: 0, reasoningTokens: 0, cost: 0 };
}

function selectedPeriods() {
  const filtered = rows.filter(row =>
    (!selectedFilters.model.size || selectedFilters.model.has(row.model)) &&
    (!selectedFilters.project.size || selectedFilters.project.has(row.project)) &&
    (!selectedFilters.tool.size || selectedFilters.tool.has(row.tool)));
  const granularity = controls.granularity.value;
  const today = new Date();
  const rangeFirst = rangeStart();
  const first = controls.range.value === 'all'
    ? filtered.reduce((min, row) => !min || row.date < min ? row.date : min, '')
    : rangeFirst;
  const last = localDateKey(today);
  if (!first) return { periods: [], selectedRows: [] };
  const periods = new Map();
  const selectedRows = [];
  const cursor = new Date(`${first}T12:00:00`);
  const end = new Date(`${last}T12:00:00`);
  while (cursor <= end) {
    const date = localDateKey(cursor);
    if (granularity === 'hour') {
      const lastHour = date === last ? today.getHours() : 23;
      for (let hour = 0; hour <= lastHour; hour++) {
        const key = periodKey(date, hour, granularity);
        periods.set(key, emptyPeriod(key));
      }
    } else {
      const key = periodKey(date, 0, granularity);
      if (!periods.has(key)) periods.set(key, emptyPeriod(key));
    }
    cursor.setDate(cursor.getDate() + 1);
  }
  for (const row of filtered) {
    if (row.date < first || row.date > last) continue;
    const key = periodKey(row.date, row.hour, granularity);
    const period = periods.get(key);
    if (!period) continue;
    selectedRows.push(row);
    for (const field of ['inputTokens', 'outputTokens', 'cachedTokens', 'reasoningTokens', 'cost']) period[field] += row[field];
  }
  return { periods: [...periods.values()], selectedRows };
}

function svgElement(name, attrs = {}, content) {
  const element = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
  if (content !== undefined) element.textContent = content;
  return element;
}

function renderLabels() {
  const unit = t(controls.granularity.value);
  const metric = controls.metric.value;
  const title = t('trend')(unit, t(metric));
  document.getElementById('chart-title').textContent = title;
  document.getElementById('chart').setAttribute('aria-label', title);
  document.getElementById('details-title').textContent = t('details')(unit);
  document.getElementById('table-metric').textContent = t(metric);
  document.getElementById('period-heading').textContent = t('periodHeading');
  document.getElementById('average-label').textContent = t('averageLabel')(unit);
  document.getElementById('peak-label').textContent = t('peakLabel')(unit);
  document.getElementById('active-label').textContent = t('activeLabel');
  document.getElementById('average-unit').textContent = t('perUnit')(unit, metric === 'cost');
}

function groupedSeries(periods, selectedRows, metric, granularity) {
  const dimensions = ['project', 'model', 'tool'].filter(key => selectedFilters[key].size);
  const periodIndexes = new Map(periods.map((period, index) => [period.key, index]));
  const groups = new Map();
  for (const row of selectedRows) {
    const value = metricValue(row, metric);
    if (value <= 0 || !dimensions.length) continue;
    const index = periodIndexes.get(periodKey(row.date, row.hour, granularity));
    if (index === undefined) continue;
    const values = dimensions.map(key => [key, row[key]]);
    const id = JSON.stringify(values);
    if (!groups.has(id)) groups.set(id, {
      id,
      label: values.map(([key, item]) => `${t(key)}: ${choiceLabel(key, item)}`).join(' · '),
      displayLabel: values.map(([key, item]) => {
        const label = choiceLabel(key, item);
        const shortPath = key === 'project' && item ? item.split(/[\\/]/).filter(Boolean).slice(-2).join('/') : '';
        return `${t(key)}: ${shortPath || label}`;
      }).join(' · '),
      values: Array(periods.length).fill(0),
    });
    groups.get(id).values[index] += value;
  }
  const palette = ['#d05a51', '#238a80', '#ce8b22', '#8a62bd', '#3778bd', '#a9558c', '#73933d', '#bd6b42'];
  const combinations = [...groups.values()].sort((a, b) => a.label.localeCompare(b.label, language === 'zh' ? 'zh-CN' : 'en-US'));
  combinations.forEach((series, index) => { series.color = palette[index % palette.length]; });
  return [{ id: 'total', label: t('seriesTotal'), displayLabel: t('seriesTotal'), color: '#514bc6', values: periods.map(period => metricValue(period, metric)) }, ...combinations];
}

function isolateSeries(id) {
  isolatedSeriesId = isolatedSeriesId === id ? null : id;
  if (chartState) renderChart(chartState.periods, chartState.series, chartState.metric, chartState.granularity, true);
}

function renderChart(periods, series, metric, granularity, preserveScroll = false) {
  const container = document.getElementById('chart');
  const legend = document.getElementById('chart-legend');
  const previousScroll = container.scrollLeft;
  container.replaceChildren();
  legend.replaceChildren();
  if (!periods.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = t('empty');
    container.append(empty);
    return;
  }
  const width = Math.max(760, periods.length * (granularity === 'hour' ? 14 : 26) + 72);
  const height = 280;
  const left = 110, right = 24, top = 24, bottom = 42;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const visible = series.filter(item => !isolatedSeriesId || item.id === isolatedSeriesId);
  const max = visible.reduce((largest, item) => item.values.reduce((peak, value) => Math.max(peak, value), largest), 1);
  const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, width, height, 'aria-hidden': 'true' });
  for (let i = 0; i <= 4; i++) {
    const y = top + plotHeight * i / 4;
    svg.append(svgElement('line', { x1: left, y1: y, x2: width - right, y2: y, class: 'grid-line' }));
    svg.append(svgElement('text', { x: left - 10, y: y + 4, 'text-anchor': 'end', class: 'axis-label' }, format(max * (4 - i) / 4, metric)));
  }
  const labelEvery = Math.max(1, Math.ceil(periods.length / Math.floor(plotWidth / 85)));
  periods.forEach((period, i) => {
    const x = left + (i + 0.5) * plotWidth / periods.length;
    if (i % labelEvery === 0 || (i === periods.length - 1 && i % labelEvery >= Math.ceil(labelEvery / 2))) svg.append(svgElement('text', { x, y: height - 13, 'text-anchor': 'middle', class: 'axis-label' }, periodLabel(period.key, granularity, true)));
  });
  for (const item of series) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'legend-item';
    button.classList.toggle('dimmed', !!isolatedSeriesId && isolatedSeriesId !== item.id);
    button.style.setProperty('--line-color', item.color);
    button.setAttribute('aria-pressed', String(isolatedSeriesId === item.id));
    button.title = item.label;
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'legend-label';
    label.textContent = item.displayLabel;
    button.append(swatch, label);
    button.addEventListener('click', () => isolateSeries(item.id));
    legend.append(button);
    if (isolatedSeriesId && isolatedSeriesId !== item.id) continue;
    const points = item.values.map((value, index) => ({
      x: left + (index + 0.5) * plotWidth / periods.length,
      y: top + plotHeight - value / max * plotHeight,
    }));
    const path = points.map((point, index) => `${index ? 'L' : 'M'}${point.x},${point.y}`).join(' ');
    svg.append(svgElement('path', { d: path, stroke: item.color, 'stroke-width': item.id === 'total' ? 3 : 2.5, class: 'series-line' }));
    const hit = svgElement('path', { d: path, class: 'series-hit' });
    hit.append(svgElement('title', {}, item.label));
    hit.addEventListener('click', () => isolateSeries(item.id));
    svg.append(hit);
    if (periods.length <= 90) points.forEach((point, index) => {
      const circle = svgElement('circle', { cx: point.x, cy: point.y, r: 3.5, fill: item.color, class: 'series-point' });
      circle.append(svgElement('title', {}, `${item.label} · ${periodLabel(periods[index].key, granularity)} · ${format(item.values[index], metric)}`));
      circle.addEventListener('click', () => isolateSeries(item.id));
      svg.append(circle);
    });
  }
  container.append(svg);
  container.scrollLeft = preserveScroll ? previousScroll : container.scrollWidth;
}

function renderTable(periods, series, metric, granularity) {
  const body = document.getElementById('table-body');
  body.replaceChildren();
  document.getElementById('row-count').textContent = t('periodCount')(periods.length);
  document.querySelectorAll('th.series-column').forEach(header => header.remove());
  const combinations = series.slice(1);
  let previous = document.getElementById('table-metric');
  for (const item of combinations) {
    const header = document.createElement('th');
    header.className = 'series-column';
    header.textContent = item.displayLabel;
    header.title = item.label;
    previous.after(header);
    previous = header;
  }
  for (let index = periods.length - 1; index >= 0; index--) {
    const period = periods[index];
    const tr = document.createElement('tr');
    const values = [periodLabel(period.key, granularity), format(metricValue(period, metric), metric),
      ...combinations.map(item => format(item.values[index], metric)),
      format(period.inputTokens, 'inputTokens'), format(period.outputTokens, 'outputTokens'), format(period.cachedTokens, 'cachedTokens'), format(period.reasoningTokens, 'reasoningTokens'), format(period.cost, 'cost')];
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = value;
      tr.append(cell);
    }
    body.append(tr);
  }
}

function render() {
  const metric = controls.metric.value;
  const granularity = controls.granularity.value;
  const { periods, selectedRows } = selectedPeriods();
  const series = groupedSeries(periods, selectedRows, metric, granularity);
  if (isolatedSeriesId && !series.some(item => item.id === isolatedSeriesId)) isolatedSeriesId = null;
  const values = periods.map(period => metricValue(period, metric));
  const sum = values.reduce((a, b) => a + b, 0);
  const peak = values.reduce((largest, value) => Math.max(largest, value), 0);
  const peakIndex = values.indexOf(peak);
  renderLabels();
  document.getElementById('sum').textContent = format(sum, metric);
  document.getElementById('average').textContent = format(periods.length ? sum / periods.length : 0, metric);
  document.getElementById('peak').textContent = format(peak, metric);
  document.getElementById('peak-date').textContent = peak > 0 ? periodLabel(periods[peakIndex].key, granularity) : '—';
  document.getElementById('active-days').textContent = String(periods.filter(period => period.inputTokens + period.outputTokens + period.cachedTokens + period.reasoningTokens > 0).length);
  document.getElementById('period-days').textContent = t('periodCount')(periods.length);
  document.getElementById('sum-unit').textContent = metric === 'cost' ? 'USD' : 'tokens';
  chartState = { periods, series, metric, granularity };
  renderChart(periods, series, metric, granularity);
  renderTable(periods, series, metric, granularity);
}

async function load() {
  const refresh = document.getElementById('refresh');
  refresh.disabled = true;
  statusState = { kind: 'loading' };
  renderStatus();
  try {
    const response = await fetch('/api/usage', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    rows = await response.json();
    refreshChoices();
    loaded = true;
    statusState = { kind: 'loaded' };
    renderStatus();
    render();
  } catch (error) {
    statusState = { kind: 'error', detail: error.message };
    renderStatus();
  } finally {
    refresh.disabled = false;
  }
}

for (const key of ['model', 'project', 'tool']) {
  controls[key].addEventListener('focus', () => showMenu(key, controls[key].value));
  controls[key].addEventListener('input', () => handleChoiceInput(key));
  controls[key].addEventListener('keydown', event => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (menuState.key !== key) showMenu(key, controls[key].value);
      else setActiveOption(key, menuState.active + (event.key === 'ArrowDown' ? 1 : -1));
    } else if (event.key === 'Enter' && menuState.key === key) {
      event.preventDefault();
      if (menuState.values.length) toggleChoice(key, menuState.values[menuState.active]);
      else hideMenu(key);
    } else if (event.key === 'Escape' && menuState.key === key) {
      event.preventDefault();
      hideMenu(key);
    }
  });
  controls[key].addEventListener('blur', () => hideMenu(key));
}
for (const key of ['metric', 'range']) controls[key].addEventListener('change', () => { refreshChoices(); render(); });
controls.granularity.addEventListener('change', render);
document.getElementById('language').addEventListener('change', event => setLanguage(event.target.value, true));
document.getElementById('refresh').addEventListener('click', load);
setLanguage(language);
load();
