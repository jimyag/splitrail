export const TABLE_PAGE_SIZE = 100;
export const SERIES_PAGE_SIZE = 12;

export function metricValue(row, metric) {
  return metric === 'total' ? row.inputTokens + row.outputTokens + row.cachedTokens : row[metric];
}

export function localDateKey(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

export function rangeStart(range, now = new Date()) {
  if (range === 'all') return null;
  const cutoff = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  cutoff.setDate(cutoff.getDate() - Number(range) + 1);
  return localDateKey(cutoff);
}

export function periodKey(date, hour, granularity) {
  if (granularity === 'hour') return `${date}T${String(hour).padStart(2, '0')}`;
  if (granularity === 'day') return date;
  if (granularity === 'month') return date.slice(0, 7);
  if (granularity === 'year') return date.slice(0, 4);
  const monday = new Date(`${date}T12:00:00`);
  monday.setDate(monday.getDate() - (monday.getDay() + 6) % 7);
  return localDateKey(monday);
}

export function selectPeriods(rows, filters, granularity, range, now = new Date()) {
  const filtered = rows.filter(row => ['model', 'project', 'tool'].every(key =>
    !filters[key].size || filters[key].has(row[key])));
  const first = range === 'all'
    ? filtered.reduce((min, row) => !min || row.date < min ? row.date : min, '')
    : rangeStart(range, now);
  const last = localDateKey(now);
  if (!first) return { periods: [], selectedRows: [] };
  const periods = new Map();
  const selectedRows = [];
  const cursor = new Date(`${first}T12:00:00`);
  const end = new Date(`${last}T12:00:00`);
  while (cursor <= end) {
    const date = localDateKey(cursor);
    const lastHour = granularity === 'hour' ? (date === last ? now.getHours() : 23) : 0;
    for (let hour = 0; hour <= lastHour; hour++) {
      const key = periodKey(date, hour, granularity);
      if (!periods.has(key)) periods.set(key, {
        key, inputTokens: 0, outputTokens: 0, cachedTokens: 0, reasoningTokens: 0, cost: 0,
      });
    }
    cursor.setDate(cursor.getDate() + 1);
  }
  for (const row of filtered) {
    if (row.date < first || row.date > last) continue;
    const period = periods.get(periodKey(row.date, row.hour, granularity));
    if (!period) continue;
    selectedRows.push(row);
    for (const field of ['inputTokens', 'outputTokens', 'cachedTokens', 'reasoningTokens', 'cost']) period[field] += row[field];
  }
  return { periods: [...periods.values()], selectedRows };
}

export function pageBounds(length, page, size) {
  const count = Math.max(1, Math.ceil(length / size));
  page = Math.max(0, Math.min(page, count - 1));
  return { page, count, start: page * size, end: Math.min(length, (page + 1) * size) };
}

export function detailPage(periods, metric, page) {
  const indexes = [];
  for (let index = periods.length - 1; index >= 0; index--) {
    if (metricValue(periods[index], metric) > 0) indexes.push(index);
  }
  const bounds = pageBounds(indexes.length, page, TABLE_PAGE_SIZE);
  return { indexes: indexes.slice(bounds.start, bounds.end), total: indexes.length, bounds };
}

export function usageSeries(periods, rows, metric, granularity, dimensions, page) {
  const groups = new Map();
  if (dimensions.length) for (const row of rows) {
    const value = metricValue(row, metric);
    if (value <= 0) continue;
    const values = dimensions.map(key => [key, row[key]]);
    const id = JSON.stringify(values);
    if (!groups.has(id)) groups.set(id, { id, dimensions: values, total: 0 });
    groups.get(id).total += value;
  }
  const ranked = [...groups.values()].sort((a, b) => b.total - a.total || a.id.localeCompare(b.id));
  const bounds = pageBounds(ranked.length, page, SERIES_PAGE_SIZE);
  const visible = ranked.slice(bounds.start, bounds.end).map(group => ({ ...group, values: Array(periods.length).fill(0) }));
  const visibleById = new Map(visible.map(group => [group.id, group]));
  const indexes = new Map(periods.map((period, index) => [period.key, index]));
  if (visible.length) for (const row of rows) {
    const group = visibleById.get(JSON.stringify(dimensions.map(key => [key, row[key]])));
    const index = indexes.get(periodKey(row.date, row.hour, granularity));
    if (group && index !== undefined) group.values[index] += metricValue(row, metric);
  }
  return { groups: visible, combinationCount: ranked.length, bounds };
}

// Preserve each bucket's extrema, including narrow spikes, without rendering
// every hourly point in a long history. The summary and table remain exact.
export function chartSamples(values, limit = 800) {
  if (values.length <= limit) return values.map((value, index) => ({ value, index }));
  const samples = [{ value: values[0], index: 0 }];
  const buckets = Math.floor((limit - 2) / 2);
  for (let bucket = 0; bucket < buckets; bucket++) {
    const start = 1 + Math.floor(bucket * (values.length - 2) / buckets);
    const end = 1 + Math.floor((bucket + 1) * (values.length - 2) / buckets);
    let min = start, max = start;
    for (let index = start + 1; index < end; index++) {
      if (values[index] < values[min]) min = index;
      if (values[index] > values[max]) max = index;
    }
    for (const index of [...new Set([min, max])].sort((a, b) => a - b)) samples.push({ value: values[index], index });
  }
  samples.push({ value: values.at(-1), index: values.length - 1 });
  return samples;
}
