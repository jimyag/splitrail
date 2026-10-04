import test from 'node:test';
import assert from 'node:assert/strict';
import { TABLE_PAGE_SIZE, SERIES_PAGE_SIZE, metricValue, periodKey, selectPeriods, pageBounds, detailPage, usageSeries, chartSamples } from './data.mjs';

const row = (overrides = {}) => ({
  date: '2025-12-31', hour: 23, model: 'a', project: 'p', tool: 't',
  inputTokens: 10, outputTokens: 3, cachedTokens: 2, reasoningTokens: 4, cost: 0.1, ...overrides,
});
const filters = (overrides = {}) => ({ model: new Set(), project: new Set(), tool: new Set(), ...overrides });
const now = new Date(2026, 0, 2, 12);

test('all granularities preserve token and cost totals across year and week boundaries', () => {
  const rows = [row(), row({ date: '2026-01-01', hour: 0, inputTokens: 20 })];
  for (const granularity of ['hour', 'day', 'week', 'month', 'year']) {
    const { periods } = selectPeriods(rows, filters(), granularity, 'all', now);
    assert.equal(periods.reduce((sum, period) => sum + metricValue(period, 'total'), 0), 40);
    assert.equal(periods.reduce((sum, period) => sum + period.reasoningTokens, 0), 8);
    assert.equal(periods.reduce((sum, period) => sum + period.cost, 0), 0.2);
  }
  assert.equal(periodKey('2025-12-31', 23, 'week'), '2025-12-29');
  assert.equal(periodKey('2026-01-04', 0, 'week'), '2025-12-29');
  assert.equal(periodKey('2026-01-05', 0, 'week'), '2026-01-05');
});

test('filters union within a dimension and intersect across dimensions', () => {
  const rows = [row(), row({ model: 'b' }), row({ model: 'c' }), row({ project: 'other' }), row({ tool: 'other' })];
  const selected = filters({ model: new Set(['a', 'b']), project: new Set(['p']), tool: new Set(['t']) });
  const { periods, selectedRows } = selectPeriods(rows, selected, 'day', 'all', now);
  assert.equal(selectedRows.length, 2);
  assert.equal(periods.reduce((sum, period) => sum + metricValue(period, 'total'), 0), 30);
});

test('range includes its first date, preserves gaps and omits future hours', () => {
  const rows = [row(), row({ date: '2026-01-01', hour: 0 }), row({ date: '2026-01-02', hour: 13 })];
  const { periods, selectedRows } = selectPeriods(rows, filters(), 'hour', '2', now);
  assert.equal(periods.length, 37);
  assert.equal(periods[0].key, '2026-01-01T00');
  assert.equal(periods.at(-1).key, '2026-01-02T12');
  assert.equal(selectedRows.length, 1);
  assert.equal(metricValue(periods[1], 'total'), 0);
  assert.deepEqual(selectPeriods([], filters(), 'day', 'all', now), { periods: [], selectedRows: [] });
});

test('combination pages cover every group exactly once and retain exact values', () => {
  const rows = Array.from({ length: 100 }, (_, index) => row({ model: `model-${index}`, inputTokens: index + 1 }));
  const { periods, selectedRows } = selectPeriods(rows, filters(), 'day', 'all', now);
  const ids = new Set();
  let total = 0;
  for (let page = 0; page < Math.ceil(rows.length / SERIES_PAGE_SIZE); page++) {
    const { groups, combinationCount } = usageSeries(periods, selectedRows, 'total', 'day', ['model'], page);
    assert.equal(combinationCount, 100);
    assert.ok(groups.length <= SERIES_PAGE_SIZE);
    for (const group of groups) {
      assert.ok(!ids.has(group.id));
      ids.add(group.id);
      assert.equal(group.values.reduce((a, b) => a + b, 0), group.total);
      total += group.total;
    }
  }
  assert.equal(ids.size, 100);
  assert.equal(total, periods.reduce((sum, period) => sum + metricValue(period, 'total'), 0));
  const first = usageSeries(periods, selectedRows, 'total', 'day', ['model'], 0);
  assert.equal(first.groups[0].dimensions[0][1], 'model-99');
  assert.deepEqual(usageSeries(periods, selectedRows, 'total', 'day', [], 0).groups, []);
  assert.deepEqual(usageSeries(periods, [row({ inputTokens: 0 })], 'inputTokens', 'day', ['model'], 0).groups, []);
});

test('a year of hourly detail is paged without omitting periods', () => {
  const { periods } = selectPeriods([row({ date: '2025-01-02' })], filters(), 'hour', 'all', now);
  assert.ok(periods.length > 8700);
  let count = 0;
  for (let page = 0; page < Math.ceil(periods.length / TABLE_PAGE_SIZE); page++) {
    const bounds = pageBounds(periods.length, page, TABLE_PAGE_SIZE);
    assert.ok(bounds.end - bounds.start <= 100);
    count += bounds.end - bounds.start;
  }
  assert.equal(count, periods.length);
  assert.equal(pageBounds(10, 99, TABLE_PAGE_SIZE).page, 0);
  assert.deepEqual(pageBounds(0, 0, TABLE_PAGE_SIZE), { page: 0, count: 1, start: 0, end: 0 });
});

test('dense chart sampling stays bounded and preserves endpoints and isolated extrema', () => {
  const values = Array(9000).fill(10);
  values[1234] = 900;
  values[6789] = 0;
  const samples = chartSamples(values);
  assert.ok(samples.length <= 800);
  assert.deepEqual(samples[0], { index: 0, value: 10 });
  assert.deepEqual(samples.at(-1), { index: 8999, value: 10 });
  assert.ok(samples.some(sample => sample.index === 1234 && sample.value === 900));
  assert.ok(samples.some(sample => sample.index === 6789 && sample.value === 0));
  assert.ok(samples.every((sample, index) => index === 0 || sample.index > samples[index - 1].index));
  assert.deepEqual(chartSamples([1, 0, 3]), [{ index: 0, value: 1 }, { index: 1, value: 0 }, { index: 2, value: 3 }]);
});

test('detail pages omit zero usage and preserve original series indexes', () => {
  const periods = Array.from({ length: 305 }, (_, index) => row({
    inputTokens: index % 2 === 0 ? index + 1 : 0,
    outputTokens: 0, cachedTokens: 0, reasoningTokens: index === 1 ? 7 : 0, cost: index === 3 ? 0.5 : 0,
  }));
  const first = detailPage(periods, 'total', 0);
  const second = detailPage(periods, 'total', 1);
  assert.equal(first.total, 153);
  assert.equal(first.bounds.count, 2);
  assert.equal(first.indexes.length, 100);
  assert.equal(second.indexes.length, 53);
  assert.deepEqual([...first.indexes, ...second.indexes], Array.from({ length: 153 }, (_, index) => 304 - index * 2));
  assert.deepEqual(detailPage(periods, 'reasoningTokens', 0).indexes, [1]);
  assert.deepEqual(detailPage(periods, 'cost', 0).indexes, [3]);
  const empty = detailPage(periods, 'outputTokens', 99);
  assert.equal(empty.total, 0);
  assert.equal(empty.bounds.page, 0);
  assert.deepEqual(empty.indexes, []);
});
