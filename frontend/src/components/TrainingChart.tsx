import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import type { TrainingData } from '../types';
import { TrendingUp, Award } from 'lucide-react';

interface Props {
  data: TrainingData;
}

type ChartType = 'training' | 'validation';

export default function TrainingChart({ data }: Props) {
  const [chartType, setChartType] = useState<ChartType>('validation');

  const colors: Record<string, string> = {
    muril: '#f59e0b',
    xlmr: '#ec4899',
    indicbert: '#3b82f6',
    mbert: '#22c55e',
    muril_domain: '#f97316',
    mbert_domain: '#10b981'
  };

  // Transform data for recharts
  const transformData = (lossType: 'training' | 'validation') => {
    const allSteps = new Set<number>();

    Object.entries(data).forEach(([, model]) => {
      if (model) model.steps.forEach((step: number) => allSteps.add(step));
    });

    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    return sortedSteps.map(step => {
      const point: Record<string, number | null> = { step };

      (Object.keys(data) as Array<keyof TrainingData>).forEach(modelKey => {
        const model = data[modelKey];
        if (!model) return;
        const idx = model.steps.indexOf(step);
        const losses = lossType === 'training' ? model.training_loss : model.validation_loss;
        point[modelKey] = idx >= 0 ? losses[idx] : null;
      });

      return point;
    });
  };

  const chartData = transformData(chartType === 'training' ? 'training' : 'validation');

  return (
    <div className="space-y-6">
      <div className="surface p-6">
      <div className="flex items-center justify-between mb-5">
        <h3 className="font-semibold text-[var(--text-primary)] flex items-center gap-2.5 text-sm">
          <span className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
            <TrendingUp className="w-4 h-4 text-purple-600 dark:text-purple-400" />
          </span>
          {chartType === 'training' ? 'Training' : 'Validation'} Loss Comparison
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setChartType('training')}
            className={`px-3.5 py-1.5 text-sm rounded-md font-medium transition-all duration-200 ${chartType === 'training'
                ? 'bg-[var(--accent)] text-white shadow-sm'
                : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] border border-[var(--border-color)]'
              }`}
          >
            Training
          </button>
          <button
            onClick={() => setChartType('validation')}
            className={`px-3.5 py-1.5 text-sm rounded-md font-medium transition-all duration-200 ${chartType === 'validation'
                ? 'bg-[var(--accent)] text-white shadow-sm'
                : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] border border-[var(--border-color)]'
              }`}
          >
            Validation
          </button>
        </div>
      </div>

      <div className="h-72 bg-[var(--bg-elevated)] rounded-lg p-4 border border-[var(--border-color)]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
            <XAxis
              dataKey="step"
              stroke="var(--text-muted)"
              fontSize={11}
              tickFormatter={(value) => `${value / 1000}k`}
            />
            <YAxis
              stroke="var(--text-muted)"
              fontSize={11}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--bg-surface)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                color: 'var(--text-primary)',
                fontSize: '12px'
              }}
              labelStyle={{ color: 'var(--text-primary)' }}
              labelFormatter={(value) => `Step ${value}`}
              formatter={(value: number, name: string) => [value?.toFixed(3) || '-', name.toUpperCase()]}
            />
            <Legend wrapperStyle={{ color: 'var(--text-secondary)', fontSize: '12px' }} />

            <Line
              type="monotone"
              dataKey="muril"
              stroke={colors.muril}
              strokeWidth={2.5}
              dot={{ r: 3, fill: colors.muril }}
              connectNulls
              name="MuRIL"
            />
            <Line
              type="monotone"
              dataKey="mbert"
              stroke={colors.mbert}
              strokeWidth={2.5}
              dot={{ r: 3, fill: colors.mbert }}
              connectNulls
              name="mBERT"
            />
            <Line
              type="monotone"
              dataKey="xlmr"
              stroke={colors.xlmr}
              strokeWidth={2.5}
              dot={{ r: 3, fill: colors.xlmr }}
              connectNulls
              name="XLM-R"
            />
            <Line
              type="monotone"
              dataKey="indicbert"
              stroke={colors.indicbert}
              strokeWidth={2.5}
              dot={{ r: 3, fill: colors.indicbert }}
              connectNulls
              name="IndicBERT"
            />
            {data.muril_domain && (
              <Line
                type="monotone"
                dataKey="muril_domain"
                stroke={colors.muril_domain}
                strokeWidth={2.5}
                strokeDasharray="5 3"
                dot={{ r: 3, fill: colors.muril_domain }}
                connectNulls
                name="MuRIL-Domain (LoRA)"
              />
            )}
            {data.mbert_domain && (
              <Line
                type="monotone"
                dataKey="mbert_domain"
                stroke={colors.mbert_domain}
                strokeWidth={2.5}
                strokeDasharray="5 3"
                dot={{ r: 3, fill: colors.mbert_domain }}
                connectNulls
                name="mBERT-Domain (LoRA)"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      </div>

      {/* Summary Stats */}
      <div className="surface p-6">
        <h3 className="font-semibold text-[var(--text-primary)] flex items-center gap-2.5 text-sm mb-5">
          <span className="w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
            <Award className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          </span>
          Overall Model Performance
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {[
          { key: 'muril', name: 'MuRIL', em: 68.5, f1: 84.0, color: colors.muril, best: data.muril.best_step, rank: 1 },
          { key: 'mbert', name: 'mBERT', em: 61.1, f1: 77.2, color: colors.mbert, best: data.mbert.best_step, rank: 2 },
          { key: 'xlmr', name: 'XLM-R', em: 61.0, f1: 77.2, color: colors.xlmr, best: data.xlmr.best_step, rank: 3 },
          { key: 'indicbert', name: 'IndicBERT', em: 9.8, f1: 35.8, color: colors.indicbert, best: data.indicbert.best_step, rank: 4 },
          ...(data.muril_domain ? [{ key: 'muril_domain', name: 'MuRIL-Domain', em: 63.0, f1: 66.8, color: colors.muril_domain, best: data.muril_domain.best_step, rank: 0, isDomain: true }] : []),
          ...(data.mbert_domain ? [{ key: 'mbert_domain', name: 'mBERT-Domain', em: 56.0, f1: 73.2, color: colors.mbert_domain, best: data.mbert_domain.best_step, rank: 0, isDomain: true }] : []),
        ].map((model) => (
          <div
            key={model.key}
            className="p-3.5 rounded-lg border bg-[var(--bg-elevated)]"
            style={{ borderColor: model.color + '40' }}
          >
            <div className="flex items-center gap-1.5 mb-1.5">
              {model.rank === 1 && <span className="text-sm">🥇</span>}
              {model.rank === 2 && <span className="text-sm">🥈</span>}
              {model.rank === 3 && <span className="text-sm">🥉</span>}
              {'isDomain' in model && <span className="text-sm">🎯</span>}
              <span className="font-semibold text-sm" style={{ color: model.color }}>{model.name}</span>
              {'isDomain' in model && (
                <span className="text-[10px] bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 px-1.5 py-0.5 rounded-full font-medium">LoRA</span>
              )}
            </div>
            <div className="text-xs text-[var(--text-secondary)] space-y-0.5">
              {'isDomain' in model ? (
                <p className="font-medium">Domain EM: {model.em}% / F1: {model.f1}%</p>
              ) : (
                <p className="font-medium">EM: {model.em}% / F1: {model.f1}%</p>
              )}
              <p className="text-[var(--text-muted)]">Best @ step {model.best.toLocaleString()}</p>
            </div>
          </div>
        ))}
        </div>
      </div>
    </div>
  );
}
