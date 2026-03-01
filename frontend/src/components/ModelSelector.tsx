import type { ModelInfo } from '../types';
import { Check, Cpu } from 'lucide-react';

interface Props {
  models: ModelInfo[];
  selectedModel: string;
  onSelectModel: (key: string) => void;
}

export default function ModelSelector({ models, selectedModel, onSelectModel }: Props) {
  return (
    <div className="surface p-5">
      <h3 className="font-semibold text-[var(--text-primary)] mb-4 flex items-center gap-2.5 text-sm">
        <span className="w-8 h-8 rounded-lg bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
          <Cpu className="w-4 h-4 text-amber-600 dark:text-amber-400" />
        </span>
        Model Selection
      </h3>
      <div className="space-y-2">
        {models.map(model => (
          <button
            key={model.key}
            onClick={() => onSelectModel(model.key)}
            className={`w-full text-left p-3.5 rounded-lg border transition-all duration-200 ${selectedModel === model.key
                ? 'border-[var(--accent)] bg-amber-50 dark:bg-amber-900/20 ring-1 ring-[var(--accent)]/20'
                : 'border-[var(--border-color)] hover:border-[var(--accent)]/50 bg-[var(--bg-surface)] hover:bg-[var(--bg-elevated)]'
              }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-medium text-[var(--text-primary)] text-sm">{model.name}</span>
                {model.recommended && (
                  <span className="text-xs bg-gradient-to-r from-amber-500 to-orange-500 text-white px-2 py-0.5 rounded-full font-medium">
                    Best
                  </span>
                )}
                {model.key.includes('-domain') && (
                  <span className="text-xs bg-gradient-to-r from-purple-500 to-indigo-500 text-white px-2 py-0.5 rounded-full font-medium">
                    LoRA
                  </span>
                )}
              </div>
              {selectedModel === model.key && (
                <div className="w-5 h-5 rounded-full bg-[var(--accent)] flex items-center justify-center">
                  <Check size={12} className="text-white" />
                </div>
              )}
            </div>
            <p className="text-xs text-[var(--text-muted)] mt-1.5">{model.description}</p>
            <div className="flex gap-4 mt-2 text-xs">
              <span className="text-blue-600 dark:text-blue-400 font-medium">EM: {model.exact_match.toFixed(1)}%</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-medium">F1: {model.f1_score.toFixed(1)}%</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
