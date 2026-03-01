import { useState, useMemo } from 'react';
import type { QAResponse } from '../types';
import { Volume2, ChevronDown, ChevronUp, Loader2, Bot, Wand2 } from 'lucide-react';

interface Props {
  answer: QAResponse;
  context: string;
  audioUrl: string | null;
  ttsLoading?: boolean;
}

export default function AnswerDisplay({ answer, context, audioUrl, ttsLoading = false }: Props) {
  const [showContext, setShowContext] = useState(false);

  const getConfidenceColor = (score: number) => {
    if (score >= 0.7) return 'text-emerald-600 dark:text-emerald-400';
    if (score >= 0.4) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getConfidenceBg = (score: number) => {
    if (score >= 0.7) return 'from-emerald-500 to-green-500';
    if (score >= 0.4) return 'from-amber-500 to-orange-500';
    return 'from-red-500 to-rose-500';
  };

  const highlightedContext = useMemo(() => {
    const before = context.slice(0, answer.start);
    const highlighted = context.slice(answer.start, answer.end);
    const after = context.slice(answer.end);
    return { before, highlighted, after };
  }, [context, answer.start, answer.end]);

  const confidencePercent = (answer.confidence * 100).toFixed(1);
  const originalConfidencePercent = (answer.original_confidence * 100).toFixed(1);

  if (answer.is_relevant === false) {
    return (
      <div className="surface overflow-hidden">
        <div className="p-6">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-lg p-6 flex items-start gap-4">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-800 dark:text-red-300 mb-2">
                ప్రశ్న సందర్భానికి సంబంధించినది కాదు
              </h3>
              <p className="text-red-700 dark:text-red-400/80 leading-relaxed font-telugu text-[var(--text-primary)]">
                మీరు అడిగిన ప్రశ్న ఈ సందర్భానికి సంబంధించినది కాదు. దయచేసి ప్రశ్నను మార్చండి లేదా వేరే సందర్భం ఇవ్వండి.
              </p>

              <div className="mt-4 flex items-center gap-4">
                {ttsLoading ? (
                  <span className="flex items-center gap-1.5 text-red-600/70 dark:text-red-400/70 text-sm">
                    <Loader2 size={14} className="animate-spin" />
                    Loading audio...
                  </span>
                ) : audioUrl ? (
                  <button
                    onClick={() => {
                      const audio = new Audio(audioUrl);
                      audio.play();
                    }}
                    className="flex items-center gap-1.5 text-red-700 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 text-sm font-medium px-3 py-2 rounded-md bg-red-100/50 dark:bg-red-900/40 hover:bg-red-200/50 dark:hover:bg-red-800/50 transition-colors"
                  >
                    <Volume2 size={16} />
                    ప్లే చేయండి (Play)
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="surface overflow-hidden">
      <div className="p-6 space-y-5">
        {/* Raw Model Output */}
        <div>
          <h4 className="text-xs font-medium text-[var(--text-secondary)] mb-2.5 flex items-center gap-2 uppercase tracking-wide">
            <span className="w-5 h-5 rounded-md bg-stone-200 dark:bg-stone-700 flex items-center justify-center">
              <Bot className="w-3 h-3 text-stone-600 dark:text-stone-300" />
            </span>
            Raw Model Output
          </h4>
          <div className="bg-stone-800 dark:bg-stone-900 rounded-lg p-4 border border-stone-700/50">
            <p className="font-telugu text-xl text-stone-200">{answer.original_answer}</p>
            <div className="mt-2.5 flex items-center gap-2 text-sm">
              <span className="text-stone-400">Confidence:</span>
              <span className={`font-semibold ${answer.original_confidence >= 0.7 ? 'text-emerald-400' : answer.original_confidence >= 0.4 ? 'text-amber-400' : 'text-red-400'}`}>
                {originalConfidencePercent}%
              </span>
            </div>
          </div>
        </div>

        {/* Refined Answer */}
        <div>
          <h4 className="text-xs font-medium text-[var(--text-secondary)] mb-2.5 flex items-center gap-2 uppercase tracking-wide">
            <span className="w-5 h-5 rounded-md bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center">
              <Wand2 className="w-3 h-3 text-white" />
            </span>
            Morphology Refined Answer
          </h4>
          <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/20 rounded-lg p-4 border border-amber-200 dark:border-amber-800/40">
            <p className="font-telugu text-2xl text-amber-800 dark:text-amber-200 font-medium">{answer.answer}</p>
            <div className="mt-3.5 flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-[var(--text-secondary)] text-sm">Confidence:</span>
                <span className={`font-bold text-lg ${getConfidenceColor(answer.confidence)}`}>
                  {confidencePercent}%
                </span>
                {answer.confidence_improvement > 0 && (
                  <span className="text-emerald-600 dark:text-emerald-400 text-xs bg-emerald-100 dark:bg-emerald-900/30 px-2 py-0.5 rounded-full font-medium">
                    ↑{(answer.confidence_improvement * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              {ttsLoading ? (
                <span className="flex items-center gap-1.5 text-[var(--text-muted)] text-sm">
                  <Loader2 size={14} className="animate-spin" />
                  Loading audio...
                </span>
              ) : audioUrl ? (
                <button
                  onClick={() => {
                    const audio = new Audio(audioUrl);
                    audio.play();
                  }}
                  className="flex items-center gap-1.5 text-[var(--accent)] hover:text-[var(--accent-hover)] text-sm font-medium px-2.5 py-1.5 rounded-md hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors"
                >
                  <Volume2 size={14} />
                  Play Audio
                </button>
              ) : null}
            </div>

            {answer.refinement_applied && answer.removed_suffixes.length > 0 && (
              <div className="mt-3.5 text-sm text-[var(--text-secondary)]">
                <span className="font-medium">Removed suffixes:</span>{' '}
                {answer.removed_suffixes.map((suffix, i) => (
                  <code key={i} className="bg-stone-200 dark:bg-stone-700 px-1.5 py-0.5 rounded mx-0.5 font-telugu text-[var(--text-primary)] text-xs">
                    {suffix}
                  </code>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Confidence Meter */}
        <div>
          <div className="flex justify-between text-xs text-[var(--text-secondary)] mb-1.5">
            <span>Confidence</span>
            <span className="font-medium">{confidencePercent}%</span>
          </div>
          <div className="w-full h-1.5 bg-stone-200 dark:bg-stone-700 rounded-full overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r ${getConfidenceBg(answer.confidence)} transition-all duration-500`}
              style={{ width: `${answer.confidence * 100}%` }}
            />
          </div>
        </div>

        {/* Show in Context */}
        <div>
          <button
            onClick={() => setShowContext(!showContext)}
            className="flex items-center gap-1.5 text-[var(--accent)] hover:text-[var(--accent-hover)] text-sm font-medium px-2.5 py-1.5 -ml-2.5 rounded-md hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors"
          >
            {showContext ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {showContext ? 'Hide' : 'Show'} answer in context
          </button>

          {showContext && (
            <div className="mt-3 p-4 bg-[var(--bg-elevated)] rounded-lg font-telugu text-lg leading-relaxed text-[var(--text-secondary)] border border-[var(--border-color)]">
              <span>{highlightedContext.before}</span>
              <mark className="answer-highlight font-medium text-[var(--text-primary)]">{highlightedContext.highlighted}</mark>
              <span>{highlightedContext.after}</span>
            </div>
          )}
        </div>
      </div>

      {/* Audio Player */}
      {audioUrl && (
        <div className="bg-[var(--bg-elevated)] px-6 py-4 border-t border-[var(--border-color)]">
          <audio controls src={audioUrl} className="w-full h-10">
            Your browser does not support audio playback.
          </audio>
        </div>
      )}
    </div>
  );
}
