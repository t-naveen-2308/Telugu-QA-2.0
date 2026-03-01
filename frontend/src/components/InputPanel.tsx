import { useState, useRef } from 'react';
import { Upload, Search, Loader2, FileText, HelpCircle, Languages } from 'lucide-react';
import { api } from '../api';

interface Props {
  context: string;
  question: string;
  onContextChange: (value: string) => void;
  onQuestionChange: (value: string) => void;
  onContextFocus: () => void;
  onQuestionFocus: () => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit: () => void;
  loading: boolean;
}

/**
 * Detect if text contains Latin characters that could be ITRANS transliteration.
 */
function hasLatinChars(text: string): boolean {
  return /[a-zA-Z]{2,}/.test(text);
}

export default function InputPanel({
  context,
  question,
  onContextChange,
  onQuestionChange,
  onContextFocus,
  onQuestionFocus,
  onFileUpload,
  onSubmit,
  loading
}: Props) {
  const [activeWord, setActiveWord] = useState<{
    word: string;
    start: number;
    end: number;
    options: string[];
    field: 'context' | 'question';
  } | null>(null);

  const [transliterating, setTransliterating] = useState<'context' | 'question' | null>(null);
  const debounceTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  // Auto-transliterate handler with word extraction
  const handleTextChange = (
    field: 'context' | 'question',
    e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>,
    setter: (v: string) => void
  ) => {
    const value = e.target.value;
    const cursor = e.target.selectionStart || 0;
    setter(value);

    // Find boundaries of the current word
    let start = cursor;
    while (start > 0 && !/\s/.test(value[start - 1])) start--;
    let end = cursor;
    while (end < value.length && !/\s/.test(value[end])) end++;

    const word = value.slice(start, end);

    if (debounceTimers.current[field]) {
      clearTimeout(debounceTimers.current[field]);
    }

    if (/^[a-zA-Z]+$/.test(word) && word.length >= 2) {
      debounceTimers.current[field] = setTimeout(async () => {
        try {
          const result = await api.transliterate(word);
          if (result.options && result.options.length > 0) {
            setActiveWord({ word, start, end, field, options: result.options });
          } else {
            setActiveWord(null);
          }
        } catch {
          setActiveWord(null);
        }
      }, 400);
    } else {
      setActiveWord(null);
    }
  };

  const applyTransliteration = (
    selectedWord: string,
    currentValue: string,
    setter: (v: string) => void
  ) => {
    if (!activeWord) return;
    const { start, end } = activeWord;
    const newValue = currentValue.slice(0, start) + selectedWord + currentValue.slice(end);
    setter(newValue);
    setActiveWord(null);
  };

  const handleKeyDown = (
    field: 'context' | 'question',
    e: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>
  ) => {
    if (e.key === 'Enter' && field === 'question') {
      onSubmit();
    }
  };

  // Manual transliterate button handler
  const handleTransliterateField = async (
    field: 'context' | 'question',
    value: string,
    setter: (v: string) => void
  ) => {
    if (!value.trim() || !hasLatinChars(value)) return;
    setTransliterating(field);
    try {
      const words = value.split(/(\s+)/);
      const converted = await Promise.all(
        words.map(async (word) => {
          if (/^[a-zA-Z]+$/.test(word.trim()) && word.trim().length >= 2) {
            try {
              const result = await api.transliterate(word.trim());
              return result.options && result.options.length > 0 ? result.options[0] : word;
            } catch {
              return word;
            }
          }
          return word;
        })
      );
      setter(converted.join(''));
    } catch {
      // ignore
    } finally {
      setTransliterating(null);
    }
  };

  return (
    <div className="space-y-5">
      {/* Context Input */}
      <div>
        <div className="flex items-center justify-between mb-2.5">
          <label className="font-medium text-[var(--text-primary)] flex items-center gap-2 text-sm">
            <span className="w-6 h-6 rounded-md bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
              <FileText className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
            </span>
            Context <span className="font-telugu text-[var(--text-muted)]">(సందర్భం)</span>
          </label>
          <div className="flex items-center gap-2">
            {hasLatinChars(context) && (
              <button
                onClick={() => handleTransliterateField('context', context, onContextChange)}
                disabled={transliterating === 'context'}
                className="text-xs text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 flex items-center gap-1 px-2 py-1.5 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all font-medium"
                title="Transliterate English → Telugu"
              >
                <Languages size={13} />
                {transliterating === 'context' ? 'Converting...' : 'Transliterate'}
              </button>
            )}
            <label className="cursor-pointer text-xs text-[var(--accent)] hover:text-[var(--accent-hover)] flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-all font-medium">
              <Upload size={13} />
              Upload .txt
              <input
                type="file"
                accept=".txt"
                onChange={onFileUpload}
                className="hidden"
              />
            </label>
          </div>
        </div>
        <div className="relative">
          <textarea
            value={context}
            onChange={(e) => handleTextChange('context', e, onContextChange)}
            onKeyDown={(e) => handleKeyDown('context', e)}
            onFocus={() => { onContextFocus(); setActiveWord(null); }}
            onBlur={async () => {
              if (activeWord && activeWord.options.length > 0) {
                applyTransliteration(activeWord.options[0], context, onContextChange);
              }
              // Transliterate remaining latin words
              if (hasLatinChars(context)) {
                handleTransliterateField('context', context, onContextChange);
              }
            }}
            placeholder="తెలుగులో టెక్స్ట్ ఇక్కడ పేస్ట్ చేయండి... (or type in English to get suggestions)"
            className="w-full h-36 p-4 rounded-lg font-telugu text-lg resize-none bg-[var(--bg-elevated)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)] transition-all"
          />
          {activeWord && activeWord.field === 'context' && (
            <div className="absolute top-100 mt-2 z-20 left-0 bg-[var(--bg-surface)] border border-[var(--border-color)] shadow-lg shadow-amber-500/10 rounded-lg overflow-hidden flex flex-wrap animate-fadeUp">
              {activeWord.options.map((opt, i) => (
                <button
                  key={i}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    applyTransliteration(opt + ' ', context, onContextChange);
                  }}
                  className={`px-4 py-2.5 font-telugu text-lg hover:bg-[var(--bg-hover)] transition-colors border-r last:border-r-0 border-[var(--border-subtle)] ${i === 0 ? 'bg-amber-50 dark:bg-amber-900/20 text-[var(--accent)] font-medium' : 'text-[var(--text-primary)]'}`}
                >
                  {opt}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Question Input */}
      <div>
        <div className="flex items-center justify-between mb-2.5">
          <label className="font-medium text-[var(--text-primary)] flex items-center gap-2 text-sm">
            <span className="w-6 h-6 rounded-md bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
              <HelpCircle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
            </span>
            Question <span className="font-telugu text-[var(--text-muted)]">(ప్రశ్న)</span>
          </label>
          {hasLatinChars(question) && (
            <button
              onClick={() => handleTransliterateField('question', question, onQuestionChange)}
              disabled={transliterating === 'question'}
              className="text-xs text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 flex items-center gap-1 px-2 py-1.5 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all font-medium"
              title="Transliterate English → Telugu"
            >
              <Languages size={13} />
              {transliterating === 'question' ? 'Converting...' : 'Transliterate'}
            </button>
          )}
        </div>
        <div className="relative">
          <input
            type="text"
            value={question}
            onChange={(e) => handleTextChange('question', e, onQuestionChange)}
            onKeyDown={(e) => handleKeyDown('question', e)}
            onFocus={() => { onQuestionFocus(); setActiveWord(null); }}
            onBlur={async () => {
              if (activeWord && activeWord.options.length > 0) {
                applyTransliteration(activeWord.options[0], question, onQuestionChange);
              }
              // Transliterate remaining latin words
              if (hasLatinChars(question)) {
                handleTransliterateField('question', question, onQuestionChange);
              }
            }}
            placeholder="తెలుగులో ప్రశ్న అడగండి... (or type in English to get suggestions)"
            className="w-full p-4 rounded-lg font-telugu text-lg bg-[var(--bg-elevated)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)] transition-all"
          />
          {activeWord && activeWord.field === 'question' && (
            <div className="absolute top-full mt-2 z-20 left-0 bg-[var(--bg-surface)] border border-[var(--border-color)] shadow-lg shadow-amber-500/10 rounded-lg overflow-hidden flex flex-wrap animate-fadeUp">
              {activeWord.options.map((opt, i) => (
                <button
                  key={i}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    applyTransliteration(opt + ' ', question, onQuestionChange);
                  }}
                  className={`px-4 py-2.5 font-telugu text-lg hover:bg-[var(--bg-hover)] transition-colors border-r last:border-r-0 border-[var(--border-subtle)] ${i === 0 ? 'bg-amber-50 dark:bg-amber-900/20 text-[var(--accent)] font-medium' : 'text-[var(--text-primary)]'}`}
                >
                  {opt}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Submit Button */}
      <button
        onClick={onSubmit}
        disabled={loading || !context.trim() || !question.trim()}
        className="btn btn-accent w-full shadow-md hover:shadow-lg transition-all"
      >
        {loading ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Finding answer...
          </>
        ) : (
          <>
            <Search size={18} />
            Get Answer / సమాధానం పొందండి
          </>
        )}
      </button>
    </div>
  );
}
