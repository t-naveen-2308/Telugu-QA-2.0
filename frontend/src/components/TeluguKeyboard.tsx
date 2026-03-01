import { useState, useEffect } from 'react';
import {
  TELUGU_VOWELS,
  TELUGU_CONSONANTS,
  TELUGU_VOWEL_MARKS,
  TELUGU_SPECIAL
} from '../types';
import { api } from '../api';
import { RefreshCw, Space } from 'lucide-react';

interface Props {
  onInsert: (char: string) => void;
  activeField: 'context' | 'question';
  onFieldChange: (field: 'context' | 'question') => void;
}

type TabType = 'vowels' | 'consonants' | 'marks' | 'numbers' | 'special' | 'transliterate';

export default function TeluguKeyboard({ onInsert, activeField, onFieldChange }: Props) {
  const [activeTab, setActiveTab] = useState<TabType>('consonants');
  const [translitInput, setTranslitInput] = useState('');
  const [translitOutput, setTranslitOutput] = useState<string[]>([]);
  const [translitLoading, setTranslitLoading] = useState(false);

  const handleTransliterate = async (input: string) => {
    if (!input.trim()) return;
    setTranslitLoading(true);
    try {
      const result = await api.transliterate(input);
      setTranslitOutput(result.options || []);
    } catch (err) {
      console.error('Transliteration error:', err);
    } finally {
      setTranslitLoading(false);
    }
  };

  const handleInsertTranslit = (text: string) => {
    if (text) {
      onInsert(text);
      setTranslitInput('');
      setTranslitOutput([]);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      if (translitInput.trim().length >= 2) {
        handleTransliterate(translitInput);
      } else {
        setTranslitOutput([]);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [translitInput]);

  const tabs = [
    { key: 'consonants' as TabType, label: 'హల్లులు', desc: 'Consonants' },
    { key: 'vowels' as TabType, label: 'అచ్చులు', desc: 'Vowels' },
    { key: 'marks' as TabType, label: 'గుణింతం', desc: 'Marks' },
    { key: 'special' as TabType, label: 'గుర్తులు', desc: 'Special' },
    { key: 'transliterate' as TabType, label: 'A→అ', desc: 'Translit' },
  ];

  const renderCharGrid = (chars: { char: string; name: string }[], cols: number = 5) => (
    <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {chars.map((item, idx) => (
        <button
          key={idx}
          onClick={() => onInsert(item.char)}
          title={item.name}
          className="telugu-key"
        >
          {item.char === ' ' ? '␣' : item.char}
        </button>
      ))}
    </div>
  );

  return (
    <div className="p-5">
      {/* Active Field Indicator */}
      <div className="flex items-center gap-4 mb-4">
        <span className="text-xs text-[var(--text-muted)] font-medium uppercase tracking-wide">Insert to:</span>
        <div className="flex gap-2">
          <button
            onClick={() => onFieldChange('context')}
            className={`px-3.5 py-1.5 text-sm rounded-lg font-medium transition-all duration-200 ${activeField === 'context'
              ? 'bg-[var(--accent)] text-white shadow-sm'
              : 'bg-[var(--bg-surface)] border border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]'
              }`}
          >
            Context
          </button>
          <button
            onClick={() => onFieldChange('question')}
            className={`px-3.5 py-1.5 text-sm rounded-lg font-medium transition-all duration-200 ${activeField === 'question'
              ? 'bg-[var(--accent)] text-white shadow-sm'
              : 'bg-[var(--bg-surface)] border border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]'
              }`}
          >
            Question
          </button>
        </div>

        {/* Quick actions */}
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => onInsert(' ')}
            className="px-3 py-1.5 bg-[var(--bg-surface)] border border-[var(--border-color)] rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] text-xs flex items-center gap-1 transition-all shadow-sm"
            title="Insert Space"
          >
            <Space size={12} />
            Space
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 overflow-x-auto pb-1">
        {tabs.map(tab => {
          const isTranslit = tab.key === 'transliterate';
          const isActive = activeTab === tab.key;

          let tabClass = 'px-3.5 py-2 text-sm rounded-lg whitespace-nowrap transition-all duration-200 ';

          if (isActive) {
            tabClass += isTranslit
              ? 'bg-gradient-to-r from-purple-500 to-indigo-500 text-white font-medium shadow-md'
              : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 font-medium';
          } else {
            tabClass += isTranslit
              ? 'bg-purple-50 dark:bg-purple-900/10 text-purple-700 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/20 border border-purple-200 dark:border-purple-800'
              : 'bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]';
          }

          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={tabClass}
            >
              <span className={isTranslit ? '' : 'font-telugu'}>{tab.label}</span>
              <span className={`text-xs ml-1 ${isActive && isTranslit ? 'text-purple-100' : 'text-[var(--text-muted)]'}`}>({tab.desc})</span>
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="bg-[var(--bg-surface)] rounded-lg p-4 border border-[var(--border-color)] min-h-[140px]">
        {activeTab === 'vowels' && renderCharGrid(TELUGU_VOWELS, 8)}
        {activeTab === 'consonants' && renderCharGrid(TELUGU_CONSONANTS, 6)}
        {activeTab === 'marks' && renderCharGrid(TELUGU_VOWEL_MARKS, 8)}
        {activeTab === 'special' && renderCharGrid(TELUGU_SPECIAL, 5)}

        {activeTab === 'transliterate' && (
          <div className="space-y-4">
            <p className="text-sm text-[var(--text-muted)]">
              Start typing English to see Telugu suggestions
            </p>
            <div className="relative">
              <input
                type="text"
                value={translitInput}
                onChange={(e) => setTranslitInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && translitOutput.length > 0) {
                    handleInsertTranslit(translitOutput[0]);
                  }
                }}
                placeholder="Type here..."
                className="w-full px-4 py-3 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)]"
              />

              {translitLoading && (
                <div className="absolute right-4 top-1/2 -translate-y-1/2">
                  <RefreshCw size={16} className="animate-spin text-[var(--text-muted)]" />
                </div>
              )}

              {translitOutput.length > 0 && translitInput.trim().length > 0 && (
                <div className="absolute top-full left-0 mt-2 w-full bg-[var(--bg-surface)] border border-[var(--border-color)] rounded-lg shadow-lg overflow-hidden z-20 animate-fadeUp">
                  {translitOutput.map((opt, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleInsertTranslit(opt)}
                      className="w-full text-left px-5 py-3 hover:bg-[var(--bg-hover)] flex items-center justify-between border-b last:border-b-0 border-[var(--border-subtle)] transition-colors group"
                    >
                      <span className="font-telugu text-lg group-hover:text-[var(--accent)]">{opt}</span>
                      <span className="text-xs text-[var(--text-muted)] group-hover:text-[var(--accent)] font-medium">Select ↵</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
