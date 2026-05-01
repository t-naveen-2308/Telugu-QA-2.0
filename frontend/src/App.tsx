import { useState, useEffect } from 'react';
import { api } from './api';
import type { ModelInfo, QAResponse, Examples, TrainingData } from './types';
import ModelSelector from './components/ModelSelector';
import InputPanel from './components/InputPanel';
import AnswerDisplay from './components/AnswerDisplay';
import TrainingChart from './components/TrainingChart';
import TeluguKeyboard from './components/TeluguKeyboard';
import VoiceInput from './components/VoiceInput';
import { FileText, Keyboard, Moon, Sun, Sparkles, Zap } from 'lucide-react';

type InputMode = 'text' | 'voice';

function App() {
  // Dark mode state
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('darkMode') === 'true' ||
        window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  // Toggle dark mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', String(darkMode));
  }, [darkMode]);

  // State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState('muril');
  const [examples, setExamples] = useState<Examples>({});
  const [selectedExample, setSelectedExample] = useState<string>('');
  const [trainingData, setTrainingData] = useState<TrainingData | null>(null);

  const [context, setContext] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [showKeyboard, setShowKeyboard] = useState(false);
  const [activeField, setActiveField] = useState<'context' | 'question'>('question');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [preloadedIrrelevantAudioUrl, setPreloadedIrrelevantAudioUrl] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        const [modelsData, examplesData, trainingDataResult] = await Promise.all([
          api.getModels(),
          api.getExamples(),
          api.getTrainingData()
        ]);
        setModels(modelsData);
        setExamples(examplesData);
        setTrainingData(trainingDataResult);
      } catch (err) {
        console.error('Failed to load initial data:', err);
      }
    };
    loadData();

    // Preload constant irrelevant answer TTS
    const irrelevantText = "మీరు అడిగిన ప్రశ్న ఈ సందర్భానికి సంబంధించినది కాదు. దయచేసి ప్రశ్నను మార్చండి లేదా వేరే సందర్భం ఇవ్వండి.";
    api.getTTS(irrelevantText)
      .then(audioBlob => {
        setPreloadedIrrelevantAudioUrl(URL.createObjectURL(audioBlob));
      })
      .catch(err => console.warn('Failed to preload irrelevant TTS audio:', err));
  }, []);

  // Load example when selected
  useEffect(() => {
    if (selectedExample && examples[selectedExample]) {
      setContext(examples[selectedExample].context);
      setQuestion(examples[selectedExample].question);
    }
  }, [selectedExample, examples]);

  // Get answer
  const handleGetAnswer = async () => {
    if (!context.trim() || !question.trim()) {
      setError('Please provide both context and question');
      return;
    }

    setLoading(true);
    setError(null);
    setAnswer(null);
    setAudioUrl(null);

    try {
      const result = await api.predict({
        context,
        question,
        model_key: selectedModel
      });
      setAnswer(result);
      setLoading(false);

      // Generate TTS in background (non-blocking)
      if (result.is_relevant === false && preloadedIrrelevantAudioUrl) {
        setAudioUrl(preloadedIrrelevantAudioUrl);
      } else {
        setTtsLoading(true);
        const ttsText = result.is_relevant === false
          ? "మీరు అడిగిన ప్రశ్న ఈ సందర్భానికి సంబంధించినది కాదు. దయచేసి ప్రశ్నను మార్చండి లేదా వేరే సందర్భం ఇవ్వండి."
          : result.answer;

        api.getTTS(ttsText)
          .then(audioBlob => {
            const url = URL.createObjectURL(audioBlob);
            setAudioUrl(url);
          })
          .catch(err => console.warn('TTS not available:', err))
          .finally(() => setTtsLoading(false));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get answer');
      setLoading(false);
    }
  };

  // Handle voice transcription
  const handleVoiceTranscription = async (text: string) => {
    setQuestion(text);
    if (context.trim()) {
      setTimeout(() => handleGetAnswer(), 100);
    }
  };

  // Handle keyboard character insert
  const handleKeyboardInsert = (char: string) => {
    if (activeField === 'context') {
      setContext(prev => prev + char);
    } else {
      setQuestion(prev => prev + char);
    }
  };

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        setContext(text);
      };
      reader.readAsText(file);
    }
    // Clear input to allow uploading the same file again
    e.target.value = '';
  };

  const currentModel = models.find(m => m.key === selectedModel);

  return (
    <div className="min-h-screen bg-[var(--bg-base)] transition-colors duration-300 overflow-hidden relative">
      {/* Decorative Background Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-orange-500/10 dark:bg-orange-600/10 blur-[100px] pointer-events-none" />
      <div className="absolute top-[20%] right-[-5%] w-[30%] h-[30%] rounded-full bg-blue-500/5 dark:bg-blue-600/5 blur-[100px] pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[20%] w-[50%] h-[30%] rounded-full bg-amber-500/10 dark:bg-amber-600/10 blur-[120px] pointer-events-none" />

      {/* Header */}
      <header className="bg-[var(--bg-surface)] border-b border-[var(--border-color)] sticky top-0 z-50 backdrop-blur-xl bg-opacity-80">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 dark:from-amber-400 dark:to-orange-500 flex items-center justify-center shadow-lg shadow-amber-500/20 dark:shadow-amber-400/10">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-600 to-amber-600 dark:from-orange-400 dark:to-amber-400 font-telugu tracking-tight drop-shadow-sm">
                  తెలుగు ప్రశ్నోత్తరాలు
                </h1>
                <p className="text-[var(--text-muted)] text-xs mt-0.5 flex items-center gap-1.5">
                  <Zap className="w-3 h-3 text-[var(--accent)]" />
                  AI-Powered Telugu Question Answering
                </p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2.5 rounded-lg bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] border border-[var(--border-color)] transition-all duration-200 text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sidebar */}
          <aside className="lg:col-span-3 space-y-5">
            {/* Model Selector */}
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />

            {/* Model Performance */}
            {currentModel && (
              <div className="surface p-5">
                <h3 className="font-semibold text-[var(--text-primary)] mb-4 flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2.5">
                    <span className="w-8 h-8 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    </span>
                    Performance
                  </div>
                  <span className="text-blue-600 dark:text-blue-400 font-bold text-[11px] bg-blue-100 dark:bg-blue-900/40 px-2 py-1 rounded-md whitespace-nowrap">
                    {currentModel.name}
                  </span>
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg p-4 bg-gradient-to-br from-blue-50 to-blue-100/50 dark:from-blue-950/40 dark:to-blue-900/20 border border-blue-200/50 dark:border-blue-800/30">
                    <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{currentModel.exact_match.toFixed(1)}%</p>
                    <p className="text-xs text-blue-600/70 dark:text-blue-400/60 mt-1 font-medium">Exact Match</p>
                  </div>
                  <div className="rounded-lg p-4 bg-gradient-to-br from-emerald-50 to-emerald-100/50 dark:from-emerald-950/40 dark:to-emerald-900/20 border border-emerald-200/50 dark:border-emerald-800/30">
                    <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{currentModel.f1_score.toFixed(1)}%</p>
                    <p className="text-xs text-emerald-600/70 dark:text-emerald-400/60 mt-1 font-medium">F1 Score</p>
                  </div>
                </div>
              </div>
            )}

            {/* Quick Examples */}
            <div className="surface p-5">
              <h3 className="font-semibold text-[var(--text-primary)] mb-4 flex items-center gap-2.5 text-sm">
                <span className="w-8 h-8 rounded-lg bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
                  <FileText className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                </span>
                Quick Examples
              </h3>
              <select
                value={selectedExample}
                onChange={(e) => setSelectedExample(e.target.value)}
                className="w-full p-3 rounded-lg text-sm bg-[var(--bg-elevated)] border border-[var(--border-color)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)] transition-all"
              >
                <option value="">Custom input</option>
                {Object.keys(examples).map(key => (
                  <option key={key} value={key}>{key}</option>
                ))}
              </select>
            </div>
          </aside>

          {/* Main Content */}
          <main className="lg:col-span-9 space-y-6">
            {/* Input Card */}
            <div className="surface overflow-hidden">
              {/* Input Mode Selection */}
              <div className="flex items-center justify-between border-b border-[var(--border-color)] bg-[var(--bg-elevated)]/50 px-5 py-3">
                <div className="flex items-center gap-3">
                  <label className="text-sm font-medium text-[var(--text-secondary)]">Input Mode:</label>
                  <select
                    value={inputMode}
                    onChange={(e) => setInputMode(e.target.value as InputMode)}
                    className="px-3 py-1.5 rounded-md text-sm bg-[var(--bg-surface)] border border-[var(--border-color)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)] transition-all"
                  >
                    <option value="text">📝 Text Input</option>
                    <option value="voice">🎤 Voice Input</option>
                  </select>
                </div>
                <button
                  onClick={() => setShowKeyboard(!showKeyboard)}
                  className={`py-2 px-3.5 flex items-center gap-2 text-sm font-medium rounded-lg transition-all duration-200 ${showKeyboard
                    ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 border border-amber-300 dark:border-amber-700'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] border border-[var(--border-color)]'
                    }`}
                  title="Telugu Keyboard"
                >
                  <Keyboard size={16} />
                  <span className="hidden sm:inline">Keyboard</span>
                </button>
              </div>

              {/* Telugu Keyboard */}
              {showKeyboard && (
                <div className="border-b border-[var(--border-color)] bg-[var(--bg-elevated)]">
                  <TeluguKeyboard
                    onInsert={handleKeyboardInsert}
                    activeField={activeField}
                    onFieldChange={setActiveField}
                  />
                </div>
              )}

              {/* Input Content */}
              <div className="p-6">
                {inputMode === 'text' ? (
                  <InputPanel
                    context={context}
                    question={question}
                    onContextChange={setContext}
                    onQuestionChange={setQuestion}
                    onContextFocus={() => setActiveField('context')}
                    onQuestionFocus={() => setActiveField('question')}
                    onFileUpload={handleFileUpload}
                    onSubmit={handleGetAnswer}
                    loading={loading}
                  />
                ) : (
                  <VoiceInput
                    context={context}
                    onContextChange={setContext}
                    onTranscription={handleVoiceTranscription}
                    onSubmit={handleGetAnswer}
                    loading={loading}
                    onFileUpload={handleFileUpload}
                  />
                )}
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="rounded-lg p-4 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900/50 text-red-700 dark:text-red-300 text-sm animate-fade-in flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-red-100 dark:bg-red-900/50 flex items-center justify-center flex-shrink-0">
                  <span className="text-red-500">!</span>
                </div>
                {error}
              </div>
            )}

            {/* Answer Display */}
            {answer && (
              <div className="animate-fade-in">
                <AnswerDisplay
                  answer={answer}
                  context={context}
                  audioUrl={audioUrl}
                  ttsLoading={ttsLoading}
                />
              </div>
            )}

            {/* Training Chart */}
            {trainingData && (
              <TrainingChart data={trainingData} />
            )}
          </main>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 py-6 border-t border-[var(--border-color)] bg-[var(--bg-surface)]">
        <div className="container mx-auto px-6 text-center">
          <p className="text-sm text-[var(--text-muted)]">
            Telugu QA System • Powered by <span className="text-[var(--accent)] font-semibold">{currentModel?.name || 'MuRIL'}</span> • TeQuAD Dataset
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
