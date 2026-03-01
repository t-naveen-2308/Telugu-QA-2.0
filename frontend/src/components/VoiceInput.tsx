import { useState, useRef } from 'react';
import { api } from '../api';
import { Mic, Square, Upload, Loader2, Search, X, FileText } from 'lucide-react';

interface Props {
  context: string;
  onContextChange: (value: string) => void;
  onTranscription: (text: string) => void;
  onSubmit: () => void;
  loading: boolean;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function VoiceInput({
  context,
  onContextChange,
  onTranscription,
  onSubmit,
  loading,
  onFileUpload
}: Props) {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [transcription, setTranscription] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      setTranscription('');
    }
  };

  const transcribeAudio = async () => {
    if (!audioBlob) return;

    setTranscribing(true);
    setError(null);

    try {
      const result = await api.transcribe(audioBlob, 'recording.webm');
      setTranscription(result.text);
      onTranscription(result.text);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Transcription failed');
    } finally {
      setTranscribing(false);
    }
  };

  const clearRecording = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setTranscription('');
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
        <textarea
          value={context}
          onChange={(e) => onContextChange(e.target.value)}
          placeholder="తెలుగులో టెక్స్ట్ ఇక్కడ పేస్ట్ చేయండి..."
          className="w-full h-32 p-4 rounded-lg font-telugu text-lg resize-none bg-[var(--bg-elevated)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)]"
        />
      </div>

      {/* Voice Recording Section */}
      <div>
        <label className="font-medium text-[var(--text-primary)] mb-2.5 flex items-center gap-2 text-sm">
          <span className="w-6 h-6 rounded-md bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
            <Mic className="w-3.5 h-3.5 text-red-600 dark:text-red-400" />
          </span>
          Voice Question
        </label>

        <div className="bg-[var(--bg-elevated)] rounded-lg p-5 border border-[var(--border-color)]">
          {/* Recording Controls */}
          <div className="flex items-center justify-center gap-4 mb-4">
            {!recording ? (
              <button
                onClick={startRecording}
                className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-red-500 to-rose-500 text-white rounded-lg hover:from-red-600 hover:to-rose-600 transition-all shadow-md text-sm font-medium"
              >
                <Mic size={18} />
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="flex items-center gap-2 px-5 py-2.5 bg-stone-700 text-white rounded-lg hover:bg-stone-800 transition-all shadow-md animate-pulse text-sm font-medium"
              >
                <Square size={18} />
                Stop Recording
              </button>
            )}

            <span className="text-[var(--text-muted)] text-sm">or</span>

            <label className="flex items-center gap-2 px-4 py-2 bg-[var(--bg-surface)] border border-[var(--border-color)] rounded-lg cursor-pointer hover:bg-[var(--bg-hover)] transition-colors text-[var(--text-secondary)] text-sm">
              <Upload size={16} />
              Upload Audio
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>

          {/* Audio Preview */}
          {audioUrl && (
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <audio controls src={audioUrl} className="flex-1 h-10" />
                <button
                  onClick={clearRecording}
                  className="p-2 text-[var(--text-muted)] hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors"
                >
                  <X size={16} />
                </button>
              </div>

              {/* Transcribe Button */}
              {!transcription && (
                <button
                  onClick={transcribeAudio}
                  disabled={transcribing}
                  className="btn btn-accent w-full shadow-md hover:shadow-lg transition-all"
                >
                  {transcribing ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      Transcribing Telugu...
                    </>
                  ) : (
                    'Transcribe Audio'
                  )}
                </button>
              )}

              {/* Transcription Result */}
              {transcription && (
                <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/20 border border-amber-200 dark:border-amber-800/40 rounded-lg p-4">
                  <p className="text-xs text-amber-700 dark:text-amber-300 mb-1 font-medium uppercase tracking-wide">Transcribed question:</p>
                  <p className="font-telugu text-xl text-amber-800 dark:text-amber-200">{transcription}</p>
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-4 bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300 px-4 py-3 rounded-lg text-sm border border-red-200 dark:border-red-900/50">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Submit Button */}
      {transcription && (
        <button
          onClick={onSubmit}
          disabled={loading || !context.trim()}
          className="btn btn-accent w-full shadow-md hover:shadow-lg transition-all"
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Getting Answer...
            </>
          ) : (
            <>
              <Search size={18} />
              Get Answer / సమాధానం పొందండి
            </>
          )}
        </button>
      )}
    </div>
  );
}
