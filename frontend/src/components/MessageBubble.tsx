/* ─────────────────────────────────────────────────────────
 * MEMOS — Chat Message Component
 * ───────────────────────────────────────────────────────── */
import type { ChatMessage, MemorySummary } from '../types/api';

interface Props {
  message: ChatMessage;
  onMemoryClick?: (memories: MemorySummary[]) => void;
}

export default function MessageBubble({ message, onMemoryClick }: Props) {
  const isUser = message.role === 'user';
  const memories = message.memoriesUsed ?? [];

  return (
    <div className="message-row">
      <div className={`message-avatar ${message.role}`}>
        {isUser ? '👤' : '🧠'}
      </div>
      <div className="message-body">
        <div className="message-role">
          {isUser ? 'You' : 'MEMOS Assistant'}
        </div>
        <div className="message-content">
          {message.content.split('\n').map((line, i) => (
            <p key={i}>{line || '\u00A0'}</p>
          ))}
        </div>
        <div className="message-meta">
          <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
          {message.processingMs && (
            <span>{message.processingMs}ms</span>
          )}
          {memories.length > 0 && (
            <span
              className="memory-badge"
              onClick={() => onMemoryClick?.(memories)}
              title="Click to see memories used"
            >
              🧠 {memories.length} {memories.length === 1 ? 'memory' : 'memories'}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
