/* ═══════════════════════════════════════════════════════════
   MEMOS — Main Application
   Memory Evolution and Management for Organized Semantic Storage
   ═══════════════════════════════════════════════════════════ */
import { useCallback, useEffect, useRef, useState } from 'react';
import './index.css';

import Sidebar from './components/Sidebar';
import MessageBubble from './components/MessageBubble';
import MemoryPanel from './components/MemoryPanel';
import { sendMessage, getMessages } from './services/api';
import type { ChatMessage, MemorySummary } from './types/api';

const HINTS = [
  "Tell me about yourself — what are you working on?",
  "I prefer dark mode and vim keybindings",
  "I'm building a project with React and FastAPI",
  "What do you remember about me?",
];

export default function App() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = useCallback(async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;
    setError('');
    setInput('');

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: msg,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await sendMessage({
        user_id: 'demo_user',
        conversation_id: conversationId ?? undefined,
        message: msg,
        retrieval_mode: 'hybrid',
      });

      if (!conversationId) {
        setConversationId(res.conversation_id);
        setRefreshKey((k) => k + 1);
      }

      const assistantMsg: ChatMessage = {
        id: res.message_id,
        role: 'assistant',
        content: res.response,
        timestamp: new Date().toISOString(),
        memoriesUsed: res.memories_used,
        processingMs: res.processing_ms,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: any) {
      setError(e.message || 'Failed to send message');
    } finally {
      setLoading(false);
    }
  }, [input, loading, conversationId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    setConversationId(null);
    setMessages([]);
    setError('');
  };

  const handleSelectConversation = async (id: string) => {
    setConversationId(id);
    setError('');
    try {
      const msgs = await getMessages(id);
      setMessages(
        msgs.map((m) => ({
          id: m.message_id,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          timestamp: m.created_at,
          processingMs: m.processing_ms ?? undefined,
        })),
      );
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleMemoryClick = (_memories: MemorySummary[]) => {
    setMemoryPanelOpen(true);
  };

  return (
    <div className="app-layout">
      {/* Left sidebar */}
      <Sidebar
        activeId={conversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
        refreshKey={refreshKey}
      />

      {/* Main chat */}
      <main className="chat-main">
        <header className="chat-header">
          <span className="chat-header-title">
            {conversationId ? 'Conversation' : 'New Conversation'}
          </span>
          <div className="chat-header-actions">
            <button
              className={`header-btn ${memoryPanelOpen ? 'active' : ''}`}
              onClick={() => setMemoryPanelOpen((v) => !v)}
              title="Toggle memory panel"
            >
              🧠
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            ⚠ {error}
          </div>
        )}

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="welcome-state">
              <div className="welcome-icon">🧠</div>
              <h2>MEMOS</h2>
              <p>
                Memory Evolution and Management for Organized Semantic Storage.
                I remember what matters — start a conversation and I'll learn your
                preferences, projects, and goals over time.
              </p>
              <div className="welcome-hints">
                {HINTS.map((h, i) => (
                  <button
                    key={i}
                    className="welcome-hint"
                    onClick={() => handleSend(h)}
                  >
                    {h}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-container">
              {messages.map((m) => (
                <MessageBubble
                  key={m.id}
                  message={m}
                  onMemoryClick={handleMemoryClick}
                />
              ))}
              {loading && (
                <div className="message-row">
                  <div className="message-avatar assistant">🧠</div>
                  <div className="message-body">
                    <div className="message-role">MEMOS Assistant</div>
                    <div className="typing-indicator">
                      <span /><span /><span />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="chat-composer">
          <div className="composer-inner">
            <div className="composer-input-wrap">
              <textarea
                ref={textareaRef}
                className="composer-textarea"
                placeholder="Type a message… (Enter to send, Shift+Enter for new line)"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={loading}
              />
            </div>
            <button
              className="btn-send"
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              title="Send message"
            >
              ↑
            </button>
          </div>
          <div className="composer-note">
            MEMOS extracts and remembers useful information across conversations
          </div>
        </div>
      </main>

      {/* Right memory panel */}
      <MemoryPanel
        visible={memoryPanelOpen}
        onClose={() => setMemoryPanelOpen(false)}
      />
    </div>
  );
}
