/* ─────────────────────────────────────────────────────────
 * MEMOS — Sidebar: conversations list + new-chat
 * ───────────────────────────────────────────────────────── */
import { useEffect, useState } from 'react';
import type { ConversationOut } from '../types/api';
import { listConversations } from '../services/api';

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  refreshKey: number;
}

export default function Sidebar({ activeId, onSelect, onNewChat, refreshKey }: Props) {
  const [conversations, setConversations] = useState<ConversationOut[]>([]);

  useEffect(() => {
    listConversations()
      .then(setConversations)
      .catch(console.error);
  }, [refreshKey]);

  const fmt = (iso: string) => {
    const d = new Date(iso);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon">M</div>
          <div>
            <h1>MEMOS</h1>
            <span className="version">v0.1.0</span>
          </div>
        </div>
        <button className="btn-new-chat" onClick={onNewChat}>
          ✦ New Conversation
        </button>
      </div>

      <div className="sidebar-conversations">
        <div className="sidebar-section-label">Conversations</div>
        {conversations.length === 0 && (
          <div className="empty-state">No conversations yet</div>
        )}
        {conversations.map((c) => (
          <div
            key={c.conversation_id}
            className={`conversation-item ${c.conversation_id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(c.conversation_id)}
          >
            <div className="conv-title">
              {c.title || `Conversation`}
            </div>
            <div className="conv-meta">
              {c.message_count} messages · {fmt(c.updated_at)}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
