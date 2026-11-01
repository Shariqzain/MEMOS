/* ─────────────────────────────────────────────────────────
 * MEMOS — Memory Inspection Panel (right sidebar)
 * ───────────────────────────────────────────────────────── */
import { useEffect, useState } from 'react';
import type { MemoryOut, MemoryVersionOut } from '../types/api';
import {
  listMemories,
  deleteMemory,
  getMemoryHistory,
  updateMemory,
} from '../services/api';

interface Props {
  visible: boolean;
  onClose: () => void;
}

const CATEGORIES = [
  '', 'preference', 'goal', 'project', 'skill',
  'constraint', 'relationship', 'temporal', 'factual', 'general',
];

export default function MemoryPanel({ visible, onClose }: Props) {
  const [memories, setMemories] = useState<MemoryOut[]>([]);
  const [filter, setFilter] = useState('');
  const [selected, setSelected] = useState<MemoryOut | null>(null);
  const [history, setHistory] = useState<MemoryVersionOut[]>([]);
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    listMemories('demo_user', filter || undefined)
      .then(setMemories)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (visible) load();
  }, [visible, filter]);

  const handleSelect = async (mem: MemoryOut) => {
    setSelected(mem);
    setEditing(false);
    try {
      const h = await getMemoryHistory(mem.memory_id);
      setHistory(h);
    } catch {
      setHistory([]);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this memory?')) return;
    try {
      await deleteMemory(id);
      setSelected(null);
      load();
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleSaveEdit = async () => {
    if (!selected) return;
    try {
      await updateMemory(selected.memory_id, { content: editContent });
      setEditing(false);
      load();
      setSelected(null);
    } catch (e: any) {
      alert(e.message);
    }
  };

  const catClass = (cat: string) => `memory-card-category cat-${cat}`;

  if (!visible) return null;

  return (
    <div className={`memory-panel ${visible ? '' : 'hidden'}`}>
      <div className="memory-panel-header">
        <h3>🧠 Memories</h3>
        <button className="header-btn" onClick={onClose} title="Close">✕</button>
      </div>

      <div className="memory-filter">
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">All categories</option>
          {CATEGORIES.filter(Boolean).map((c) => (
            <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
          ))}
        </select>
      </div>

      <div className="memory-list">
        {loading && <div className="empty-state">Loading…</div>}
        {!loading && memories.length === 0 && (
          <div className="empty-state">No memories stored yet.<br />Start chatting to build memory!</div>
        )}
        {memories.map((m) => (
          <div
            key={m.memory_id}
            className="memory-card"
            onClick={() => handleSelect(m)}
          >
            <span className={catClass(m.category)}>{m.category}</span>
            <div className="memory-card-content">{m.content}</div>
            <div className="memory-card-footer">
              <span className="memory-score">⬆ {(m.importance * 100).toFixed(0)}%</span>
              <span className="memory-score">🎯 {(m.confidence * 100).toFixed(0)}%</span>
              <span>v{m.version}</span>
            </div>
            <div className="memory-card-actions">
              <button
                className="memory-action-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelected(m);
                  setEditing(true);
                  setEditContent(m.content);
                }}
              >
                Edit
              </button>
              <button
                className="memory-action-btn danger"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(m.memory_id);
                }}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* ── Detail / Edit Modal ───────────────────────────── */}
      {selected && (
        <div className="modal-overlay" onClick={() => { setSelected(null); setEditing(false); }}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>{editing ? 'Edit Memory' : 'Memory Detail'}</h3>

            {editing ? (
              <>
                <div className="modal-field">
                  <label>Content</label>
                  <textarea
                    rows={4}
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                  />
                </div>
                <div className="modal-actions">
                  <button className="btn btn-ghost" onClick={() => setEditing(false)}>Cancel</button>
                  <button className="btn btn-primary" onClick={handleSaveEdit}>Save</button>
                </div>
              </>
            ) : (
              <>
                <div className="modal-field">
                  <label>Content</label>
                  <p style={{ fontSize: 14, lineHeight: 1.6 }}>{selected.content}</p>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div className="modal-field">
                    <label>Category</label>
                    <p style={{ fontSize: 13 }}>{selected.category}</p>
                  </div>
                  <div className="modal-field">
                    <label>Status</label>
                    <p style={{ fontSize: 13 }}>{selected.status}</p>
                  </div>
                  <div className="modal-field">
                    <label>Importance</label>
                    <p style={{ fontSize: 13 }}>{(selected.importance * 100).toFixed(0)}%</p>
                  </div>
                  <div className="modal-field">
                    <label>Confidence</label>
                    <p style={{ fontSize: 13 }}>{(selected.confidence * 100).toFixed(0)}%</p>
                  </div>
                </div>
                {selected.entities.length > 0 && (
                  <div className="modal-field">
                    <label>Entities</label>
                    <p style={{ fontSize: 13 }}>
                      {selected.entities.map((e) => e.name).join(', ')}
                    </p>
                  </div>
                )}
                <div className="modal-field">
                  <label>Version {selected.version} · Created {new Date(selected.created_at).toLocaleDateString()}</label>
                </div>

                {/* History */}
                {history.length > 0 && (
                  <div className="modal-field">
                    <label>Version History</label>
                    <div className="version-timeline">
                      {history.map((v) => (
                        <div key={v.version_id} className="version-entry">
                          <div className="version-dot" />
                          <div className="version-info">
                            <div className="version-action">{v.evolution_action}</div>
                            <div className="version-content">{v.content}</div>
                            <div className="version-meta">
                              v{v.version} · {v.change_reason || '—'} · {new Date(v.created_at).toLocaleString()}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="modal-actions">
                  <button className="btn btn-danger" onClick={() => handleDelete(selected.memory_id)}>
                    Delete
                  </button>
                  <button className="btn btn-ghost" onClick={() => { setEditing(true); setEditContent(selected.content); }}>
                    Edit
                  </button>
                  <button className="btn btn-ghost" onClick={() => { setSelected(null); setEditing(false); }}>
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
