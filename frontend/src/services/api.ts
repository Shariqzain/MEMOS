/* ─────────────────────────────────────────────────────────
 * MEMOS — Frontend API Service
 * Centralised backend communication layer.
 * ───────────────────────────────────────────────────────── */

import type {
  ChatRequest,
  ChatResponse,
  ConversationOut,
  EvaluationSummary,
  HealthResponse,
  MemoryOut,
  MemoryVersionOut,
  MessageOut,
} from '../types/api';

let API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

async function executeFetch(url: string, options: RequestInit) {
  return await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers as Record<string, string> },
    ...options,
  });
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let url = `${API_BASE}${path}`;
  let res: Response;
  try {
    res = await executeFetch(url, options);
  } catch (err) {
    // Automatic fallback between 8001 and 8000
    const altBase = API_BASE.includes(':8001')
      ? API_BASE.replace(':8001', ':8000')
      : API_BASE.replace(':8000', ':8001');
    try {
      res = await executeFetch(`${altBase}${path}`, options);
      API_BASE = altBase; // Cache working port
    } catch {
      throw err;
    }
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/* ── Health ──────────────────────────────────────────────── */
export const getHealth = () => request<HealthResponse>('/health');

/* ── Chat ───────────────────────────────────────────────── */
export const sendMessage = (req: ChatRequest) =>
  request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify(req),
  });

/* ── Conversations ──────────────────────────────────────── */
export const listConversations = (userId = 'demo_user') =>
  request<ConversationOut[]>(`/api/conversations?user_id=${userId}`);

export const getMessages = (conversationId: string, userId = 'demo_user') =>
  request<MessageOut[]>(
    `/api/conversations/${conversationId}/messages?user_id=${userId}`,
  );

/* ── Memories ───────────────────────────────────────────── */
export const listMemories = (userId = 'demo_user', category?: string) => {
  let url = `/api/memories?user_id=${userId}`;
  if (category) url += `&category=${category}`;
  return request<MemoryOut[]>(url);
};

export const getMemory = (memoryId: string, userId = 'demo_user') =>
  request<MemoryOut>(`/api/memories/${memoryId}?user_id=${userId}`);

export const updateMemory = (
  memoryId: string,
  data: Partial<{ content: string; category: string; importance: number; confidence: number; status: string }>,
  userId = 'demo_user',
) =>
  request<MemoryOut>(`/api/memories/${memoryId}?user_id=${userId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });

export const deleteMemory = (memoryId: string, userId = 'demo_user') =>
  request<{ message: string }>(`/api/memories/${memoryId}?user_id=${userId}`, {
    method: 'DELETE',
  });

export const getMemoryHistory = (memoryId: string, userId = 'demo_user') =>
  request<MemoryVersionOut[]>(
    `/api/memories/${memoryId}/history?user_id=${userId}`,
  );

export const restoreMemory = (
  memoryId: string,
  version: number,
  userId = 'demo_user',
) =>
  request<MemoryOut>(
    `/api/memories/${memoryId}/restore?version=${version}&user_id=${userId}`,
    { method: 'POST' },
  );

/* ── Evaluation ─────────────────────────────────────────── */
export const getEvaluationSummary = (userId = 'demo_user') =>
  request<EvaluationSummary>(`/api/evaluation/summary?user_id=${userId}`);
