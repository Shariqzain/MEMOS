/* ─────────────────────────────────────────────────────────
 * MEMOS — TypeScript API types
 * ───────────────────────────────────────────────────────── */

export interface MemorySummary {
  memory_id: string;
  content: string;
  category: string;
  importance: number;
  confidence: number;
  rrf_score?: number;
}

export interface RetrievalDiagnostics {
  semantic_candidates: number;
  graph_candidates: number;
  final_memories: number;
  retrieval_mode: string;
  timing_ms?: number;
}

export interface ChatRequest {
  user_id: string;
  conversation_id?: string;
  message: string;
  retrieval_mode?: string;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  response: string;
  memories_used: MemorySummary[];
  diagnostics?: RetrievalDiagnostics;
  processing_ms?: number;
}

export interface ConversationOut {
  conversation_id: string;
  user_id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface MessageOut {
  message_id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  memory_ids_used: string[];
  created_at: string;
  processing_ms?: number;
}

export interface MemoryOut {
  memory_id: string;
  user_id: string;
  content: string;
  category: string;
  entities: Array<{ name: string; entity_type: string }>;
  importance: number;
  confidence: number;
  memory_type: string;
  status: string;
  version: number;
  source_conversation_id?: string;
  supersedes_memory_id?: string;
  superseded_by_memory_id?: string;
  created_at: string;
  updated_at: string;
  last_accessed_at?: string;
}

export interface MemoryVersionOut {
  version_id: string;
  memory_id: string;
  version: number;
  content: string;
  category: string;
  evolution_action: string;
  change_reason?: string;
  created_at: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  services: Record<string, string>;
}

export interface EvaluationSummary {
  total_memories: number;
  active_memories: number;
  superseded_memories: number;
  archived_memories: number;
  avg_importance: number;
  avg_confidence: number;
  total_conversations: number;
  total_messages: number;
  memory_categories: Record<string, number>;
}

// Internal UI types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  memoriesUsed?: MemorySummary[];
  processingMs?: number;
}
