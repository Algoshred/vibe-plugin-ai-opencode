/**
 * vibe-plugin-opencode
 *
 * OpenCode AI agent provider for VibeControls Agent.
 * Dual-mode: SDK (local HTTP API via fetch) or CLI (`opencode` binary).
 * Auto-detects mode based on available resources.
 */

import { Elysia } from "elysia";

// ── Locally Redeclared Interfaces ────────────────────────────────────────
// (Avoid hard dependency on @vibecontrols/agent)

type ProviderMode = "sdk" | "cli";

interface AIModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsStreaming: boolean;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

interface AIProviderCapabilities {
  streaming: boolean;
  vision: boolean;
  fileAttachments: boolean;
  toolUse: boolean;
  mcpSupport: boolean;
  voiceMode: boolean;
  cancelSupport: boolean;
  modelListing: boolean;
}

interface AIFileAttachment {
  filename: string;
  mimeType: string;
  content: Buffer | string;
  size: number;
}

interface VibePlugin {
  name: string;
  version: string;
  description?: string;
  tags?: Array<
    "backend" | "frontend" | "cli" | "provider" | "adapter" | "integration"
  >;
  cliCommand?: string;
  apiPrefix?: string;
  prerequisites?: Array<{
    name: string;
    kind: "binary" | "npm" | "pip" | "cargo" | "manual";
    requiresSudo: boolean;
    description?: string;
  }>;
  createRoutes?: () => unknown;
  providers?: { ai?: AIAgentProvider; [key: string]: unknown };
  onServerStart?: (
    app: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
  onServerStop?: () => void | Promise<void>;
  onCliSetup?: (
    program: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
}

interface HostServices {
  logger?: {
    info: (source: string, msg: string) => void;
    warn: (source: string, msg: string) => void;
    error: (source: string, msg: string) => void;
    debug: (source: string, msg: string) => void;
  };
  serviceRegistry?: {
    getService: <T>(pluginName: string, serviceName: string) => T | undefined;
  };
  getConfig: (key: string) => string | undefined;
}

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  thinkingSteps?: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
}

interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}

interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  agentMetadata?: Record<string, unknown>;
  createdAt: string;
}

interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
  modelBreakdown?: Record<
    string,
    { inputTokens: number; outputTokens: number; requestCount: number }
  >;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancelRequest?(sessionId: string): Promise<void>;
  getCapabilities?(): AIProviderCapabilities;
  attachFiles?(sessionId: string, files: AIFileAttachment[]): Promise<void>;
  getMode?(): ProviderMode;
  setMode?(mode: ProviderMode): void;
}

interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
    agentMetadata?: Record<string, unknown>;
  }): unknown;
}

// ── Adapter Interface ───────────────────────────────────────────────────

interface ProviderAdapter {
  sendPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
    remoteSessionId?: string,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    remoteSessionId?: string;
    metadata?: Record<string, unknown>;
  }>;
  streamPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    remoteSessionId?: string,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    remoteSessionId?: string;
    metadata?: Record<string, unknown>;
  }>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancel?(abortController: AbortController): void;
  createRemoteSession?(config: AISessionConfig): Promise<string>;
  destroyRemoteSession?(remoteSessionId: string): Promise<void>;
}

// ── Constants ───────────────────────────────────────────────────────────

const PROVIDER_NAME = "opencode";
const CLI_COMMAND = "opencode";
const DISPLAY_NAME = "OpenCode";
const DEFAULT_MODEL = "default";
const DEFAULT_PORT = 3100;
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk", "cli"];
const CLI_INSTALL_COMMAND = ["npm", "install", "-g", "opencode"];

// ── OpenCode API response types ─────────────────────────────────────────

interface OpenCodeSessionResponse {
  id: string;
  model?: string;
  [key: string]: unknown;
}

interface OpenCodePromptResponse {
  content?: string;
  text?: string;
  response?: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    inputTokens?: number;
    outputTokens?: number;
  };
  model?: string;
  [key: string]: unknown;
}

// ── SDK Adapter (HTTP API) ──────────────────────────────────────────────

class OpenCodeSdkAdapter implements ProviderAdapter {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async createRemoteSession(config: AISessionConfig): Promise<string> {
    const body: Record<string, unknown> = {};
    if (config.model) body.model = config.model;
    if (config.systemPrompt) body.systemPrompt = config.systemPrompt;
    if (config.workingDirectory)
      body.workingDirectory = config.workingDirectory;

    const res = await fetch(`${this.baseUrl}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(
        `OpenCode session creation failed (${res.status}): ${errText}`,
      );
    }

    const data = (await res.json()) as OpenCodeSessionResponse;
    return data.id;
  }

  async destroyRemoteSession(remoteSessionId: string): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/api/sessions/${remoteSessionId}`, {
        method: "DELETE",
      });
    } catch {
      // Best-effort cleanup
    }
  }

  async sendPrompt(
    prompt: string,
    _model: string,
    _config: AISessionConfig,
    remoteSessionId?: string,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    remoteSessionId?: string;
    metadata?: Record<string, unknown>;
  }> {
    const sessionId = remoteSessionId;
    if (!sessionId) {
      throw new Error("Remote session ID is required for SDK mode");
    }

    const res = await fetch(
      `${this.baseUrl}/api/sessions/${sessionId}/prompt`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      },
    );

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`OpenCode prompt failed (${res.status}): ${errText}`);
    }

    const data = (await res.json()) as OpenCodePromptResponse;
    const content = data.content ?? data.text ?? data.response ?? "";
    const inputTokens =
      data.usage?.promptTokens ??
      data.usage?.inputTokens ??
      Math.ceil(prompt.length / 4);
    const outputTokens =
      data.usage?.completionTokens ??
      data.usage?.outputTokens ??
      Math.ceil(content.length / 4);

    return {
      content,
      inputTokens,
      outputTokens,
      remoteSessionId: sessionId,
      metadata: { mode: "sdk", provider: PROVIDER_NAME, model: data.model },
    };
  }

  async streamPrompt(
    prompt: string,
    _model: string,
    _config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    remoteSessionId?: string,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    remoteSessionId?: string;
    metadata?: Record<string, unknown>;
  }> {
    const sessionId = remoteSessionId;
    if (!sessionId) {
      throw new Error("Remote session ID is required for SDK mode streaming");
    }

    const res = await fetch(
      `${this.baseUrl}/api/sessions/${sessionId}/prompt`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ prompt, stream: true }),
      },
    );

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(
        `OpenCode stream prompt failed (${res.status}): ${errText}`,
      );
    }

    let fullContent = "";

    if (res.body) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE lines
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).trim();
              if (dataStr === "[DONE]") continue;

              try {
                const parsed = JSON.parse(dataStr) as {
                  content?: string;
                  text?: string;
                  type?: string;
                };
                const text = parsed.content ?? parsed.text ?? "";
                if (text) {
                  fullContent += text;
                  onChunk({ type: "text", content: text });
                }
              } catch {
                // Non-JSON SSE data -- treat as raw text
                if (dataStr) {
                  fullContent += dataStr;
                  onChunk({ type: "text", content: dataStr });
                }
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } else {
      // Fallback: non-streaming response body
      const data = (await res.json()) as OpenCodePromptResponse;
      fullContent = data.content ?? data.text ?? data.response ?? "";
      if (fullContent) {
        onChunk({ type: "text", content: fullContent });
      }
    }

    onChunk({ type: "done", content: "" });

    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(fullContent.length / 4);

    return {
      content: fullContent,
      inputTokens,
      outputTokens,
      remoteSessionId: sessionId,
      metadata: { mode: "sdk", provider: PROVIDER_NAME },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });

      if (res.ok) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} HTTP API reachable at ${this.baseUrl} (mode: sdk)`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} API returned ${res.status}`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} API not reachable at ${this.baseUrl}`,
      };
    }
  }

  cancel(_abortController: AbortController): void {
    _abortController.abort();
  }
}

// ── CLI Adapter ─────────────────────────────────────────────────────────

class OpenCodeCliAdapter implements ProviderAdapter {
  async sendPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }> {
    const args = this.buildArgs(model, prompt);
    const proc = Bun.spawn([CLI_COMMAND, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      timeout: (config.providerConfig?.timeoutMs as number) || 300_000,
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0 && !stdout) {
      throw new Error(
        `${DISPLAY_NAME} CLI exited with code ${exitCode}: ${stderr}`,
      );
    }

    const content = stdout.trim() || stderr.trim();
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);

    return {
      content,
      inputTokens,
      outputTokens,
      metadata: { mode: "cli", exitCode, provider: PROVIDER_NAME },
    };
  }

  async streamPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }> {
    const args = this.buildArgs(model, prompt);
    const proc = Bun.spawn([CLI_COMMAND, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      timeout: (config.providerConfig?.timeoutMs as number) || 300_000,
    });

    let fullContent = "";
    const reader = proc.stdout.getReader();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = new TextDecoder().decode(value);
        fullContent += text;
        onChunk({ type: "text", content: text });
      }
    } finally {
      reader.releaseLock();
    }

    await proc.exited;
    onChunk({ type: "done", content: "" });

    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(fullContent.length / 4);

    return {
      content: fullContent,
      inputTokens,
      outputTokens,
      metadata: { mode: "cli", provider: PROVIDER_NAME },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const proc = Bun.spawnSync([CLI_COMMAND, "--version"], {
        timeout: 5000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} CLI ${proc.stdout.toString().trim()} (mode: cli)`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not available (exit code ${proc.exitCode})`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not installed or not in PATH`,
      };
    }
  }

  private buildArgs(model: string, prompt: string): string[] {
    const args: string[] = ["run"];
    if (model && model !== "default") args.push("--model", model);
    args.push(prompt);
    return args;
  }
}

// ── Managed Session ─────────────────────────────────────────────────────

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  files: AIFileAttachment[];
  remoteSessionId: string | null;
  abortController: AbortController | null;
  createdAt: string;
  updatedAt: string;
}

// ── Provider Implementation ─────────────────────────────────────────────

class OpenCodeProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private adapter: ProviderAdapter | null = null;
  private currentMode: ProviderMode | null = null;

  setHostServices(hs: HostServices) {
    this.hostServices = hs;
    this.logIngester =
      hs.serviceRegistry?.getService<LogIngester>("ai", "log-ingester") ?? null;
  }

  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }

  getDisplayName(): string {
    return DISPLAY_NAME;
  }

  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }

  getMode(): ProviderMode {
    if (!this.currentMode) {
      this.autoDetectMode();
    }
    return this.currentMode!;
  }

  setMode(mode: ProviderMode): void {
    if (!SUPPORTED_MODES.includes(mode)) {
      throw new Error(`${DISPLAY_NAME} does not support ${mode} mode`);
    }
    this.currentMode = mode;
    this.adapter = null;
    this.log("info", `Mode set to: ${mode}`);
  }

  getCapabilities(): AIProviderCapabilities {
    return {
      streaming: true,
      vision: false,
      fileAttachments: true,
      toolUse: true,
      mcpSupport: true,
      voiceMode: false,
      cancelSupport: true,
      modelListing: false,
    };
  }

  async listModels(): Promise<AIModelInfo[]> {
    // OpenCode models depend on its configured provider; we cannot enumerate them
    return [];
  }

  async cancelRequest(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.abortController) {
      session.abortController.abort();
      session.abortController = null;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Request cancelled for session ${sessionId}`);
    }
  }

  async attachFiles(
    sessionId: string,
    files: AIFileAttachment[],
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    session.files.push(...files);
    session.updatedAt = new Date().toISOString();
    this.log(
      "debug",
      `Attached ${files.length} file(s) to session ${sessionId}`,
    );
  }

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();

    // Return existing session if already in memory
    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return {
        id,
        name: existing.config.name,
        status: "active",
        agentType: existing.config.agentType,
        provider: PROVIDER_NAME,
        config: existing.config,
        stats: existing.stats,
        createdAt: existing.createdAt,
        updatedAt: now,
      };
    }

    let remoteSessionId: string | null = null;

    // In SDK mode, create a remote session on the OpenCode server
    if (this.getMode() === "sdk") {
      const adapter = this.getAdapter();
      if (adapter.createRemoteSession) {
        try {
          remoteSessionId = await adapter.createRemoteSession(config);
          this.log(
            "debug",
            `Remote OpenCode session created: ${remoteSessionId}`,
          );
        } catch (err) {
          this.log(
            "error",
            `Failed to create remote session: ${err instanceof Error ? err.message : "unknown"}`,
          );
        }
      }
    }

    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      files: [],
      remoteSessionId,
      abortController: null,
      createdAt: now,
      updatedAt: now,
    };

    this.sessions.set(id, session);
    this.log(
      "info",
      `Session created: ${id} (${config.name}) [${this.getMode()}]`,
    );

    return {
      id,
      name: config.name,
      status: "active",
      agentType: config.agentType,
      provider: PROVIDER_NAME,
      config,
      stats: session.stats,
      createdAt: now,
      updatedAt: now,
    };
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");

    session.status = "processing";
    session.updatedAt = new Date().toISOString();
    const startTime = Date.now();

    const fullPrompt = this.buildFullPrompt(prompt, context, session);
    const model = session.config.model || DEFAULT_MODEL;
    const adapter = this.getAdapter();

    this.logIngester?.append({ sessionId, type: "input", content: prompt });

    try {
      const result = await adapter.sendPrompt(
        fullPrompt,
        model,
        session.config,
        session.remoteSessionId ?? undefined,
      );
      const durationMs = Date.now() - startTime;

      if (result.remoteSessionId && !session.remoteSessionId) {
        session.remoteSessionId = result.remoteSessionId;
      }

      this.updateStats(session, result.inputTokens, result.outputTokens, model);
      session.status = "active";
      session.updatedAt = new Date().toISOString();

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model,
        durationMs,
      });

      return {
        content: result.content,
        model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({ sessionId, type: "error", content: errorMsg });
      throw err;
    }
  }

  async streamPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");

    session.status = "processing";
    session.abortController = new AbortController();
    session.updatedAt = new Date().toISOString();
    const startTime = Date.now();

    const fullPrompt = this.buildFullPrompt(prompt, context, session);
    const model = session.config.model || DEFAULT_MODEL;
    const adapter = this.getAdapter();

    this.logIngester?.append({ sessionId, type: "input", content: prompt });

    try {
      const result = await adapter.streamPrompt(
        fullPrompt,
        model,
        session.config,
        onChunk ?? (() => {}),
        session.remoteSessionId ?? undefined,
      );
      const durationMs = Date.now() - startTime;

      if (result.remoteSessionId && !session.remoteSessionId) {
        session.remoteSessionId = result.remoteSessionId;
      }

      this.updateStats(session, result.inputTokens, result.outputTokens, model);
      session.status = "active";
      session.abortController = null;
      session.updatedAt = new Date().toISOString();

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model,
        durationMs,
      });

      return {
        content: result.content,
        model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.abortController = null;
      session.updatedAt = new Date().toISOString();
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({ sessionId, type: "error", content: errorMsg });
      throw err;
    }
  }

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }

  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    const session = this.sessions.get(sessionId);
    return (
      session?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }

  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    Object.assign(session.config, config);
    session.updatedAt = new Date().toISOString();
  }

  async destroySession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.abortController) session.abortController.abort();

      // Cleanup remote session in SDK mode
      if (session.remoteSessionId && this.getMode() === "sdk") {
        const adapter = this.getAdapter();
        if (adapter.destroyRemoteSession) {
          adapter.destroyRemoteSession(session.remoteSessionId).catch(() => {});
        }
      }

      session.status = "terminated";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Session terminated: ${sessionId}`);
    }
  }

  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => ({
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    }));
  }

  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    const adapter = this.getAdapter();
    return adapter.healthCheck();
  }

  // ── Private Helpers ─────────────────────────────────────────────────

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;

    const mode = this.getMode();
    if (mode === "sdk") {
      const port = process.env["OPENCODE_PORT"] || String(DEFAULT_PORT);
      const baseUrl = process.env["OPENCODE_URL"] || `http://localhost:${port}`;
      this.adapter = new OpenCodeSdkAdapter(baseUrl);
    } else {
      this.adapter = new OpenCodeCliAdapter();
    }

    return this.adapter;
  }

  private autoDetectMode(): void {
    // SDK mode: check if OpenCode local server is configured (port or URL)
    const opencodeUrl = process.env["OPENCODE_URL"];
    const opencodePort = process.env["OPENCODE_PORT"];

    if (opencodeUrl || opencodePort) {
      this.currentMode = "sdk";
      this.log(
        "info",
        `Auto-detected SDK mode (${opencodeUrl ? "OPENCODE_URL" : "OPENCODE_PORT"} set)`,
      );
      return;
    }

    // CLI mode: check if the opencode binary exists
    try {
      // Cross-platform: `which` on POSIX, `where.exe` on Windows.
      const finder = process.platform === "win32" ? "where.exe" : "which";
      const proc = Bun.spawnSync([finder, CLI_COMMAND], {
        timeout: 3000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) {
        this.currentMode = "cli";
        this.log("info", "Auto-detected CLI mode (opencode binary found)");
        return;
      }
    } catch {
      // binary not found
    }

    this.currentMode = "cli";
    this.log(
      "info",
      "No OPENCODE_URL/OPENCODE_PORT or CLI binary found, defaulting to CLI mode (will error on use)",
    );
  }

  private buildFullPrompt(
    prompt: string,
    context: AIContext[] | undefined,
    session: ManagedSession,
  ): string {
    const parts: string[] = [];

    if (session.config.systemPrompt) {
      parts.push(`System: ${session.config.systemPrompt}\n`);
    }

    if (context && context.length > 0) {
      const contextStr = context
        .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
        .join("\n\n");
      parts.push(contextStr);
    }

    if (session.files.length > 0) {
      const fileContext = session.files
        .map((f) => {
          const text =
            typeof f.content === "string"
              ? f.content
              : f.content.toString("utf-8");
          return `--- File: ${f.filename} (${f.mimeType}) ---\n${text}`;
        })
        .join("\n\n");
      parts.push(fileContext);
    }

    parts.push(prompt);
    return parts.join("\n\n");
  }

  private updateStats(
    session: ManagedSession,
    inputTokens: number,
    outputTokens: number,
    model: string,
  ): void {
    session.stats.inputTokens += inputTokens;
    session.stats.outputTokens += outputTokens;
    session.stats.requestCount += 1;
    // OpenCode costs depend on its configured backend -- no cost estimation

    if (!session.stats.modelBreakdown) session.stats.modelBreakdown = {};
    const breakdown = session.stats.modelBreakdown[model] ?? {
      inputTokens: 0,
      outputTokens: 0,
      requestCount: 0,
    };
    breakdown.inputTokens += inputTokens;
    breakdown.outputTokens += outputTokens;
    breakdown.requestCount += 1;
    session.stats.modelBreakdown[model] = breakdown;
  }

  private log(level: "info" | "error" | "debug", msg: string) {
    this.hostServices?.logger?.[level]?.(`${PROVIDER_NAME}-provider`, msg);
  }
}

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([CLI_COMMAND, "--version"], {
      timeout: 5000,
      stdout: "pipe",
      stderr: "ignore",
    });
    if (proc.exitCode === 0) return proc.stdout.toString().trim();
  } catch {
    // Binary not found.
  }
  return null;
}

function createPrereqsRoutes() {
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => {
      const version = getCliVersion();
      return {
        satisfied: Boolean(version),
        missing: version
          ? []
          : [
              {
                name: CLI_COMMAND,
                kind: "npm" as const,
                requiresSudo: false,
                description: `${DISPLAY_NAME} CLI for CLI mode`,
              },
            ],
      };
    })
    .post("/install", () => {
      if (getCliVersion()) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }

      const proc = Bun.spawnSync(CLI_INSTALL_COMMAND, {
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }
      return {
        ok: false,
        installed: [],
        pendingSudo: [],
        errors: [
          {
            name: CLI_COMMAND,
            message:
              proc.stderr.toString().trim() ||
              `Run manually: ${CLI_INSTALL_COMMAND.join(" ")}`,
          },
        ],
      };
    });
}

const provider = new OpenCodeProvider();

export const vibePlugin: VibePlugin = {
  name: "opencode",
  version: "1.0.0",
  description:
    "OpenCode AI agent provider for VibeControls (SDK + CLI dual-mode)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: "npm",
      requiresSudo: false,
      description: `${DISPLAY_NAME} CLI for CLI mode`,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),

  onServerStart(_app, hostServices) {
    if (hostServices) provider.setHostServices(hostServices);
  },

  onServerStop() {
    for (const [id] of (provider as OpenCodeProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
};

export default vibePlugin;
