/**
 * vibe-plugin-opencode
 *
 * OpenCode AI agent provider for VibeControls Agent.
 * Dual-mode: SDK (local HTTP API via fetch) or CLI (`opencode` binary).
 * Auto-detects mode based on available resources.
 */

import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Elysia } from "elysia";
import type {
  HostServices,
  VibePlugin,
  ProfileContext,
} from "@vibecontrols/plugin-sdk";
import {
  BoundLogger,
  ProviderRegistry,
  TelemetryEmitter,
  createLifecycleHooks,
} from "@vibecontrols/plugin-sdk";

// ── AI Provider Contract Types ──────────────────────────────────────────
// (provider-specific contract — kept inline; not part of the SDK surface)

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
  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null;
  sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }>;
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
/**
 * Resolve CLI binary path with platform-correct extension.
 * On Windows, Bun.spawn calls CreateProcess directly (no PATHEXT), so a bare
 * name won't find `name.exe`/`name.cmd`. Bun.which searches PATH like the shell.
 */
function platformExeName(base: string): string {
  return process.platform === "win32" ? `${base}.exe` : base;
}

function resolveCliBin(): string {
  const found =
    typeof Bun !== "undefined" && typeof Bun.which === "function"
      ? Bun.which(CLI_COMMAND, { PATH: process.env.PATH })
      : null;
  if (found) return found;
  return platformExeName(CLI_COMMAND);
}
const CLI_BIN = resolveCliBin();

const DISPLAY_NAME = "OpenCode";
const DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514";
const DEFAULT_PORT = 3100;
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk", "cli"];
const CLI_INSTALL_COMMAND = ["npm", "install", "-g", "opencode"];

const OPENCODE_FALLBACK_MODELS = [
  "anthropic/claude-sonnet-4-20250514",
  "anthropic/claude-sonnet-4-5-20250929",
  "openai/gpt-5.5",
  "openai/gpt-5.5-pro",
  "openrouter/anthropic/claude-sonnet-4",
  "openrouter/openai/gpt-5.5",
  "openrouter/google/gemini-2.5-flash",
  "opencode/claude-sonnet-4",
  "opencode/claude-sonnet-4-5",
  "opencode/gpt-5.5",
  "opencode/gpt-5.5-pro",
  "opencode/gemini-3-flash",
];

/** The OpenCode server message response: `{ info: {...}, parts: [...] }`. */
interface OpenCodeMessageResponse {
  info?: {
    modelID?: string;
    tokens?: { input?: number; output?: number };
  };
  parts?: Array<{ type?: string; text?: string }>;
}

/**
 * Split a `provider/model` id into the `{ providerID, modelID }` shape the
 * OpenCode server expects. The provider is the first segment; everything after
 * the first `/` is the model id (model ids can themselves contain slashes,
 * e.g. `openrouter/openai/gpt-4o-mini`).
 */
function splitOpenCodeModel(model: string): {
  providerID: string;
  modelID: string;
} {
  const slash = model.indexOf("/");
  if (slash === -1) return { providerID: "opencode", modelID: model };
  return {
    providerID: model.slice(0, slash),
    modelID: model.slice(slash + 1),
  };
}

/** Concatenate the text parts of an OpenCode assistant message. */
function extractMessageText(data: OpenCodeMessageResponse): string {
  return (data.parts ?? [])
    .filter((p) => p.type === "text" && typeof p.text === "string")
    .map((p) => p.text ?? "")
    .join("")
    .trim();
}

function toOpenCodeModelInfo(id: string): AIModelInfo {
  return {
    id,
    name: id,
    provider: PROVIDER_NAME,
    contextWindow: 0,
    maxOutputTokens: 0,
    supportsVision: false,
    supportsStreaming: true,
    inputPricePerMToken: 0,
    outputPricePerMToken: 0,
  };
}

function fallbackModels(): AIModelInfo[] {
  return OPENCODE_FALLBACK_MODELS.map(toOpenCodeModelInfo);
}

// ── SDK Adapter (HTTP API) ──────────────────────────────────────────────

class OpenCodeSdkAdapter implements ProviderAdapter {
  private readonly resolveBaseUrl: () => Promise<string>;

  /**
   * Takes an async resolver so the managed `opencode serve` is started lazily
   * on first use (and an explicit OPENCODE_URL still wins). Each request
   * resolves the live base URL before calling the server.
   */
  constructor(resolveBaseUrl: () => Promise<string>) {
    this.resolveBaseUrl = resolveBaseUrl;
  }

  async createRemoteSession(_config: AISessionConfig): Promise<string> {
    const baseUrl = await this.resolveBaseUrl();
    // OpenCode server: POST /session creates a session and returns { id, ... }.
    const res = await fetch(`${baseUrl}/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(
        `OpenCode session creation failed (${res.status}): ${errText}`,
      );
    }

    const data = (await res.json()) as { id: string };
    return data.id;
  }

  async destroyRemoteSession(remoteSessionId: string): Promise<void> {
    try {
      const baseUrl = await this.resolveBaseUrl();
      await fetch(`${baseUrl}/session/${remoteSessionId}`, {
        method: "DELETE",
      });
    } catch {
      // Best-effort cleanup
    }
  }

  async sendPrompt(
    prompt: string,
    model: string,
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

    const baseUrl = await this.resolveBaseUrl();
    // OpenCode server: POST /session/{id}/message runs the prompt to completion
    // and returns { info: {...usage...}, parts: [{type:"text",text}, ...] }.
    const res = await fetch(`${baseUrl}/session/${sessionId}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parts: [{ type: "text", text: prompt }],
        model: splitOpenCodeModel(model),
      }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`OpenCode prompt failed (${res.status}): ${errText}`);
    }

    const data = (await res.json()) as OpenCodeMessageResponse;
    const content = extractMessageText(data);
    const inputTokens =
      data.info?.tokens?.input ?? Math.ceil(prompt.length / 4);
    const outputTokens =
      data.info?.tokens?.output ?? Math.ceil(content.length / 4);

    return {
      content,
      inputTokens,
      outputTokens,
      remoteSessionId: sessionId,
      metadata: {
        mode: "sdk",
        provider: PROVIDER_NAME,
        model: data.info?.modelID,
      },
    };
  }

  async streamPrompt(
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
  }> {
    // The OpenCode message endpoint runs to completion server-side; emit the
    // final assistant text as a single chunk so the streaming contract holds.
    const result = await this.sendPrompt(
      prompt,
      model,
      config,
      remoteSessionId,
    );
    if (result.content) onChunk({ type: "text", content: result.content });
    onChunk({ type: "done", content: "" });
    return result;
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    let baseUrl = "";
    try {
      baseUrl = await this.resolveBaseUrl();
      const res = await fetch(`${baseUrl}/api/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });

      if (res.ok) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} HTTP API reachable at ${baseUrl} (mode: sdk)`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} API returned ${res.status}`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} API not reachable${baseUrl ? ` at ${baseUrl}` : ""}`,
      };
    }
  }

  cancel(_abortController: AbortController): void {
    _abortController.abort();
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

// ── Managed Server ──────────────────────────────────────────────────────

/**
 * Owns a single long-lived `opencode serve` process. OpenCode is a server-first
 * harness: its HTTP API returns clean, structured assistant messages, whereas
 * a one-shot `opencode run` against a pipe intermittently drops its answer. We
 * therefore route every request through a managed server (an explicit
 * OPENCODE_URL operator override still wins).
 */
class OpenCodeServerManager {
  private proc: ReturnType<typeof Bun.spawn> | null = null;
  private baseUrl: string | null = null;
  private starting: Promise<string> | null = null;

  async ensure(
    resolveEnv: () => Promise<Record<string, string>>,
    log: (level: "info" | "error" | "debug", msg: string) => void,
  ): Promise<string> {
    const external = process.env["OPENCODE_URL"]?.trim();
    if (external) return external;
    if (this.baseUrl && this.proc && this.proc.exitCode === null) {
      return this.baseUrl;
    }
    if (this.starting) return this.starting;
    this.starting = this.start(resolveEnv, log).finally(() => {
      this.starting = null;
    });
    return this.starting;
  }

  private async healthy(baseUrl: string): Promise<boolean> {
    try {
      const res = await fetch(`${baseUrl}/api/health`, {
        signal: AbortSignal.timeout(2000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  private async start(
    resolveEnv: () => Promise<Record<string, string>>,
    log: (level: "info" | "error" | "debug", msg: string) => void,
  ): Promise<string> {
    const port = Number(process.env["OPENCODE_PORT"]) || DEFAULT_PORT;
    const baseUrl = `http://localhost:${port}`;
    if (await this.healthy(baseUrl)) {
      this.baseUrl = baseUrl;
      return baseUrl;
    }
    const env = await resolveEnv();
    log("info", `Starting managed opencode server on ${baseUrl}`);
    this.proc = Bun.spawn([CLI_BIN, "serve", "--port", String(port)], {
      env,
      stdin: "ignore",
      stdout: "ignore",
      stderr: "ignore",
    });
    for (let i = 0; i < 60; i += 1) {
      if (await this.healthy(baseUrl)) {
        this.baseUrl = baseUrl;
        return baseUrl;
      }
      await Bun.sleep(500);
    }
    throw new Error(
      `OpenCode server did not become healthy at ${baseUrl} within 30s`,
    );
  }

  shutdown(): void {
    try {
      this.proc?.kill();
    } catch {
      /* best effort */
    }
    this.proc = null;
    this.baseUrl = null;
  }
}

// ── Provider Implementation ─────────────────────────────────────────────

class OpenCodeProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private logger: BoundLogger | null = null;
  private adapter: ProviderAdapter | null = null;
  private readonly server = new OpenCodeServerManager();

  /** Resolve the OpenCode server base URL, starting a managed server if needed. */
  private async resolveServerBaseUrl(): Promise<string> {
    return this.server.ensure(
      () => this.resolveCliEnv(),
      (level, msg) => this.log(level, msg),
    );
  }

  /** Stop the managed OpenCode server (called on plugin shutdown). */
  shutdownServer(): void {
    this.server.shutdown();
  }

  /**
   * Lazily create (and cache) the server-side session for a local session. Done
   * at first prompt rather than at createSession() so the one-time server start
   * + DB migration is paid under the prompt's (long) timeout, not the create's.
   */
  private async ensureRemoteSession(
    session: ManagedSession,
    adapter: ProviderAdapter,
  ): Promise<string | undefined> {
    if (session.remoteSessionId) return session.remoteSessionId;
    if (!adapter.createRemoteSession) return undefined;
    const remoteId = await adapter.createRemoteSession(session.config);
    session.remoteSessionId = remoteId;
    this.log("debug", `Remote OpenCode session created: ${remoteId}`);
    return remoteId;
  }
  private currentMode: ProviderMode | null = null;

  setHostServices(hs: HostServices) {
    this.hostServices = hs;
    this.logger = new BoundLogger(hs.logger, `${PROVIDER_NAME}-provider`);
    const registry = new ProviderRegistry(hs);
    this.logIngester =
      registry.getProvider<LogIngester>("ai", "log-ingester") ?? null;
  }

  /**
   * OpenCode resolves provider credentials from its own auth store first and
   * env vars second. We point it at a scoped data dir (off the operator's
   * `~/.local/share/opencode`) so a stale/personal `opencode auth login` never
   * shadows a key the user saved in the agent config bag — then inject those
   * keys as the env vars OpenCode reads for each upstream provider.
   */
  private opencodeDataHome(): string {
    const home =
      process.env["OPENCODE_XDG_DATA_HOME"]?.trim() ||
      join(tmpdir(), "vibe-opencode-data");
    try {
      mkdirSync(home, { recursive: true });
    } catch {
      /* best effort */
    }
    return home;
  }

  /**
   * Build the env for an `opencode run` spawn: the agent env, a scoped
   * XDG_DATA_HOME, plus every upstream provider key resolved from env → the
   * agent config bag. OpenCode is a multi-provider harness, so we surface all
   * keys we can and let the chosen `provider/model` pick the right one.
   */
  private async resolveCliEnv(): Promise<Record<string, string>> {
    // Curate the env rather than spreading the whole agent process env: the
    // long-running daemon carries vars that make `opencode run` route its
    // answer through a TUI buffer (empty piped stdout). A minimal env — the
    // shell essentials + a scoped data dir + provider keys — matches a clean
    // shell invocation and makes the captured output deterministic.
    const env: Record<string, string> = {
      XDG_DATA_HOME: this.opencodeDataHome(),
      CI: "true",
    };
    for (const passthrough of [
      "HOME",
      "PATH",
      "USER",
      "LANG",
      "LC_ALL",
      "TMPDIR",
      "SHELL",
      "TERM",
    ]) {
      const v = process.env[passthrough];
      if (v) env[passthrough] = v;
    }
    // env var name -> config-bag key (same name here, but kept explicit).
    const keyNames = [
      "ANTHROPIC_API_KEY",
      "OPENAI_API_KEY",
      "OPENROUTER_API_KEY",
      "GEMINI_API_KEY",
      "GOOGLE_API_KEY",
      "GROQ_API_KEY",
      "MISTRAL_API_KEY",
      "DEEPSEEK_API_KEY",
      "OPENCODE_API_KEY",
    ];
    for (const name of keyNames) {
      const fromEnv = process.env[name]?.trim();
      if (fromEnv) {
        env[name] = fromEnv;
        continue;
      }
      if (this.hostServices?.getConfig) {
        try {
          const v = (await this.hostServices.getConfig(name))?.trim();
          if (v) env[name] = v;
        } catch {
          /* ignore a single key lookup failure */
        }
      }
    }
    return env;
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
      modelListing: true,
    };
  }

  async listModels(): Promise<AIModelInfo[]> {
    const adapter = this.getAdapter();
    if (adapter.listModels) return adapter.listModels();
    return fallbackModels();
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

    // The server-side session is created lazily on the first prompt (see
    // ensureRemoteSession) so createSession never blocks on the one-time
    // `opencode serve` start + DB migration.
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
      remoteSessionId: null,
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

    try {
      const remoteId = await this.ensureRemoteSession(session, adapter);
      const result = await adapter.sendPrompt(
        fullPrompt,
        model,
        session.config,
        remoteId,
      );
      const durationMs = Date.now() - startTime;

      if (result.remoteSessionId && !session.remoteSessionId) {
        session.remoteSessionId = result.remoteSessionId;
      }

      this.updateStats(session, result.inputTokens, result.outputTokens, model);
      session.status = "active";
      session.updatedAt = new Date().toISOString();

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

    try {
      const remoteId = await this.ensureRemoteSession(session, adapter);
      const result = await adapter.streamPrompt(
        fullPrompt,
        model,
        session.config,
        onChunk ?? (() => {}),
        remoteId,
      );
      const durationMs = Date.now() - startTime;

      if (result.remoteSessionId && !session.remoteSessionId) {
        session.remoteSessionId = result.remoteSessionId;
      }

      this.updateStats(session, result.inputTokens, result.outputTokens, model);
      session.status = "active";
      session.abortController = null;
      session.updatedAt = new Date().toISOString();

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

      // Clean up the managed-server remote session.
      if (session.remoteSessionId) {
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

  // ── `vibe ai run` / `vibe ai sdk` integration ────────────────────────

  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null {
    const env: Record<string, string> = {};
    const port = process.env["OPENCODE_PORT"]?.trim();
    const url = process.env["OPENCODE_URL"]?.trim();
    if (port) env["OPENCODE_PORT"] = port;
    if (url) env["OPENCODE_URL"] = url;
    return { binary: CLI_COMMAND, env };
  }

  async sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }> {
    const adapter = new OpenCodeSdkAdapter(() => this.resolveServerBaseUrl());
    const model = opts.model ?? DEFAULT_MODEL;
    const config: AISessionConfig = {
      name: "vibe-ai-sdk",
      agentType: PROVIDER_NAME,
      model,
      maxTokens: opts.maxTokens,
      providerConfig: opts.extras,
    };
    const remoteSessionId = await adapter.createRemoteSession(config);
    const result = await adapter.sendPrompt(
      opts.prompt,
      model,
      config,
      remoteSessionId,
    );
    return {
      text: result.content,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        model,
      },
    };
  }

  // ── Private Helpers ─────────────────────────────────────────────────

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;

    // OpenCode is server-first: both "sdk" and "cli" mode route through the
    // managed `opencode serve` HTTP API, which returns clean structured
    // responses. (A one-shot `opencode run` against a pipe intermittently drops
    // its answer, so we do not use it.) The provider's mode is retained for the
    // UI but does not change the transport here.
    this.adapter = new OpenCodeSdkAdapter(() => this.resolveServerBaseUrl());
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
      // Cross-platform binary discovery via Bun.which (handles PATHEXT on Windows).
      if (Bun.which(CLI_COMMAND, { PATH: process.env.PATH })) {
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
    this.logger?.[level](msg);
  }
}

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([CLI_BIN, "--version"], {
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

const PLUGIN_NAME = "opencode";
const PLUGIN_VERSION = "1.0.0";

const provider = new OpenCodeProvider();

const lifecycle = createLifecycleHooks({
  name: PLUGIN_NAME,
  telemetryEventName: "ai.provider.ready",
  onInit: (hostServices: HostServices) => {
    provider.setHostServices(hostServices);
    new TelemetryEmitter(PLUGIN_NAME, PLUGIN_VERSION, hostServices).emit(
      "ai.provider.ready",
      { provider: PLUGIN_NAME },
    );
  },
  onShutdown: () => {
    for (const [id] of (provider as OpenCodeProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
    provider.shutdownServer();
  },
});

type OpenCodeVibePlugin = VibePlugin & {
  providers?: { ai?: AIAgentProvider };
};

export const createPlugin = (_ctx: ProfileContext): OpenCodeVibePlugin => ({
  capabilities: {
    secrets: "read",
    subprocess: true,
    gateway: false,
    telemetry: true,
  },
  name: PLUGIN_NAME,
  version: PLUGIN_VERSION,
  description:
    "OpenCode AI agent provider for VibeControls (SDK + CLI dual-mode)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: "npm",
      requiresSudo: false,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),
  onServerStart: lifecycle.onServerStart,
  onServerStop: lifecycle.onServerStop,
});
