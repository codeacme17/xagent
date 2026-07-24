"use client"

import React, { useCallback, useEffect, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { AppProvider, useApp, type AppProviderTransportConfig } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { uploadPublicChatFile } from "@/lib/public-chat-file-upload"
import { normalizeTaskStatus } from "@/lib/task-status"
import {
  getApiUrl,
  getFilePublicDownloadUrl,
  getFilePublicPreviewUrl,
  setPublicAccessToken,
} from "@/lib/utils"

interface PublicAgentChatPageProps {
  authMode: "widget" | "share"
  routeToken: string
  guestId?: string | null
  searchAgentId?: number | null
  embedTicket?: string | null
  widgetKey?: string | null
}

type PublicAuthResult = {
  access_token: string
  agent_id?: number | null
  agent_name?: string | null
  agent_logo?: string | null
  agent_description?: string | null
  suggested_prompts?: string[] | null
  // Set instead of agent_id when the share token exposes a workforce.
  workforce_id?: number | null
}

interface PublicConversationContentProps {
  authMode: "widget" | "share"
  routeToken: string
  normalizedGuestId?: string | null
  accessToken: string
  agentId: number | null
  workforceId: number | null
  agentName: string | null
  agentLogo: string | null
  agentDescription: string | null
  suggestedPrompts: string[]
  onAuthInvalidated?: () => void
}

// WS-close reasons that mean "this task isn't yours" rather than a transport
// failure — used to distinguish a per-guest access denial (recoverable by
// starting a fresh session) from a generic connection drop (must not wipe the
// session). #973. Scoped to the guest-mismatch case only: "Share link is
// unavailable" is emitted by many non-recoverable causes (owner disabled the
// link, unpublished agent/workforce, channel mismatch), so treating it as
// recoverable would trigger a pointless clear + re-auth round-trip that still
// lands on the terminal error. "Access denied for this guest" is the backend
// HTTPException.detail surfaced as event.reason on a 4003 close; "Access
// denied" is use-websocket.ts's fallback when a 4003 carries no reason.
const SHARE_ACCESS_DENIED_REASONS = new Set([
  "Access denied for this guest",
  "Access denied",
])

type PublicMessageConfig = Record<string, unknown>

function PublicConversationContent({
  authMode,
  routeToken,
  normalizedGuestId,
  accessToken,
  agentId,
  workforceId,
  agentName,
  agentLogo,
  agentDescription,
  suggestedPrompts,
  onAuthInvalidated,
}: PublicConversationContentProps) {
  const { state, dispatch, sendMessage, setTaskId, connectionError } = useApp()
  const { t } = useI18n()
  const [createTaskError, setCreateTaskError] = useState<string | null>(null)
  const [draftMessage, setDraftMessage] = useState("")
  const [draftFiles, setDraftFiles] = useState<File[]>([])
  const [isBootstrappingTask, setIsBootstrappingTask] = useState(false)
  const [hasResolvedStoredTask, setHasResolvedStoredTask] = useState(false)
  const storageKey = authMode === "share"
    ? `${authMode}_task_${routeToken}_${agentId ?? "anonymous"}`
    : `${authMode}_task_${agentId ?? "anonymous"}_${normalizedGuestId ?? "anonymous"}`
  const publicApiPrefix = authMode === "share" ? "/api/share" : "/api/widget"

  useEffect(() => {
    setHasResolvedStoredTask(false)
    const savedTaskId = localStorage.getItem(storageKey)
    if (!savedTaskId) {
      setTaskId(null, { navigate: false })
      setHasResolvedStoredTask(true)
      return
    }

    const parsedTaskId = parseInt(savedTaskId, 10)
    if (Number.isNaN(parsedTaskId)) {
      setTaskId(null, { navigate: false })
      setHasResolvedStoredTask(true)
      return
    }

    setTaskId(parsedTaskId, { navigate: false })
    setHasResolvedStoredTask(true)
  }, [setTaskId, storageKey])

  useEffect(() => {
    if (!hasResolvedStoredTask) {
      return
    }

    if (state.taskId) {
      localStorage.setItem(storageKey, state.taskId.toString())
      return
    }

    localStorage.removeItem(storageKey)
  }, [hasResolvedStoredTask, state.taskId, storageKey])

  // Recover a returning share visitor whose persisted taskId belongs to a
  // different guest_id (e.g. a pre-#973 task created before per-guest
  // isolation): the WS connect closes 4003 with an access-denied reason. Drop
  // the stale taskId and fall back to the start screen so the visitor opens a
  // fresh session under their current guest token — instead of being stuck on
  // an error. Scoped to access-denied reasons so a transient transport drop
  // never wipes a live session.
  useEffect(() => {
    if (authMode !== "share" || !connectionError || !state.taskId) {
      return
    }
    if (!SHARE_ACCESS_DENIED_REASONS.has(connectionError.message)) {
      return
    }
    try {
      localStorage.removeItem(storageKey)
    } catch {
      // Non-fatal: localStorage may be unavailable (private mode / sandboxed
      // iframe); the reset below still recovers the session.
    }
    setTaskId(null, { navigate: false })
  }, [authMode, connectionError, state.taskId, storageKey, setTaskId])

  const handleSend = useCallback(async (
    message: string,
    config?: PublicMessageConfig,
    files?: File[],
  ) => {
    if (state.taskId) {
      await sendMessage(message, config, files)
      return
    }

    // For a workforce share the first turn starts inside task creation, which
    // rejects an empty message server-side (400) — AFTER any files uploaded
    // above would already be orphaned. Guard the empty case here so files are
    // never uploaded for a turn that cannot start.
    if (workforceId && !message.trim()) {
      return
    }

    setIsBootstrappingTask(true)
    try {
      const taskPayload: Record<string, string | number | string[]> = {
        title: message,
        description: message,
      }
      if (agentId) {
        taskPayload.agent_id = agentId
      }

      setCreateTaskError(null)

      // Workforce shares start their first turn inside task creation, so any
      // opening-message attachments must be uploaded (task-lessly — no task
      // exists yet) and threaded in as file ids BEFORE the run begins;
      // otherwise the first turn never sees them.
      if (workforceId && files?.length) {
        const uploaded = await Promise.all(files.map((file) => uploadPublicChatFile({
          url: `${getApiUrl()}${publicApiPrefix}/files/upload`,
          accessToken,
          file,
          taskType: "task",
          fallbackError: t("files.uploadFailed"),
        })))
        taskPayload.files = uploaded.map((item) => item.file_id)
      }

      const response = await fetch(`${getApiUrl()}${publicApiPrefix}/chat/task/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${accessToken}`,
        },
        body: JSON.stringify(taskPayload),
      })

      if (!response.ok) {
        // A share task-create carries no task id, so 401/403 here means the
        // guest token itself is no longer valid (rotated/disabled link, or a
        // legacy token rejected post-#973). Drop the persisted token and force
        // a fresh auth rather than leaving the visitor on a dead session.
        if (authMode === "share" && (response.status === 401 || response.status === 403)) {
          try {
            localStorage.removeItem(storageKey)
          } catch {
            // Non-fatal: still force a fresh auth below even if localStorage
            // is unavailable (private mode / sandboxed iframe).
          }
          onAuthInvalidated?.()
        }
        const errorData = await response.json().catch(() => null)
        const errorMessage = errorData?.detail || t("widgetChat.messages.error_init")
        setCreateTaskError(errorMessage)
        throw new Error(errorMessage)
      }

      const taskData = await response.json()
      const newTaskId = taskData.task_id
      if (typeof newTaskId !== "number") {
        throw new Error("Task creation failed")
      }

      setTaskId(newTaskId, { navigate: false })
      dispatch({
        type: "SET_CURRENT_TASK",
        payload: {
          id: newTaskId.toString(),
          title: taskData.title || message,
          status: normalizeTaskStatus(taskData.status) || "pending",
          description: taskData.description || message,
          createdAt: taskData.created_at || new Date().toISOString(),
          updatedAt:
            taskData.updated_at
            || taskData.created_at
            || new Date().toISOString(),
          agentId: taskData.agent_id ?? agentId ?? undefined,
          agentName: taskData.agent_name || agentName || undefined,
          agentLogoUrl: taskData.agent_logo_url || agentLogo || undefined,
        },
      })

      if (!workforceId) {
        await sendMessage(message, { ...config, targetTaskId: newTaskId }, files)
      }
      // Workforce share sessions already started their first turn (with the
      // files threaded in above) inside task creation — the connection
      // replays it from history, so re-sending over the websocket would
      // duplicate the turn.
      setDraftMessage("")
      setDraftFiles([])
    } catch (error) {
      setIsBootstrappingTask(false)
      throw error
    }
  }, [accessToken, agentId, agentLogo, agentName, authMode, dispatch, onAuthInvalidated, publicApiPrefix, sendMessage, setTaskId, state.taskId, storageKey, t, workforceId])

  useEffect(() => {
    if (state.taskId || createTaskError) {
      setIsBootstrappingTask(false)
    }
  }, [createTaskError, state.taskId])

  const resolvedAgentName = state.currentTask?.agentName || agentName || t("widgetChat.title")
  const resolvedAgentLogo = state.currentTask?.agentLogoUrl || agentLogo || null
  const shouldShowStartScreen = !state.taskId && hasResolvedStoredTask
  const showStatus = !createTaskError
  const statusText = state.isHistoryLoading
    ? t("widgetChat.status.initializing")
    : state.currentTask?.status === "running" || state.isProcessing || isBootstrappingTask
      ? t("widgetChat.status.connecting")
      : t("widgetChat.status.online")

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="flex-none p-4 border-b bg-card text-card-foreground shadow-sm z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary overflow-hidden">
            {resolvedAgentLogo ? (
              <img
                src={resolvedAgentLogo.startsWith("http") ? resolvedAgentLogo : `${getApiUrl()}${resolvedAgentLogo.startsWith("/") ? "" : "/"}${resolvedAgentLogo}`}
                alt={resolvedAgentName}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-5 h-5 rounded-full bg-primary/20" />
            )}
          </div>
          <div>
            <h1 className="text-sm font-semibold">{resolvedAgentName}</h1>
            {showStatus ? (
              <p className="text-xs text-muted-foreground">{statusText}</p>
            ) : (
              <p className="text-xs text-destructive">{createTaskError}</p>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        {!hasResolvedStoredTask && !state.taskId ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : shouldShowStartScreen ? (
          <div className="h-full overflow-y-auto">
            <main className="container max-w-4xl mx-auto px-4 py-8">
              <ChatStartScreen
                title={resolvedAgentName}
                description={agentDescription || undefined}
                prompts={suggestedPrompts}
                onSend={(message, files, config) => handleSend(message, config, files)}
                isSending={isBootstrappingTask || state.isProcessing}
                inputValue={draftMessage}
                onInputChange={setDraftMessage}
                files={draftFiles}
                onFilesChange={setDraftFiles}
                readOnlyConfig={true}
                hideConfig={true}
                compactInput={true}
                deferFileUpload={true}
                autoFocus={true}
                inputMinHeightClass="min-h-[44px]"
              />
            </main>
          </div>
        ) : (
          <TaskConversationPanel
            mode="page"
            showTaskActions={false}
            showTokenUsage={false}
            showDagPreview={false}
            showTaskFiles={false}
            hideFileUpload={false}
            hideConfig={true}
            compactInput={true}
            deferFileUpload={true}
            onSend={handleSend}
          />
        )}
      </div>
    </div>
  )
}

export function PublicAgentChatPage({
  authMode,
  routeToken,
  guestId,
  searchAgentId = null,
  embedTicket = null,
  widgetKey = null,
}: PublicAgentChatPageProps) {
  const { t } = useI18n()
  const normalizedGuestId = authMode === "widget" ? (guestId || "anonymous") : null
  const [isInitializing, setIsInitializing] = useState(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [authResult, setAuthResult] = useState<PublicAuthResult | null>(null)
  // Bumped to force a fresh /api/share/auth when a persisted guest token turns
  // out to be invalid (see onAuthInvalidated below).
  const [reauthNonce, setReauthNonce] = useState(0)

  // The guest token is minted server-side and carries the per-guest isolation
  // credential (#973). Persist it per share link and REUSE it across reloads
  // instead of re-authing every mount: re-authing would mint a new guest_id
  // each time, so a returning visitor's own tasks would fail the per-guest
  // check. Widget keeps its old behavior (its guest_id is client-supplied).
  const shareAuthStorageKey = authMode === "share" ? `share_auth_${routeToken}` : null

  const onAuthInvalidated = useCallback(() => {
    if (shareAuthStorageKey) {
      try {
        localStorage.removeItem(shareAuthStorageKey)
      } catch {
        // Non-fatal: the state resets below re-auth regardless of whether
        // localStorage is available (private mode / sandboxed iframe).
      }
    }
    setAuthResult(null)
    setPublicAccessToken(null)
    setIsInitializing(true)
    setReauthNonce((n) => n + 1)
  }, [shareAuthStorageKey])

  useEffect(() => {
    const persistShareAuth = (data: PublicAuthResult) => {
      if (!shareAuthStorageKey) {
        return
      }
      try {
        localStorage.setItem(shareAuthStorageKey, JSON.stringify(data))
      } catch {
        // Non-fatal: without persistence the visitor simply re-auths (and gets
        // a new guest session) on the next reload.
      }
    }

    const readPersistedShareAuth = (): PublicAuthResult | null => {
      if (!shareAuthStorageKey) {
        return null
      }
      try {
        const raw = localStorage.getItem(shareAuthStorageKey)
        if (!raw) {
          return null
        }
        const parsed: unknown = JSON.parse(raw)
        // Reject anything that isn't a well-shaped auth blob so a corrupt or
        // cross-version localStorage entry falls back to a clean re-auth rather
        // than flowing malformed values downstream. agent_id/workforce_id are
        // display/routing hints (never the isolation credential — that lives in
        // the signed guest JWT), but keep their optional-number contract intact.
        const isNullableNumber = (value: unknown) =>
          value === undefined || value === null || typeof value === "number"
        if (
          !parsed
          || typeof parsed !== "object"
          || typeof (parsed as PublicAuthResult).access_token !== "string"
          || !(parsed as PublicAuthResult).access_token
          || !isNullableNumber((parsed as PublicAuthResult).agent_id)
          || !isNullableNumber((parsed as PublicAuthResult).workforce_id)
        ) {
          return null
        }
        return parsed as PublicAuthResult
      } catch {
        return null
      }
    }

    const initPublicChat = async () => {
      try {
        const persisted = readPersistedShareAuth()
        if (persisted) {
          setAuthResult(persisted)
          setPublicAccessToken(persisted.access_token ?? null)
          setErrorMessage(null)
          return
        }

        const authPath = authMode === "share" ? "/api/share/auth" : "/api/widget/auth"
        const authPayload = authMode === "share"
          ? { share_token: routeToken }
          : {
              guest_id: normalizedGuestId,
              agent_id: searchAgentId,
              embed_ticket: embedTicket || undefined,
              // Direct visits (no embed ticket) authenticate with the widget
              // key carried in the opened URL.
              widget_key: embedTicket ? undefined : widgetKey || undefined,
            }

        const authResponse = await fetch(`${getApiUrl()}${authPath}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(authPayload),
        })

        if (!authResponse.ok) {
          const errorData = await authResponse.json().catch(() => null)
          throw new Error(errorData?.detail || "Widget authentication failed")
        }

        const authData = await authResponse.json()
        persistShareAuth(authData)
        setAuthResult(authData)
        setPublicAccessToken(authData.access_token ?? null)
        setErrorMessage(null)
      } catch (error) {
        console.error(error)
        setErrorMessage((error as Error).message || t("widgetChat.messages.error_init"))
        setPublicAccessToken(null)
      } finally {
        setIsInitializing(false)
      }
    }

    initPublicChat()
    return () => setPublicAccessToken(null)
  }, [authMode, embedTicket, widgetKey, normalizedGuestId, routeToken, searchAgentId, shareAuthStorageKey, reauthNonce, t])

  const publicAccessToken = authResult?.access_token ?? ""

  const transport = useMemo<AppProviderTransportConfig>(() => ({
    buildWebSocketUrl: ({ baseUrl, taskId, token }) =>
      `${baseUrl}/${authMode === "share" ? "api/share" : "api/widget"}/chat/ws/${taskId}${token ? `?token=${token}` : ""}`,
    buildFilePreviewUrl: ({ baseUrl, fileId }) =>
      getFilePublicPreviewUrl(fileId, baseUrl),
    buildFileDownloadUrl: ({ baseUrl, fileId }) =>
      getFilePublicDownloadUrl(fileId, baseUrl),
    uploadFiles: (files, params) =>
      Promise.all(files.map((file) =>
        uploadPublicChatFile({
          url: `${getApiUrl()}/${authMode === "share" ? "api/share" : "api/widget"}/files/upload`,
          accessToken: publicAccessToken,
          file,
          taskType: params.taskType,
          taskId: params.taskId,
          fallbackError: t("files.uploadFailed"),
        }),
      )),
  }), [authMode, publicAccessToken, t])

  if (isInitializing) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!authResult || errorMessage) {
    return (
      <div className="h-screen flex items-center justify-center bg-background p-6">
        <div className="max-w-md text-center text-sm text-muted-foreground">
          {errorMessage || t("widgetChat.messages.error_init")}
        </div>
      </div>
    )
  }

  const resolvedAuthResult = authResult

  return (
    <AppProvider token={publicAccessToken} transport={transport}>
      <PublicConversationContent
        authMode={authMode}
        routeToken={routeToken}
        normalizedGuestId={normalizedGuestId}
        accessToken={resolvedAuthResult.access_token}
        agentId={resolvedAuthResult.agent_id ?? searchAgentId ?? null}
        workforceId={resolvedAuthResult.workforce_id ?? null}
        agentName={resolvedAuthResult.agent_name ?? null}
        agentLogo={resolvedAuthResult.agent_logo ?? null}
        agentDescription={resolvedAuthResult.agent_description ?? null}
        suggestedPrompts={resolvedAuthResult.suggested_prompts ?? []}
        onAuthInvalidated={onAuthInvalidated}
      />
    </AppProvider>
  )
}
