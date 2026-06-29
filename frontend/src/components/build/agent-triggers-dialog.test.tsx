/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => {
  return (key: string, vars?: Record<string, string | number>) => {
    if (vars?.count) return `${key}:${vars.count}`
    return key
  }
})

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
}))

import { AgentTriggersDialog } from "./agent-triggers-dialog"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("AgentTriggersDialog", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    apiRequestMock.mockImplementation((url: string, options?: { method?: string; body?: string }) => {
      if (url === "http://api.local/api/agents/42/triggers") {
        return Promise.resolve(
          jsonResponse([
            {
              id: 9,
              user_id: 1,
              agent_id: 42,
              type: "gmail",
              name: "Support inbox",
              enabled: true,
              config: {
                watch_label: "INBOX",
                sender_filter: "boss@company.com",
                subject_keyword: "urgent",
              },
              prompt_template: "Reply to {{payload}}",
              webhook_token: null,
              webhook_secret: null,
              next_run_at: null,
              last_run_at: null,
              last_error: null,
              created_at: null,
              updated_at: null,
            },
          ]),
        )
      }
      if (url === "http://api.local/api/agents/42/triggers/9/runs") {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "PUT") {
        return Promise.resolve(
          jsonResponse({
            id: 42,
            widget_enabled: true,
            allowed_domains: ["example.com", "docs.example.com"],
          }),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("renders an existing Gmail trigger with its filters", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        agentName="Inbox Agent"
        open
        onOpenChange={vi.fn()}
        gmailConnection={{
          isConnected: true,
          connectedAccount: "gerard.santos@gmail.com",
        }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))

    expect(await screen.findByLabelText("triggers.form.watchLabel")).toHaveValue("INBOX")
    expect(screen.getByLabelText("triggers.form.senderFilter")).toHaveValue("boss@company.com")
    expect(screen.getByLabelText("triggers.form.subjectKeyword")).toHaveValue("urgent")
    expect(screen.getByText("triggers.gmail.connected")).toBeInTheDocument()
    expect(screen.getByText("gerard.santos@gmail.com")).toBeInTheDocument()
  })

  it("prompts for Gmail connection when the connector is missing", async () => {
    const onConnectGmail = vi.fn()

    render(
      <AgentTriggersDialog
        agentId={42}
        agentName="Inbox Agent"
        open
        onOpenChange={vi.fn()}
        initialType="gmail"
        gmailConnection={{
          isConnected: false,
          connectedAccount: null,
        }}
        onConnectGmail={onConnectGmail}
      />,
    )

    expect(await screen.findByText("triggers.gmail.notConnected")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "triggers.gmail.connect" }))

    expect(onConnectGmail).toHaveBeenCalledTimes(1)
  })

  it("renders App Widget settings and updates allowed domains", async () => {
    const onWidgetConfigUpdated = vi.fn()

    render(
      <AgentTriggersDialog
        agentId={42}
        agentName="Inbox Agent"
        open
        onOpenChange={vi.fn()}
        appWidget={{
          widget_enabled: true,
          allowed_domains: ["example.com"],
        }}
        onWidgetConfigUpdated={onWidgetConfigUpdated}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.appWidget.title"))

    expect(await screen.findByText("example.com")).toBeInTheDocument()
    expect(screen.getByText((content) => content.includes("widget.js"))).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText("triggers.appWidget.domainPlaceholder"), {
      target: { value: "docs.example.com" },
    })
    fireEvent.click(screen.getByRole("button", { name: "triggers.appWidget.addDomain" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          allowed_domains: ["example.com", "docs.example.com"],
        }),
      })
    })
    expect(onWidgetConfigUpdated).toHaveBeenCalledWith(
      expect.objectContaining({
        widget_enabled: true,
        allowed_domains: ["example.com", "docs.example.com"],
      }),
    )
  })
})
