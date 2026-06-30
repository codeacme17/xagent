/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
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
    apiRequestMock.mockImplementation((url: string) => {
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

    expect(await screen.findByText("triggers.cards.gmail.title")).toBeInTheDocument()
    expect(screen.queryByText("triggers.cards.appWidget.title")).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("triggers.cards.gmail.title"))

    expect(await screen.findByLabelText("triggers.form.watchLabel")).toHaveValue("INBOX")
    expect(screen.getByText("triggers.form.watchLabelHelp")).toBeInTheDocument()
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

})
