import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) => {
      const labels: Record<string, string> = {
        "conversationLogs.title": "Conversation Logs",
        "conversationLogs.logs": "logs",
        "conversationLogs.searchPlaceholder": "Search conversations",
        "conversationLogs.agentFilterAll": "All agents",
        "conversationLogs.readOnly": "Read-only",
        "conversationLogs.empty.title": "No conversation logs yet",
        "conversationLogs.empty.filteredTitle": "No matching logs",
        "conversationLogs.sources.all": "All",
        "conversationLogs.sources.widget": "Widget",
        "conversationLogs.sources.restApi": "REST API",
        "conversationLogs.sources.sharedLink": "Shareable Link",
        "conversationLogs.sources.webhook": "Webhook",
      }
      if (key === "conversationLogs.pagination.summary" && vars) {
        return `Showing ${vars.from}-${vars.to} of ${vars.total}`
      }
      return labels[key] || key
    },
  }),
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Bot: Icon,
    CalendarClock: Icon,
    ChevronLeft: Icon,
    ChevronRight: Icon,
    Clock3: Icon,
    Filter: Icon,
    Inbox: Icon,
    MessageSquare: Icon,
    MessageSquareText: Icon,
    Search: Icon,
    ShieldCheck: Icon,
    Tags: Icon,
  }
})

import { ConversationLogsPage } from "./conversation-logs"

const listPayload = {
  logs: [
    {
      task_id: 101,
      title: "REST lead intake",
      description: "Lead from REST",
      status: "completed",
      source: "rest_api",
      source_label: "REST API",
      agent_id: 7,
      agent_name: "Sales Agent",
      created_at: "2026-06-29T01:00:00Z",
      updated_at: "2026-06-29T01:01:00Z",
      last_activity_at: "2026-06-29T01:01:00Z",
      total_tokens: 12,
      message_count: 2,
    },
    {
      task_id: 202,
      title: "Webhook CRM event",
      description: "Webhook payload",
      status: "completed",
      source: "webhook",
      source_label: "Webhook",
      agent_id: 7,
      agent_name: "Sales Agent",
      created_at: "2026-06-29T02:00:00Z",
      updated_at: "2026-06-29T02:01:00Z",
      last_activity_at: "2026-06-29T02:01:00Z",
      total_tokens: 20,
      message_count: 2,
    },
  ],
  source_counts: {
    all: 2,
    widget: 0,
    rest_api: 1,
    shared_link: 0,
    webhook: 1,
  },
  agents: [
    {
      agent_id: 7,
      agent_name: "Sales Agent",
      agent_logo_url: null,
    },
  ],
  pagination: {
    page: 1,
    per_page: 20,
    total: 2,
    total_pages: 1,
  },
}

const detailPayload = {
  log: listPayload.logs[0],
  transcript: [
    {
      id: 1,
      role: "user",
      content: "Qualify this lead",
      message_type: "chat",
      created_at: "2026-06-29T01:00:10Z",
    },
    {
      id: 2,
      role: "assistant",
      content: "Lead qualified",
      message_type: "chat",
      created_at: "2026-06-29T01:00:20Z",
    },
  ],
  metadata: {
    task: {
      task_id: 101,
      input: "Qualify this lead",
      output: "Lead qualified",
      error_message: null,
      description: "Lead from REST",
      agent_config: {},
    },
    trigger: null,
    public_context: null,
  },
  read_only: true,
}

describe("ConversationLogsPage", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation((url: string) => {
      const parsed = new URL(url)
      if (parsed.pathname === "/api/conversation-logs/101") {
        return Promise.resolve(
          new Response(JSON.stringify(detailPayload), { status: 200 })
        )
      }
      if (parsed.pathname === "/api/conversation-logs") {
        return Promise.resolve(
          new Response(JSON.stringify(listPayload), { status: 200 })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("loads logs, source tabs, and a read-only selected transcript", async () => {
    render(<ConversationLogsPage />)

    expect(await screen.findByText("Conversation Logs")).toBeInTheDocument()
    expect((await screen.findAllByText("Sales Agent")).length).toBeGreaterThan(0)
    expect(screen.getByText("All 2")).toBeInTheDocument()
    expect(screen.getByText("REST API 1")).toBeInTheDocument()
    expect(screen.getByText("Webhook 1")).toBeInTheDocument()
    expect(await screen.findByText("Qualify this lead")).toBeInTheDocument()
    expect(screen.getByText("Lead qualified")).toBeInTheDocument()
    expect(screen.getByText("Read-only")).toBeInTheDocument()
    expect(screen.queryByText("Upload")).not.toBeInTheDocument()
    expect(screen.queryByText("Send")).not.toBeInTheDocument()
  })

  it("sends source tab and search state to the list API", async () => {
    render(<ConversationLogsPage />)

    await screen.findAllByText("Sales Agent")
    fireEvent.click(screen.getByText("Webhook 1"))
    fireEvent.change(screen.getByPlaceholderText("Search conversations"), {
      target: { value: "crm" },
    })

    await waitFor(() => {
      const listUrls = apiRequestMock.mock.calls
        .map(([url]) => String(url))
        .filter((url) => new URL(url).pathname === "/api/conversation-logs")
      expect(listUrls.some((url) => url.includes("source=webhook"))).toBe(true)
      expect(listUrls.some((url) => url.includes("search=crm"))).toBe(true)
    })
  })
})
