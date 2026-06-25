import { describe, expect, it } from "vitest"

import { findMatchingMcpApp, findMatchingMcpServer, mcpNameMatches } from "./mcp-lookup"

describe("MCP lookup helpers", () => {
  it("matches saved app ids to connected server names", () => {
    const server = findMatchingMcpServer(
      [{ name: "Outlook", app_id: "outlook" }],
      "outlook"
    )

    expect(server?.name).toBe("Outlook")
  })

  it("matches slug ids to display names", () => {
    const app = findMatchingMcpApp(
      [{ id: "google-drive", name: "Google Drive" }],
      "Google Drive"
    )

    expect(app?.id).toBe("google-drive")
    expect(mcpNameMatches("google-drive", "Google Drive")).toBe(true)
  })
})
