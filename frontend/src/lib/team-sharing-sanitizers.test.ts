import { describe, expect, it } from "vitest"

import {
  sanitizeAppIntegrations,
  sanitizeUnsharedConnectors,
  sanitizeUnsharedKnowledgeBases,
} from "./team-sharing-sanitizers"

describe("team sharing response sanitizers", () => {
  it("keeps only well-formed unshared connectors", () => {
    expect(
      sanitizeUnsharedConnectors([
        { type: "mcp", id: 1, name: "GitHub" },
        { type: "mcp", name: "Missing", reason: "unresolved" },
        null,
        "bad",
        { type: "mcp", id: null, name: "Missing id" },
      ]),
    ).toEqual([
      { type: "mcp", id: 1, name: "GitHub" },
      { type: "mcp", name: "Missing", reason: "unresolved" },
    ])
  })

  it("keeps only well-formed unshared knowledge bases", () => {
    expect(
      sanitizeUnsharedKnowledgeBases([{ name: "support" }, null, { name: 42 }, "bad"]),
    ).toEqual([{ name: "support" }])
  })

  it("rejects malformed app integration payloads", () => {
    const app = {
      id: "github",
      name: "GitHub",
      description: "GitHub connector",
      icon: "github",
      server_id: 7,
    }
    expect(sanitizeAppIntegrations([app, null, 1, { id: "missing-fields" }])).toEqual([app])
    expect(sanitizeAppIntegrations({ apps: [app] })).toEqual([])
  })
})
