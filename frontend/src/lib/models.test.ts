import { describe, expect, it } from "vitest"
import { hostnameFromUrl } from "./models"

describe("hostnameFromUrl", () => {
  it("returns empty string for missing url", () => {
    expect(hostnameFromUrl(undefined)).toBe("")
    expect(hostnameFromUrl(null)).toBe("")
    expect(hostnameFromUrl("")).toBe("")
  })

  it("keeps non-standard ports so same-host models stay distinguishable", () => {
    expect(hostnameFromUrl("http://localhost:9997/v1")).toBe("localhost:9997")
    expect(hostnameFromUrl("http://localhost:11434")).toBe("localhost:11434")
  })

  it("drops default ports per URL semantics", () => {
    expect(hostnameFromUrl("https://api.openai.com/v1")).toBe("api.openai.com")
  })

  it("returns empty string for non-absolute urls instead of a garbled fallback", () => {
    expect(hostnameFromUrl("/v1")).toBe("")
    expect(hostnameFromUrl("not a url")).toBe("")
  })
})
