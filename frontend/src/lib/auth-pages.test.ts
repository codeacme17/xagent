import { describe, expect, it } from "vitest"
import { isAuthPublicPath } from "./auth-pages"

describe("auth public paths", () => {
  it("allows the OIDC callback route through the auth guard", () => {
    expect(isAuthPublicPath("/auth/oidc/callback")).toBe(true)
  })
})
