import { describe, expect, it, vi } from "vitest"
import {
  AUTH_CACHE_KEY,
  clearAuthTokenPayload,
  storeAuthTokenPayload,
} from "./auth-cache"

describe("auth cache helpers", () => {
  it("stores the same token payload shape used by password login", () => {
    const dispatch = vi.spyOn(window, "dispatchEvent")

    storeAuthTokenPayload({
      user: { id: 42, username: "person@example.com", is_admin: false },
      access_token: "access-token",
      refresh_token: "refresh-token",
      expires_in: 120,
      refresh_expires_in: 240,
    })

    expect(localStorage.getItem("auth_token")).toBe("access-token")
    expect(JSON.parse(localStorage.getItem("auth_user") || "{}")).toEqual({
      id: 42,
      username: "person@example.com",
      is_admin: false,
    })

    const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    expect(cache.user.username).toBe("person@example.com")
    expect(cache.token).toBe("access-token")
    expect(cache.refreshToken).toBe("refresh-token")
    expect(dispatch).toHaveBeenCalled()
  })

  it("clears legacy and current auth cache keys", () => {
    localStorage.setItem("auth_token", "access-token")
    localStorage.setItem("auth_user", "{}")
    localStorage.setItem(AUTH_CACHE_KEY, "{}")

    clearAuthTokenPayload()

    expect(localStorage.getItem("auth_token")).toBeNull()
    expect(localStorage.getItem("auth_user")).toBeNull()
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })
})
