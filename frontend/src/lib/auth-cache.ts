export const AUTH_CACHE_KEY = "auth_cache"
export const AUTH_TOKEN_UPDATED_EVENT = "auth-token-updated"

export interface AuthUser {
  id: string | number
  username: string
  is_admin?: boolean
}

export interface AuthTokenPayload {
  user: AuthUser
  access_token: string
  refresh_token?: string
  expires_in?: number
  refresh_expires_in?: number
}

export function storeAuthTokenPayload(data: AuthTokenPayload) {
  const userData = {
    id: data.user.id,
    username: data.user.username,
    is_admin: data.user.is_admin,
  }

  localStorage.setItem("auth_token", data.access_token)
  localStorage.setItem("auth_user", JSON.stringify(userData))
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    user: userData,
    token: data.access_token,
    refreshToken: data.refresh_token,
    expiresAt: Date.now() + (data.expires_in || 1800) * 1000,
    refreshExpiresAt: Date.now() + (data.refresh_expires_in || 604800) * 1000,
    timestamp: Date.now(),
  }))

  window.dispatchEvent(new StorageEvent(AUTH_TOKEN_UPDATED_EVENT, {
    key: AUTH_CACHE_KEY,
    newValue: localStorage.getItem(AUTH_CACHE_KEY),
  }))
}

export function clearAuthTokenPayload() {
  localStorage.removeItem("auth_token")
  localStorage.removeItem("auth_user")
  localStorage.removeItem(AUTH_CACHE_KEY)
}
