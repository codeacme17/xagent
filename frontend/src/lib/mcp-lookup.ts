export interface McpLookupReference {
  name?: unknown
  id?: unknown
  app_id?: unknown
}

const normalizeMcpLookupValue = (value: unknown): string => {
  return String(value ?? "").trim().toLowerCase()
}

const getMcpLookupKeys = (...values: unknown[]): Set<string> => {
  const keys = new Set<string>()

  values.forEach((value) => {
    const normalized = normalizeMcpLookupValue(value)
    if (!normalized) return

    keys.add(normalized)
    keys.add(normalized.replace(/\s+/g, "-"))
  })

  return keys
}

const hasSharedMcpLookupKey = (left: Set<string>, right: Set<string>): boolean => {
  for (const key of left) {
    if (right.has(key)) return true
  }
  return false
}

export const mcpNameMatches = (left: unknown, right: unknown): boolean => {
  return hasSharedMcpLookupKey(getMcpLookupKeys(left), getMcpLookupKeys(right))
}

export const findMatchingMcpServer = <T extends McpLookupReference>(
  servers: T[],
  serverName: string
): T | undefined => {
  const targetKeys = getMcpLookupKeys(serverName)
  return servers.find((server) =>
    hasSharedMcpLookupKey(getMcpLookupKeys(server.name, server.app_id), targetKeys)
  )
}

export const findMatchingMcpApp = <T extends McpLookupReference>(
  apps: T[],
  serverName: string
): T | undefined => {
  const targetKeys = getMcpLookupKeys(serverName)
  return apps.find((app) =>
    hasSharedMcpLookupKey(getMcpLookupKeys(app.name, app.id), targetKeys)
  )
}
