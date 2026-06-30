import { describe, expect, it } from "vitest"

import { getNavigationGroupsForUser } from "./sidebar-navigation"

describe("sidebar navigation", () => {
  it("exposes Conversation Logs under the More resource menu", () => {
    const groups = getNavigationGroupsForUser({ is_admin: false })
    const resources = groups.find((group) => group.titleKey === "nav.sections.resources")
    const more = resources?.items.find((item) => item.href === "__resources_more__")

    expect(more?.children).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "Conversation Logs",
          nameKey: "nav.conversationLogs",
          href: "/conversation-logs",
        }),
      ])
    )

    const channels = more?.children?.find((item) => item.href === "/channels")
    const conversationLogs = more?.children?.find(
      (item) => item.href === "/conversation-logs"
    )
    expect(conversationLogs?.icon).not.toBe(channels?.icon)
  })
})
