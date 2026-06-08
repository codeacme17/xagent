"use client"

import { useParams, useSearchParams } from "next/navigation"
import { PublicAgentChatPage } from "@/components/widget/public-agent-chat-page"

export default function WidgetChatPage() {
  const params = useParams()
  const searchParams = useSearchParams()
  const token = params.token as string

  return (
    <PublicAgentChatPage
      authMode="widget"
      routeToken={token}
      guestId={searchParams.get("guest_id")}
      searchAgentId={searchParams.get("agent_id") ? parseInt(searchParams.get("agent_id") as string, 10) : null}
    />
  )
}
