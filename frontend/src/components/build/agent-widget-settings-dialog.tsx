"use client"

import React, { useEffect, useMemo, useState } from "react"
import { Check, ChevronLeft, Code2, Copy, X, Zap } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getBrowserLocationOrigin } from "@/lib/browser-location"
import { copyToClipboard } from "@/lib/clipboard"
import { getApiUrl } from "@/lib/utils"

export interface AgentWidgetConfig {
  widget_enabled: boolean
  allowed_domains: string[]
}

interface AgentWidgetSettingsDialogProps {
  agentId: number | null
  agentName?: string
  open: boolean
  onOpenChange: (open: boolean) => void
  widgetConfig: AgentWidgetConfig
  onWidgetConfigUpdated?: (updatedAgent: Record<string, unknown>) => void
}

async function parseWidgetConfigError(response: Response): Promise<Error> {
  try {
    const data = await response.json()
    if (typeof data?.detail === "string" && data.detail.trim()) {
      return new Error(data.detail)
    }
  } catch {
    // Use the generic fallback below.
  }
  return new Error("Failed to update widget configuration")
}

export function AgentWidgetSettingsDialog({
  agentId,
  agentName,
  open,
  onOpenChange,
  widgetConfig,
  onWidgetConfigUpdated,
}: AgentWidgetSettingsDialogProps) {
  const { t } = useI18n()
  const [appOrigin, setAppOrigin] = useState(() => getBrowserLocationOrigin())
  const [widgetState, setWidgetState] = useState<AgentWidgetConfig>(widgetConfig)
  const [newDomain, setNewDomain] = useState("")
  const [isUpdating, setIsUpdating] = useState(false)
  const [copiedSnippet, setCopiedSnippet] = useState(false)

  useEffect(() => {
    if (open) {
      setAppOrigin(getBrowserLocationOrigin())
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    setWidgetState({
      widget_enabled: Boolean(widgetConfig.widget_enabled),
      allowed_domains: Array.isArray(widgetConfig.allowed_domains) ? widgetConfig.allowed_domains : [],
    })
    setNewDomain("")
    setCopiedSnippet(false)
  }, [open, widgetConfig.allowed_domains, widgetConfig.widget_enabled])

  const allowedDomains = Array.isArray(widgetState.allowed_domains)
    ? widgetState.allowed_domains
    : []

  const widgetSnippet = useMemo(() => {
    if (!agentId || !appOrigin) return ""
    return `<script
  src="${appOrigin}/widget.js"
  data-agent-id="${agentId}"
  data-button-size="60px"
  data-button-color="#000"
  data-icon-color="#fff"
  data-panel-bg-color="#fff">
</script>`
  }, [agentId, appOrigin])

  const handleWidgetConfigUpdate = async (updates: Partial<AgentWidgetConfig>) => {
    if (!agentId) return
    setIsUpdating(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      })
      if (!response.ok) {
        throw await parseWidgetConfigError(response)
      }
      const updatedAgent = await response.json()
      const nextState = {
        widget_enabled:
          typeof updatedAgent.widget_enabled === "boolean"
            ? updatedAgent.widget_enabled
            : updates.widget_enabled ?? widgetState.widget_enabled,
        allowed_domains: Array.isArray(updatedAgent.allowed_domains)
          ? updatedAgent.allowed_domains
          : updates.allowed_domains ?? widgetState.allowed_domains,
      }
      setWidgetState(nextState)
      onWidgetConfigUpdated?.(updatedAgent)
      toast.success(t("appWidget.messages.updated"))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("appWidget.messages.updateFailed"))
    } finally {
      setIsUpdating(false)
    }
  }

  const handleCopySnippet = async () => {
    if (!widgetSnippet) return
    if (await copyToClipboard(widgetSnippet)) {
      setCopiedSnippet(true)
      toast.success(t("common.copied"))
      window.setTimeout(() => setCopiedSnippet(false), 2000)
    } else {
      toast.error(t("appWidget.messages.copyFailed"))
    }
  }

  const handleAddDomain = () => {
    const domain = newDomain.trim().toLowerCase()
    if (!domain) return
    const existingDomains = new Set(allowedDomains.map((item) => item.toLowerCase()))
    if (existingDomains.has(domain)) {
      setNewDomain("")
      return
    }
    setNewDomain("")
    void handleWidgetConfigUpdate({
      allowed_domains: [...allowedDomains, domain],
    })
  }

  const handleRemoveDomain = (domain: string) => {
    void handleWidgetConfigUpdate({
      allowed_domains: allowedDomains.filter((item) => item !== domain),
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="flex max-h-[88vh] w-[calc(100vw-2rem)] max-w-none flex-col overflow-hidden p-0 sm:max-w-[680px]"
      >
        <DialogHeader className="border-b px-5 py-4 pr-12">
          <DialogTitle className="flex items-center gap-2 text-base">
            <Zap className="h-4 w-4 text-primary" />
            {t("triggers.title")}
          </DialogTitle>
          <DialogDescription>
            {agentName ? `${agentName} · ${t("triggers.subtitle")}` : t("triggers.subtitle")}
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-3 border-b pb-4">
              <div className="flex min-w-0 items-center gap-3">
                <Button variant="ghost" size="sm" className="-ml-2" onClick={() => onOpenChange(false)}>
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  {t("common.back")}
                </Button>
                <div className="flex min-w-0 items-center gap-2 border-l pl-3">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted">
                    <Code2 className="h-4 w-4" />
                  </div>
                  <div className="truncate text-sm font-semibold">
                    {t("appWidget.dialog.title")}
                  </div>
                </div>
              </div>
              <Switch
                aria-label={t("appWidget.dialog.enabledLabel")}
                checked={widgetState.widget_enabled}
                disabled={isUpdating || !agentId}
                onCheckedChange={(checked) => void handleWidgetConfigUpdate({ widget_enabled: checked })}
              />
            </div>

            <section className="space-y-3">
              <div>
                <h3 className="text-sm font-medium">{t("appWidget.dialog.allowedDomains")}</h3>
                <p className="text-xs text-muted-foreground">
                  {t("deploy_agent.access_control.allowed_domains_desc")}
                </p>
              </div>
              <div className="flex gap-2">
                <Input
                  value={newDomain}
                  onChange={(event) => setNewDomain(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault()
                      handleAddDomain()
                    }
                  }}
                  placeholder={t("appWidget.dialog.domainPlaceholder")}
                  disabled={isUpdating || !agentId}
                />
                <Button
                  type="button"
                  variant="secondary"
                  onClick={handleAddDomain}
                  disabled={isUpdating || !agentId || !newDomain.trim()}
                >
                  {t("appWidget.dialog.addDomain")}
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {allowedDomains.length > 0 ? (
                  allowedDomains.map((domain) => (
                    <Badge key={domain} variant="secondary" className="gap-1 px-2.5 py-1">
                      {domain}
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        onClick={() => handleRemoveDomain(domain)}
                        disabled={isUpdating}
                        aria-label={t("appWidget.dialog.removeDomain")}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">
                    {t("appWidget.dialog.noDomains")}
                  </span>
                )}
              </div>
            </section>

            <section className="space-y-3">
              <div>
                <h3 className="text-sm font-medium">{t("appWidget.dialog.embedTitle")}</h3>
                <p className="text-xs text-muted-foreground">
                  {t("appWidget.dialog.embedDescription")}
                </p>
              </div>
              <div className="rounded-md bg-muted p-3">
                <div className="flex items-start gap-2">
                  <pre className="min-w-0 flex-1 max-h-56 overflow-auto whitespace-pre-wrap break-all text-xs text-muted-foreground">
                    {widgetSnippet}
                  </pre>
                  <Button
                    type="button"
                    variant="secondary"
                    size="icon"
                    onClick={() => void handleCopySnippet()}
                    disabled={!widgetSnippet}
                    aria-label={t("appWidget.dialog.copySnippet")}
                    title={t("appWidget.dialog.copySnippet")}
                    className="h-8 w-8 shrink-0"
                  >
                    {copiedSnippet ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            </section>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
