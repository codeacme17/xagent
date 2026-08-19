import React, { useEffect, useMemo, useRef, useState } from "react"
import { Interaction } from "@/contexts/app-context-chat"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { MultiSelect } from "@/components/ui/multi-select"
import { Select } from "@/components/ui/select"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown, ChevronRight, MessageSquare, Upload, File as FileIcon, X, Globe } from "lucide-react"
import { generateClientMessageId } from "@/lib/utils"

interface ClarificationFormProps {
  message?: string
  interactions: Interaction[]
  messageId?: string
  active?: boolean
  filesDisabled?: boolean
  onSend?: (message: string, files?: File[], metadata?: any) => Promise<void> | void
}

const FILE_ACTION_VALUE_RE = /(^|[-_\s])(upload|file)(?=$|[-_\s])/i

const isFileActionValue = (value: unknown): boolean =>
  typeof value === "string" && FILE_ACTION_VALUE_RE.test(value)

const isFileActionOption = (
  option: { value?: string; action_type?: string } | undefined,
): boolean => {
  const actionType = option?.action_type?.toLowerCase()
  if (actionType !== undefined) {
    return actionType === "upload"
  }
  return isFileActionValue(option?.value)
}

const isFileActionSelection = (
  option: { value?: string; action_type?: string } | undefined,
  value: unknown,
): boolean => option
  ? isFileActionOption(option)
  : isFileActionValue(value)

type SendDisposition = "not_sent" | "rejected" | "outcome_unknown"

/**
 * Delivery failures carry whether the turn definitely never reached the agent.
 * Plain errors (local validation, unexpected throws) carry nothing, and are
 * left unqualified rather than guessed at: telling a visitor to resubmit a
 * turn that may have landed is worse than saying nothing.
 */
const readSendDisposition = (error: unknown): SendDisposition | null => {
  if (typeof error !== "object" || error === null || !("disposition" in error)) {
    return null
  }
  const disposition = (error as { disposition: unknown }).disposition
  return disposition === "not_sent"
    || disposition === "rejected"
    || disposition === "outcome_unknown"
    ? disposition
    : null
}

/**
 * Only the reasons the sender can act on — the backend's rejection text, an
 * upload response detail — are shown as-is. Connection plumbing messages stay
 * behind the localized string: they are English diagnostics, and a widget
 * visitor is not the audience for them.
 */
const readSendReason = (error: unknown): string => {
  if (
    typeof error !== "object"
    || error === null
    || (error as { userFacing?: unknown }).userFacing !== true
  ) {
    return ""
  }
  return error instanceof Error ? error.message.trim() : ""
}

export function ClarificationForm({
  interactions,
  messageId,
  active = true,
  filesDisabled: filesDisabledOverride,
  onSend,
}: ClarificationFormProps) {
  // If onSend is provided, use it (e.g., from builder chat), otherwise use useApp
  let sendMessage: any, dispatch: any, contextFilesDisabled: boolean | undefined;
  try {
    const appCtx = useApp();
    sendMessage = appCtx.sendMessage;
    dispatch = appCtx.dispatch;
    contextFilesDisabled = appCtx.filesDisabled;
  } catch {
    // We might not be in the app context (e.g., agent builder chat)
  }
  const filesDisabled = filesDisabledOverride ?? contextFilesDisabled ?? true

  const { t } = useI18n()
  const [formState, setFormState] = useState<Record<string, any>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(!active)
  const [isOpen, setIsOpen] = useState(active)
  const [sendFailure, setSendFailure] = useState<{ message: string; hint: string | null } | null>(null)
  // An unresolved submission keeps its client message id, so a retry lands on
  // the server's existing claim instead of opening a second turn. Cleared on
  // success, and when the server explicitly asks for a fresh id.
  const deliveryAttemptRef = useRef<string | null>(null)
  // Set only when a resubmit could duplicate something the server cannot
  // deduplicate - an attachment that may have landed under an id this client
  // never learned. An unknown *delivery* outcome does not block: the retry
  // reuses the same client message id, so the server adjudicates it.
  const [resubmitBlocked, setResubmitBlocked] = useState(false)

  useEffect(() => {
    if (active) {
      // A new clarification round reuses this component instance on the live
      // turn render path, so every per-submission guard has to be cleared -
      // otherwise round 1's block, or its client message id, leaks into
      // round 2's answer.
      setIsSubmitted(false)
      setIsOpen(true)
      setResubmitBlocked(false)
      setSendFailure(null)
      deliveryAttemptRef.current = null
    }
  }, [active])

  const normalizedInteractions = useMemo(() => {
    const seenFields = new Set<string>()
    return interactions.map((interaction: any, index) => {
      const rawType = interaction.type
      const type =
        rawType === "text" || rawType === "input" || rawType === "textarea" || rawType === "string"
          ? "text_input"
          : rawType === "file" || rawType === "upload"
            ? "file_upload"
            : rawType === "number" || rawType === "integer"
              ? "number_input"
              : rawType === "boolean"
                ? "confirm"
                : rawType
      const rawField = interaction.field || interaction.id || interaction.name || interaction.properties?.field || interaction.properties?.id || `response_${index}`
      const baseField = typeof rawField === "string" && rawField.trim() ? rawField.trim() : `response_${index}`
      const field = seenFields.has(baseField) ? `${baseField}_${index}` : baseField
      seenFields.add(field)
      const rawOptions = Array.isArray(interaction.options)
        ? interaction.options
        : Array.isArray(interaction.actions)
          ? interaction.actions
          : undefined
      const options = rawOptions
        ?.map((opt: any) => ({
          value: typeof opt?.value === "string" ? opt.value : String(opt?.label || ""),
          label: typeof opt?.label === "string" ? opt.label : String(opt?.value || ""),
          description: typeof opt?.description === "string" ? opt.description : undefined,
          action_type: typeof opt?.action_type === "string" ? opt.action_type : undefined,
        }))
        .filter((opt: { value: string; label: string }) => opt.value && opt.label)
      return {
        ...interaction,
        type,
        field,
        ...(options ? { options } : {}),
      }
    }) as Interaction[]
  }, [interactions])

  useEffect(() => {
    if (!filesDisabled) return

    setFormState((previous) => {
      const next = { ...previous }
      let changed = false

      for (const interaction of normalizedInteractions) {
        if (interaction.type === "file_upload" && interaction.field in next) {
          delete next[interaction.field]
          changed = true
        }

        if (interaction.type === "action_cards") {
          const fileField = `${interaction.field}_files`
          if (fileField in next) {
            delete next[fileField]
            changed = true
          }
          const selectedOption = interaction.options?.find(
            (option) => option.value === next[interaction.field],
          )
          if (isFileActionSelection(
            selectedOption,
            next[interaction.field],
          )) {
            delete next[interaction.field]
            changed = true
          }
        }
      }

      return changed ? next : previous
    })
  }, [filesDisabled, normalizedInteractions])

  const handleInputChange = (field: string, value: any) => {
    setFormState((prev) => ({ ...prev, [field]: value }))
    if (resubmitBlocked) return
    setSendFailure(null)
  }

  const handleSubmit = async () => {
    // Construct the message
    const metadata: any = {}
    const lines = normalizedInteractions.flatMap(interaction => {
      const value = formState[interaction.field]

      if (filesDisabled && interaction.type === "file_upload") {
        return []
      }
      if (
        filesDisabled
        && interaction.type === "action_cards"
        && isFileActionSelection(
          interaction.options?.find((option) => option.value === value),
          value,
        )
      ) {
        return []
      }

      // Skip empty values unless it's a boolean (confirm) which might be false
      if (value === undefined || value === null || (typeof value === "string" && value.trim() === "") || (Array.isArray(value) && value.length === 0)) {
        // If it's a confirm type, default to false if undefined? Or maybe it's required?
        if (interaction.type === "confirm" && value === undefined) {
          return [{ field: interaction.field, label: interaction.label || interaction.field, value: t("chatPage.clarification.no"), isFile: false }]
        }
        return []
      }

      let displayValue = value
      let isFile = false

      if (interaction.type === "select_multiple" && Array.isArray(value)) {
        const labels = value.map(v => interaction.options?.find(o => o.value === v)?.label || v)
        displayValue = labels.join(", ")
      } else if (interaction.type === "select_one" || interaction.type === "action_cards") {
        const label = interaction.options?.find(o => o.value === value)?.label || value
        displayValue = label
      } else if (interaction.type === "confirm") {
        displayValue = value ? t("chatPage.clarification.yes") : t("chatPage.clarification.no")
      } else if (interaction.type === "file_upload") {
        isFile = true
      }

      const results = [{ field: interaction.field, label: interaction.label || interaction.field, value: isFile ? value : displayValue, isFile }]

      // For action_cards, if files were uploaded alongside it, add them too
      if (interaction.type === "action_cards" && !filesDisabled) {
        const fileValue = formState[`${interaction.field}_files`]
        if (fileValue && ((fileValue instanceof FileList && fileValue.length > 0) || (Array.isArray(fileValue) && fileValue.length > 0))) {
          results.push({ field: `${interaction.field}_files`, label: t("chatPage.clarification.uploadedFiles"), value: fileValue, isFile: true })
        }

        const urlValue = formState[`${interaction.field}_url`]
        if (urlValue && typeof urlValue === "string" && urlValue.trim() !== "") {
          results.push({ field: `${interaction.field}_url`, label: t("chatPage.clarification.websiteUrl") || "Website URL", value: urlValue, isFile: false })
          metadata.url = urlValue
        }
      }

      return results
    }).filter(Boolean) as any[]

    if (lines.length === 0) {
      toast.error(t("chatPage.clarification.required"))
      return
    }

    // Separate files and text
    const textParts = lines.filter(l => !l.isFile).map(l => `${l.label}: ${l.value}`)
    const fileParts = lines.filter(l => l.isFile)

    const textMessage = textParts.join("\n")
    const files: File[] = []

    fileParts.forEach(part => {
      if (part.value instanceof FileList) {
        for (let i = 0; i < part.value.length; i++) {
          files.push(part.value[i])
        }
      } else if (Array.isArray(part.value)) {
        // Assuming array of Files
        part.value.forEach((f: any) => {
          if (f instanceof File) files.push(f)
        })
      }
    })

    try {
      setIsSubmitting(true)
      setSendFailure(null)
      // If textMessage is empty but we have files, send a generic message?
      const outboundFiles = filesDisabled ? [] : files
      const finalMessage = textMessage || (outboundFiles.length > 0 ? t("chatPage.clarification.uploadedFiles") : t("chatPage.clarification.confirmed"))

      if (onSend) {
        await onSend(finalMessage, outboundFiles, metadata);
      } else if (sendMessage) {
        const clientMessageId = deliveryAttemptRef.current ?? generateClientMessageId()
        deliveryAttemptRef.current = clientMessageId
        await sendMessage(
          finalMessage,
          { force: true, metadata, clientMessageId },
          outboundFiles,
        )
      }

      deliveryAttemptRef.current = null
      setIsSubmitted(true)
      setIsOpen(false)
      if (!onSend && dispatch) {
        dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
      }
    } catch (error) {
      console.error("Failed to send clarification response", error)
      // The rejection reason ("a previous guidance message is still being
      // applied", the upload failure detail) is the only actionable part of
      // the failure; the fixed string is a last resort.
      const detail = readSendReason(error)
      const disposition = readSendDisposition(error)
      const failure = {
        message: detail || t("chatPage.clarification.sendError"),
        hint: disposition === "outcome_unknown"
          ? t("chatPage.clarification.sendOutcomeUnknown")
          : disposition === "not_sent" || disposition === "rejected"
            ? t("chatPage.clarification.sendNotSent")
            : null,
      }
      if (
        error
        && typeof error === "object"
        && (error as { retryWithNewId?: unknown }).retryWithNewId === true
      ) {
        deliveryAttemptRef.current = null
      }
      setResubmitBlocked(
        typeof error === "object"
        && error !== null
        && (error as { requiresReconciliation?: unknown }).requiresReconciliation === true,
      )
      setSendFailure(failure)
      toast.error(failure.message, failure.hint ? { description: failure.hint } : undefined)
    } finally {
      setIsSubmitting(false)
    }
  }

  const renderField = (interaction: Interaction) => {
    const value = formState[interaction.field]

    switch (interaction.type) {
      case "text_input":
        return interaction.multiline ? (
          <Textarea
            placeholder={interaction.placeholder}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        ) : (
          <Input
            placeholder={interaction.placeholder}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        )

      case "number_input":
        return (
          <Input
            type="number"
            placeholder={interaction.placeholder}
            min={interaction.min}
            max={interaction.max}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        )

      case "select_one":
        return (
          <Select
            value={value}
            onValueChange={(v) => handleInputChange(interaction.field, v)}
            options={interaction.options || []}
            placeholder={t("chatPage.clarification.selectOption")}
          />
        )

      case "select_multiple":
        return (
          <MultiSelect
            values={value || []}
            onValuesChange={(v) => handleInputChange(interaction.field, v)}
            options={interaction.options || []}
            placeholder={interaction.placeholder || t("chatPage.clarification.selectOptions")}
          />
        )

      case "file_upload":
        if (filesDisabled) return null

        const fileValue = formState[interaction.field]
        const files: File[] = []
        if (fileValue instanceof FileList) {
          for (let i = 0; i < fileValue.length; i++) {
            files.push(fileValue[i])
          }
        } else if (Array.isArray(fileValue)) {
          files.push(...fileValue)
        }

        const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
          if (e.target.files && e.target.files.length > 0) {
            const newFiles = Array.from(e.target.files)
            // Always allow multiple files
            handleInputChange(interaction.field, [...files, ...newFiles])
          }
          // Reset input value to allow selecting same file again
          e.target.value = ''
        }

        const removeFile = (index: number) => {
          const newFiles = [...files]
          newFiles.splice(index, 1)
          handleInputChange(interaction.field, newFiles)
        }

        return (
          <div className="grid w-full gap-2">
            {files.length > 0 && (
              <div className="grid gap-2">
                {files.map((file, index) => (
                  <div key={index} className="flex items-center gap-2 rounded-md border p-2 text-sm bg-muted/50">
                    <FileIcon className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="flex-1 truncate">{file.name}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={() => removeFile(index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            {/* Always show upload area to allow adding more files */}
            <div className="relative group cursor-pointer">
              <div className="flex h-24 w-full flex-col items-center justify-center gap-2 rounded-md border border-dashed bg-muted/30 hover:bg-muted/50 transition-colors">
                <Upload className="h-6 w-6 text-muted-foreground group-hover:scale-110 transition-transform" />
                <span className="text-xs text-muted-foreground">{t("chatPage.fileUpload.hintDragClick")}</span>
              </div>
              <Input
                type="file"
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                accept={Array.isArray(interaction.accept) ? interaction.accept.join(",") : interaction.accept}
                multiple={interaction.multiple ?? true}
                onChange={handleFileChange}
              />
            </div>
            <div className="text-xs text-muted-foreground">
              {t("chatPage.clarification.acceptedFormats")}: {Array.isArray(interaction.accept) ? interaction.accept.join(", ") : interaction.accept || t("chatPage.clarification.any")}
            </div>
          </div>
        )

      case "confirm":
        return (
          <div className="flex items-center space-x-2">
            <Switch
              id={interaction.field}
              checked={!!value}
              onCheckedChange={(checked) => handleInputChange(interaction.field, checked)}
            />
            <Label htmlFor={interaction.field}>{t("chatPage.clarification.yes")}</Label>
          </div>
        )

      case "action_cards":
        const selectedOption = interaction.options?.find((opt) => opt.value === value)
        const visibleOptions = filesDisabled
          ? interaction.options?.filter((option) => !isFileActionOption(option))
          : interaction.options
        const selectedActionType = selectedOption?.action_type?.toLowerCase()
        const isUploadSelected = !filesDisabled && (
          selectedActionType === 'upload' || isFileActionValue(value)
        );
        const isWebsiteSelected = selectedActionType === 'input_url' || (typeof value === 'string' && (value.toLowerCase().includes('website') || value.toLowerCase().includes('url') || value.toLowerCase().includes('import')));

        return (
          <div className="flex flex-col gap-4 w-full">
            <div className="grid w-full grid-cols-1 sm:grid-cols-2 gap-4">
              {visibleOptions?.map((opt) => (
                <div
                  key={opt.value}
                  className={`flex flex-col items-center justify-center gap-2 rounded-lg border p-6 cursor-pointer transition-all ${value === opt.value ? 'border-primary bg-primary/5 shadow-sm ring-1 ring-primary' : 'bg-card hover:bg-muted/50 hover:border-muted-foreground/30'
                    }`}
                  onClick={() => handleInputChange(interaction.field, opt.value)}
                >
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                    {opt.value.toLowerCase().includes('upload') || opt.value.toLowerCase().includes('file') ? (
                      <Upload className="h-5 w-5 text-primary" />
                    ) : opt.value.toLowerCase().includes('no') || opt.value.toLowerCase().includes('skip') ? (
                      <X className="h-5 w-5 text-primary" />
                    ) : (
                      <Globe className="h-5 w-5 text-primary" />
                    )}
                  </div>
                  <div className="flex flex-col items-center text-center gap-1">
                    <span className="font-medium text-sm text-foreground">{opt.label}</span>
                    {opt.description && <span className="text-xs text-muted-foreground">{opt.description}</span>}
                  </div>
                </div>
              ))}
            </div>
            {isUploadSelected && (
              <div className="mt-2 w-full animate-in fade-in slide-in-from-top-4">
                <div className="text-sm font-medium mb-2">{t("chatPage.fileUpload.hintDragClick")}</div>
                {renderField({ ...interaction, type: "file_upload", field: `${interaction.field}_files` })}
              </div>
            )}
            {isWebsiteSelected && (
              <div className="mt-2 w-full animate-in fade-in slide-in-from-top-4">
                <div className="text-sm font-medium mb-2">Enter website URL</div>
                {renderField({
                  ...interaction,
                  type: "text_input",
                  field: `${interaction.field}_url`,
                  placeholder: "https://...",
                  default: interaction.default_value
                })}
              </div>
            )}
          </div>
        )

      default:
        return <div className="text-destructive text-sm">{t("chatPage.clarification.unsupportedType", { type: interaction.type })}</div>
    }
  }

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="w-full space-y-2 rounded-lg border bg-card text-card-foreground shadow-sm my-2"
    >
      <CollapsibleTrigger asChild>
        <div className="flex items-center justify-between p-4 bg-muted/80 cursor-pointer hover:bg-muted/60 transition-colors">
          <div className="flex items-center gap-2 font-semibold">
            <MessageSquare className="h-4 w-4" />
            <span className="text-sm">{t("chatPage.clarification.title")}</span>
          </div>
          <div>
            {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </div>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-4 p-4">
        <div className="space-y-4">
          {normalizedInteractions.map((interaction, index) => (
            filesDisabled && interaction.type === "file_upload" ? null : (
            <div key={`${interaction.field}-${index}`} className="space-y-2">
              <Label className="text-sm font-medium">
                {interaction.label || interaction.field}
                {interaction.type === "confirm" ? "" : ":"}
              </Label>

              {renderField(interaction)}
            </div>
            )
          ))}
        </div>

        {sendFailure && (
          <div role="alert" className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
            <div>{sendFailure.message}</div>
            {sendFailure.hint && (
              <div className="mt-1 text-xs text-destructive/80">{sendFailure.hint}</div>
            )}
          </div>
        )}

        <div className="pt-2 flex gap-2">
          <Button className="flex-1" size="sm" onClick={handleSubmit} disabled={!active || isSubmitting || isSubmitted || resubmitBlocked}>
            {isSubmitting ? t("chatPage.clarification.submitting") : t("chatPage.clarification.submit")}
          </Button>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
