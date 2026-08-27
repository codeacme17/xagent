import type { TranslationKey } from "@/i18n/translations"

const CLIENT_ERROR_CODES = [
  "message_processing_failed",
  "task_execution_failed",
  "guidance_in_progress",
  "upload_too_large",
  "upload_proxy_error",
  "upload_failed",
] as const

export type ClientErrorCode = (typeof CLIENT_ERROR_CODES)[number]

const CLIENT_ERROR_TRANSLATION_KEYS: Record<ClientErrorCode, TranslationKey> = {
  message_processing_failed: "clientErrors.messageProcessingFailed",
  task_execution_failed: "clientErrors.taskExecutionFailed",
  guidance_in_progress: "clientErrors.guidanceInProgress",
  upload_too_large: "clientErrors.uploadTooLarge",
  upload_proxy_error: "clientErrors.uploadProxyError",
  upload_failed: "clientErrors.uploadFailed",
}

const CLIENT_ERROR_FALLBACKS: Record<ClientErrorCode, string> = {
  message_processing_failed: "The message could not be processed. Please try again.",
  task_execution_failed: "Task execution failed.",
  guidance_in_progress: "A previous guidance message is still being applied. Please wait for it to finish.",
  upload_too_large: "File is too large. Please reduce the upload size and try again.",
  upload_proxy_error: "Upload failed before reaching the application. Please check the server upload limit.",
  upload_failed: "Upload failed. Please try again.",
}

const CLIENT_ERROR_CODE_SET = new Set<string>(CLIENT_ERROR_CODES)

export function readClientErrorCode(value: unknown): ClientErrorCode | null {
  return typeof value === "string" && CLIENT_ERROR_CODE_SET.has(value)
    ? value as ClientErrorCode
    : null
}

export function clientErrorTranslationKey(code: ClientErrorCode): TranslationKey {
  return CLIENT_ERROR_TRANSLATION_KEYS[code]
}

export function clientErrorFallback(code: ClientErrorCode): string {
  return CLIENT_ERROR_FALLBACKS[code]
}
