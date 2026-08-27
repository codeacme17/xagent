import { describe, expect, it } from "vitest"

import {
  clientErrorFallback,
  clientErrorTranslationKey,
  readClientErrorCode,
} from "@/lib/client-errors"

describe("client error wire contract", () => {
  it.each([
    ["message_processing_failed", "clientErrors.messageProcessingFailed"],
    ["task_execution_failed", "clientErrors.taskExecutionFailed"],
    ["guidance_in_progress", "clientErrors.guidanceInProgress"],
    ["upload_too_large", "clientErrors.uploadTooLarge"],
    ["upload_proxy_error", "clientErrors.uploadProxyError"],
    ["upload_failed", "clientErrors.uploadFailed"],
  ] as const)("maps %s to a typed translation key", (code, key) => {
    expect(readClientErrorCode(code)).toBe(code)
    expect(clientErrorTranslationKey(code)).toBe(key)
    expect(clientErrorFallback(code).trim()).not.toBe("")
  })

  it("rejects unknown and non-string codes", () => {
    expect(readClientErrorCode("provider_secret")).toBeNull()
    expect(readClientErrorCode({ error_code: "upload_failed" })).toBeNull()
  })
})
