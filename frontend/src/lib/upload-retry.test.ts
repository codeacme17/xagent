import { describe, expect, it, vi } from "vitest"

import {
  isRetriableUploadStatus,
  UploadRequestError,
  withUploadRetry,
} from "./upload-retry"

const retriableError = (status: number) =>
  new UploadRequestError("Storage unavailable", {
    status,
    retriable: isRetriableUploadStatus(status),
  })

describe("withUploadRetry", () => {
  it("retries a 503 and returns the eventual success", async () => {
    const sleep = vi.fn(async () => {})
    const perform = vi.fn()
      .mockRejectedValueOnce(retriableError(503))
      .mockResolvedValueOnce("uploaded")

    await expect(withUploadRetry(perform, { sleep })).resolves.toBe("uploaded")
    expect(perform).toHaveBeenCalledTimes(2)
    expect(sleep).toHaveBeenCalledWith(400)
  })

  it("backs off exponentially and gives up with the last error", async () => {
    const sleep = vi.fn<(ms: number) => Promise<void>>(async () => {})
    const perform = vi.fn().mockRejectedValue(retriableError(503))

    await expect(withUploadRetry(perform, { sleep })).rejects.toThrow(
      "Storage unavailable",
    )
    expect(perform).toHaveBeenCalledTimes(3)
    expect(sleep.mock.calls.map(([ms]) => ms)).toEqual([400, 800])
  })

  it("does not retry a client-side rejection", async () => {
    const sleep = vi.fn(async () => {})
    const perform = vi.fn().mockRejectedValue(retriableError(413))

    await expect(withUploadRetry(perform, { sleep })).rejects.toThrow(
      "Storage unavailable",
    )
    expect(perform).toHaveBeenCalledTimes(1)
    expect(sleep).not.toHaveBeenCalled()
  })

  it("does not retry a gateway timeout, whose outcome is unknown", async () => {
    const sleep = vi.fn(async () => {})
    const perform = vi.fn().mockRejectedValue(retriableError(504))

    await expect(withUploadRetry(perform, { sleep })).rejects.toThrow(
      "Storage unavailable",
    )
    expect(perform).toHaveBeenCalledTimes(1)
    expect(sleep).not.toHaveBeenCalled()
  })

  it("does not retry a rejected request whose outcome is unknown", async () => {
    const sleep = vi.fn(async () => {})
    // A dropped connection surfaces as a plain TypeError from fetch: the
    // server may already have stored the file, so a retry could duplicate it.
    const perform = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))

    await expect(withUploadRetry(perform, { sleep })).rejects.toThrow(
      "Failed to fetch",
    )
    expect(perform).toHaveBeenCalledTimes(1)
    expect(sleep).not.toHaveBeenCalled()
  })

  it("waits on a real timer when no sleep is injected", async () => {
    vi.useFakeTimers()
    try {
      const perform = vi.fn()
        .mockRejectedValueOnce(retriableError(503))
        .mockResolvedValueOnce("uploaded")
      const upload = withUploadRetry(perform, { attempts: 2 })

      await vi.advanceTimersByTimeAsync(400)

      await expect(upload).resolves.toBe("uploaded")
      expect(perform).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("abandons the backoff wait once the caller loses ownership", async () => {
    const cancellation = Promise.reject<never>(new Error("connection changed"))
    cancellation.catch(() => {})
    const perform = vi.fn().mockRejectedValue(retriableError(503))

    await expect(withUploadRetry(perform, {
      cancellation,
      sleep: () => new Promise<void>(() => {}),
    })).rejects.toThrow("connection changed")
    expect(perform).toHaveBeenCalledTimes(1)
  })
})
