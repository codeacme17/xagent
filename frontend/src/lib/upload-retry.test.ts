import { describe, expect, it, vi } from "vitest"

import {
  isRetriableUploadStatus,
  uploadOutcomeForStatus,
  UploadRequestError,
  withUploadRetry,
} from "./upload-retry"

const retriableError = (status: number) =>
  new UploadRequestError("Storage unavailable", {
    status,
    retriable: isRetriableUploadStatus(status),
    outcome: uploadOutcomeForStatus(status),
  })

describe("withUploadRetry", () => {
  it("retries a 503 and returns the eventual success", async () => {
    const sleep = vi.fn(async () => {})
    const perform = vi.fn()
      .mockRejectedValueOnce(retriableError(503))
      .mockResolvedValueOnce("uploaded")

    await expect(withUploadRetry(perform, { sleep, random: () => 1 }))
      .resolves.toBe("uploaded")
    expect(perform).toHaveBeenCalledTimes(2)
    expect(sleep).toHaveBeenCalledWith(400)
  })

  it("backs off exponentially and gives up with the last error", async () => {
    const sleep = vi.fn<(ms: number) => Promise<void>>(async () => {})
    const perform = vi.fn().mockRejectedValue(retriableError(503))

    await expect(withUploadRetry(perform, { sleep, random: () => 1 }))
      .rejects.toThrow("Storage unavailable")
    expect(perform).toHaveBeenCalledTimes(3)
    expect(sleep.mock.calls.map(([ms]) => ms)).toEqual([400, 800])
  })

  it("spreads retries with full jitter so a fleet does not march in lockstep", async () => {
    const sleep = vi.fn<(ms: number) => Promise<void>>(async () => {})
    const perform = vi.fn().mockRejectedValue(retriableError(503))

    await expect(withUploadRetry(perform, { sleep, random: () => 0.25 }))
      .rejects.toThrow("Storage unavailable")
    expect(sleep.mock.calls.map(([ms]) => ms)).toEqual([100, 200])
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

  it.each([502, 504])(
    "does not retry a %i, whose outcome is unknown",
    async (status) => {
      // Both mean a proxy reached the upstream and could not get a usable
      // answer back, so the upload may already have landed.
      const sleep = vi.fn(async () => {})
      const perform = vi.fn().mockRejectedValue(retriableError(status))

      await expect(withUploadRetry(perform, { sleep })).rejects.toThrow(
        "Storage unavailable",
      )
      expect(perform).toHaveBeenCalledTimes(1)
      expect(sleep).not.toHaveBeenCalled()
    },
  )

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
      const upload = withUploadRetry(perform)

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

describe("uploadOutcomeForStatus", () => {
  it("treats a refusal as proof that nothing was stored", () => {
    // 4xx is the server declining the request; 503 is the upload endpoint
    // answering only after it rolled back what it had staged.
    for (const status of [400, 401, 413, 422, 429, 503]) {
      expect(uploadOutcomeForStatus(status)).toBe("refused")
    }
  })

  it("treats anything else as possibly stored", () => {
    // A proxy reached the upstream (502/504), the server failed after
    // committing (500), or the success body could not be read (200).
    for (const status of [200, 500, 502, 504]) {
      expect(uploadOutcomeForStatus(status)).toBe("unknown")
    }
  })
})
