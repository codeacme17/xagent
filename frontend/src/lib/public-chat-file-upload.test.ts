import { afterEach, describe, expect, it, vi } from "vitest"

import { UPLOAD_ERROR_MESSAGES } from "./api-wrapper"
import { uploadPublicChatFile } from "./public-chat-file-upload"
import { UploadRequestError } from "./upload-retry"

describe("uploadPublicChatFile", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("rejects backend HTTP failures instead of silently accepting them", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "File is too large" }), {
        status: 413,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
    })).rejects.toThrow("File is too large")

    const [, request] = fetchMock.mock.calls[0]
    expect(new Headers(request?.headers).get("Authorization")).toBe(
      "Bearer guest-token",
    )
    const body = request?.body as FormData
    expect(body.get("file")).toBe(file)
    expect(body.get("task_type")).toBe("task")
    expect(body.get("task_id")).toBe("42")
  })

  it("returns normalized file metadata for successful uploads", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ success: true, file_id: "file-1" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      fallbackError: "Upload failed",
    })).resolves.toEqual({
      file_id: "file-1",
      name: "trip.txt",
      size: 4,
      type: "text/plain",
    })
  })
})

describe("uploadPublicChatFile transient failures", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  const jsonResponse = (body: unknown, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })

  it("retries a 503 from durable storage and keeps the submission alive", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ detail: "Storage unavailable" }, 503))
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "file-9" }, 200))
    const sleep = vi.fn(async () => {})
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/widget/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
      retry: { sleep },
    })).resolves.toMatchObject({ file_id: "file-9" })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(sleep).toHaveBeenCalledTimes(1)
  })

  it("surfaces the storage failure with its status once retries are exhausted", async () => {
    // A fresh Response per attempt: a body can only be read once.
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => jsonResponse({ detail: "Storage unavailable" }, 503),
    )
    const sleep = vi.fn(async () => {})
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    const failure = await uploadPublicChatFile({
      url: "http://api.local/api/widget/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      fallbackError: "Upload failed",
      retry: { sleep },
    }).catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(UploadRequestError)
    expect(failure).toMatchObject({ status: 503, retriable: true })
    expect((failure as Error).message).toBe("Storage unavailable")
  })

  it("does not retry a permanent rejection", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Unsupported file type" }, 400),
    )
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/widget/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      fallbackError: "Upload failed",
    })).rejects.toThrow("Unsupported file type")

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("reports a gateway HTML failure instead of the generic fallback", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response("<html><body>503 Service Unavailable</body></html>", {
        status: 503,
        headers: { "Content-Type": "text/html" },
      }),
    )
    const sleep = vi.fn(async () => {})
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/widget/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      fallbackError: "Upload failed",
      retry: { sleep },
    })).rejects.toThrow(UPLOAD_ERROR_MESSAGES.proxy)
  })
})
