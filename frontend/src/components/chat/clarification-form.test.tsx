/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const appContextMock = vi.hoisted(() => ({
  dispatch: vi.fn(),
  filesDisabled: false,
  providerAvailable: true,
  sendMessage: vi.fn(),
}))
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => {
    if (!appContextMock.providerAvailable) {
      throw new Error("App provider is unavailable")
    }
    return appContextMock
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

import { ClarificationForm } from "./clarification-form"

describe("ClarificationForm Session file capability", () => {
  beforeEach(() => {
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("removes direct file upload UI and drops staged files after files are disabled", async () => {
    const onSend = vi.fn()
    const interactions = [
      {
        type: "file_upload" as const,
        field: "evidence",
        label: "Evidence",
      },
      {
        type: "text_input" as const,
        field: "note",
        label: "Note",
        placeholder: "Add a note",
      },
    ]
    const { container, rerender } = render(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    )
    expect(fileInput).not.toBeNull()
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["secret"], "secret.txt", { type: "text/plain" })],
      },
    })
    expect(screen.getByText("secret.txt")).toBeInTheDocument()

    appContextMock.filesDisabled = true
    rerender(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(screen.queryByText("secret.txt")).not.toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText("Add a note"), {
      target: { value: "Continue without a file" },
    })
    fireEvent.click(
      screen.getByRole("button", {
        name: "chatPage.clarification.submit",
      }),
    )

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "Note: Continue without a file",
        [],
        {},
      )
    })
  })

  it("removes action-card upload choices and drops their staged files", async () => {
    const onSend = vi.fn()
    const interactions = [
      {
        type: "action_cards" as const,
        field: "source",
        label: "Source",
        options: [
          {
            label: "Upload a file",
            value: "upload",
            action_type: "upload",
          },
          {
            label: "Skip upload",
            value: "skip_upload",
            action_type: "skip",
          },
        ],
      },
    ]
    const { container, rerender } = render(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    fireEvent.click(screen.getByText("Upload a file"))
    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    )
    expect(fileInput).not.toBeNull()
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["secret"], "secret.csv", { type: "text/csv" })],
      },
    })
    expect(screen.getByText("secret.csv")).toBeInTheDocument()

    appContextMock.filesDisabled = true
    rerender(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    expect(screen.queryByText("Upload a file")).not.toBeInTheDocument()
    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(screen.queryByText("secret.csv")).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("Skip upload"))
    fireEvent.click(
      screen.getByRole("button", {
        name: "chatPage.clarification.submit",
      }),
    )

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith("Source: Skip upload", [], {})
    })
  })

  it("preserves file submission for legacy contexts where files are enabled", async () => {
    const onSend = vi.fn()
    const file = new File(["report"], "report.txt", { type: "text/plain" })
    const { container } = render(
      <ClarificationForm
        interactions={[
          {
            type: "file_upload",
            field: "evidence",
            label: "Evidence",
          },
        ]}
        onSend={onSend}
      />,
    )

    fireEvent.change(
      container.querySelector<HTMLInputElement>('input[type="file"]')!,
      { target: { files: [file] } },
    )
    fireEvent.click(
      screen.getByRole("button", {
        name: "chatPage.clarification.submit",
      }),
    )

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "chatPage.clarification.uploadedFiles",
        [file],
        {},
      )
    })
  })

  it("fails closed for file uploads when no app provider or override is available", () => {
    appContextMock.providerAvailable = false
    const { container } = render(
      <ClarificationForm
        interactions={[{ type: "file_upload", field: "evidence", label: "Evidence" }]}
        onSend={vi.fn()}
      />,
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
  })

  it("allows builder callers to explicitly enable file uploads without an app provider", () => {
    appContextMock.providerAvailable = false
    const { container } = render(
      <ClarificationForm
        filesDisabled={false}
        interactions={[{ type: "file_upload", field: "evidence", label: "Evidence" }]}
        onSend={vi.fn()}
      />,
    )

    expect(container.querySelector('input[type="file"]')).not.toBeNull()
  })
})

describe("ClarificationForm delivery failures", () => {
  beforeEach(() => {
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  const deliveryError = (
    message: string,
    disposition: string,
    userFacing = false,
  ) => Object.assign(new Error(message), { disposition, userFacing })

  const submitAnswer = async (onSend: ReturnType<typeof vi.fn>) => {
    render(
      <ClarificationForm
        interactions={[{ type: "text_input" as const, field: "city", label: "City" }]}
        onSend={onSend}
      />,
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )
  }

  it("surfaces the backend rejection reason instead of the generic toast", async () => {
    const onSend = vi.fn().mockRejectedValue(deliveryError(
      "A previous guidance message is still being applied. Please wait for it to finish.",
      "rejected",
      true,
    ))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "A previous guidance message is still being applied. Please wait for it to finish.",
        { description: "chatPage.clarification.sendNotSent" },
      )
    })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "A previous guidance message is still being applied.",
    )
  })

  it("keeps the form submittable after a failure that never reached the agent", async () => {
    const onSend = vi.fn().mockRejectedValue(
      deliveryError("Durable storage is temporarily unavailable", "not_sent", true),
    )

    await submitAnswer(onSend)

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(
      "Durable storage is temporarily unavailable",
      { description: "chatPage.clarification.sendNotSent" },
    ))
    const submit = screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })
    expect(submit).toBeEnabled()
    expect(screen.getByRole("textbox")).toHaveValue("Beijing")
  })

  it("warns instead of inviting a resubmit when the outcome is unknown", async () => {
    const onSend = vi.fn().mockRejectedValue(deliveryError(
      "The task is busy applying an earlier answer.",
      "outcome_unknown",
      true,
    ))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "The task is busy applying an earlier answer.",
        { description: "chatPage.clarification.sendOutcomeUnknown" },
      )
    })
  })

  it("keeps connection plumbing diagnostics away from the visitor", async () => {
    const onSend = vi.fn().mockRejectedValue(deliveryError(
      "Message not sent: the connection changed before delivery.",
      "not_sent",
    ))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "chatPage.clarification.sendError",
        { description: "chatPage.clarification.sendNotSent" },
      )
    })
    expect(await screen.findByRole("alert")).not.toHaveTextContent(
      "the connection changed before delivery",
    )
  })

  it("falls back to the generic string when the failure carries no reason", async () => {
    const onSend = vi.fn().mockRejectedValue(new Error("   "))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "chatPage.clarification.sendError",
        undefined,
      )
    })
  })

  it("clears the failure once the visitor edits an answer", async () => {
    const onSend = vi.fn().mockRejectedValue(
      deliveryError("Durable storage is temporarily unavailable", "not_sent", true),
    )

    await submitAnswer(onSend)

    await screen.findByRole("alert")
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Shanghai" } })
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
  })
})

describe("ClarificationForm resubmission safety", () => {
  beforeEach(() => {
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  const deliveryError = (
    message: string,
    disposition: string,
    extra: Record<string, unknown> = {},
  ) => Object.assign(new Error(message), { disposition, userFacing: true, ...extra })

  const renderForm = () => render(
    <ClarificationForm
      interactions={[{ type: "text_input" as const, field: "city", label: "City" }]}
    />,
  )


  const submit = () => fireEvent.click(
    screen.getByRole("button", { name: "chatPage.clarification.submit" }),
  )

  const sentIds = () => appContextMock.sendMessage.mock.calls.map(
    ([, config]) => (config as { clientMessageId?: string })?.clientMessageId,
  )

  it("stops a second submission when an attachment may already have landed", async () => {
    // Uploaded bytes have no server-side dedup, so this is the one case a
    // human has to reconcile before sending the draft again.
    appContextMock.sendMessage.mockRejectedValue(deliveryError(
      "The upload could not be completed or rolled back.",
      "outcome_unknown",
      { requiresReconciliation: true },
    ))
    renderForm()
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })

    submit()

    await waitFor(() => expect(screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })).toBeDisabled())
    // Editing an answer must not talk the visitor back into resubmitting.
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Shanghai" } })
    expect(screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })).toBeDisabled()
    expect(screen.getByRole("alert")).toHaveTextContent(
      "chatPage.clarification.sendOutcomeUnknown",
    )
    expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1)
  })

  it("lets an unknown delivery outcome be retried under the same id", async () => {
    // A reconnect during ack-wait is an ordinary event. The turn keeps its
    // client message id, so the server adjudicates a duplicate instead of the
    // form locking the visitor out until they reload the page.
    appContextMock.sendMessage.mockRejectedValue(deliveryError(
      "Message delivery was not acknowledged. Your draft was kept.",
      "outcome_unknown",
    ))
    renderForm()
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })

    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
    expect(screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })).toBeEnabled()
    expect(screen.getByRole("alert")).toHaveTextContent(
      "chatPage.clarification.sendOutcomeUnknown",
    )

    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(2))
    const [first, second] = sentIds()
    expect(second).toBe(first)
  })

  it("clears a previous round's block when the form is asked again", async () => {
    // The live turn render path keeps one instance across clarification
    // rounds, so a stale block would silently disable round two.
    appContextMock.sendMessage.mockRejectedValue(deliveryError(
      "The upload could not be completed or rolled back.",
      "outcome_unknown",
      { requiresReconciliation: true },
    ))
    const interactions = [{ type: "text_input" as const, field: "city", label: "City" }]
    const { rerender } = render(
      <ClarificationForm interactions={interactions} active />,
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    submit()
    await waitFor(() => expect(screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })).toBeDisabled())

    rerender(<ClarificationForm interactions={interactions} active={false} />)
    rerender(<ClarificationForm interactions={interactions} active />)

    expect(screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })).toBeEnabled()
    expect(screen.queryByRole("alert")).toBeNull()

    appContextMock.sendMessage.mockResolvedValue(undefined)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Shanghai" } })
    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(2))
    const [first, second] = sentIds()
    expect(second).not.toBe(first)
  })

  it("retries an unresolved submission under its original client message id", async () => {
    appContextMock.sendMessage.mockRejectedValue(
      deliveryError("Durable storage is temporarily unavailable", "not_sent"),
    )
    renderForm()
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })

    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(2))

    const [first, second] = sentIds()
    expect(first).toBeTruthy()
    expect(second).toBe(first)
  })

  it("mints a fresh client message id when the server asks for one", async () => {
    appContextMock.sendMessage.mockRejectedValue(deliveryError(
      "Message id was already used for different content or files.",
      "rejected",
      { retryWithNewId: true },
    ))
    renderForm()
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })

    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
    submit()
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(2))

    const [first, second] = sentIds()
    expect(first).toBeTruthy()
    expect(second).not.toBe(first)
  })
})
