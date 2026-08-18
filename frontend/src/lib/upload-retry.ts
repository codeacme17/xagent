/**
 * Bounded retry for file uploads.
 *
 * Only failures that carry a definite retriable HTTP status are retried: a
 * rejected `fetch` (connection dropped, request aborted mid-body) leaves the
 * outcome unknown, and the server may already have persisted the upload, so
 * retrying it would duplicate the file. Callers therefore have to raise an
 * `UploadRequestError` — anything else propagates on the first attempt.
 */

/**
 * Statuses that mean the request was *refused* before anything was stored, so
 * re-sending it cannot duplicate the file.
 *
 * 504 is deliberately excluded: a gateway timeout means the upstream did
 * receive the request and simply did not answer in time, so the upload may
 * well have landed.
 */
const RETRIABLE_UPLOAD_STATUSES = new Set([502, 503])

export function isRetriableUploadStatus(status: number): boolean {
  return RETRIABLE_UPLOAD_STATUSES.has(status)
}

export class UploadRequestError extends Error {
  readonly status: number | null
  readonly retriable: boolean

  constructor(
    message: string,
    { status, retriable }: { status: number | null; retriable: boolean },
  ) {
    super(message)
    this.name = "UploadRequestError"
    this.status = status
    this.retriable = retriable
  }
}

export interface UploadRetryOptions {
  /** Total attempts, including the first one. */
  attempts?: number
  /** Delay before the second attempt; doubles for each further attempt. */
  baseDelayMs?: number
  /**
   * Rejects when the caller no longer owns the upload (connection swapped,
   * message superseded). Aborts the backoff wait instead of letting a dead
   * submission keep retrying.
   */
  cancellation?: Promise<never>
  sleep?: (ms: number) => Promise<void>
}

const defaultSleep = (ms: number) =>
  new Promise<void>(resolve => { setTimeout(resolve, ms) })

export async function withUploadRetry<T>(
  perform: () => Promise<T>,
  {
    attempts = 3,
    baseDelayMs = 400,
    cancellation,
    sleep = defaultSleep,
  }: UploadRetryOptions = {},
): Promise<T> {
  const totalAttempts = Math.max(1, attempts)
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await perform()
    } catch (error) {
      const retriable =
        error instanceof UploadRequestError && error.retriable
      if (!retriable || attempt >= totalAttempts) throw error
      const delay = baseDelayMs * 2 ** (attempt - 1)
      const wait = sleep(delay)
      await (cancellation ? Promise.race([wait, cancellation]) : wait)
    }
  }
}
