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
 * Only 503 qualifies. The upload endpoint raises it for durable storage after
 * compensating whatever it had staged, and a gateway raises it when it has no
 * upstream to hand the request to; either way nothing was retained.
 *
 * 502 and 504 are excluded for the same reason: both mean a proxy did reach
 * the upstream and then failed to get a usable answer out of it, so the upload
 * may well have landed. Widening this set past "provably refused" needs a
 * server-side idempotency key, not a bigger status list.
 */
const RETRIABLE_UPLOAD_STATUSES = new Set([503])

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
  /** Injectable for tests; returns [0, 1). */
  random?: () => number
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
    random = Math.random,
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
      // Full jitter. A storage outage fails every in-flight upload at once,
      // and the widget fans out one request per file, so a fixed schedule
      // would march the whole fleet back onto the endpoint in lockstep.
      const delay = Math.round(random() * baseDelayMs * 2 ** (attempt - 1))
      const wait = sleep(delay)
      await (cancellation ? Promise.race([wait, cancellation]) : wait)
    }
  }
}
