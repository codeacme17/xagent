import {
  getUploadErrorMessage,
  isJsonRecord,
  parseApiResponse,
  UPLOAD_ERROR_MESSAGES,
} from "@/lib/api-wrapper"
import {
  isRetriableUploadStatus,
  uploadOutcomeForStatus,
  UploadRequestError,
  withUploadRetry,
  type UploadRetryOptions,
} from "@/lib/upload-retry"

export interface PublicChatUploadedFile {
  file_id: string
  name?: string
  size?: number
  type?: string
}

interface UploadPublicChatFileOptions {
  url: string
  accessToken: string
  file: File
  taskType: string
  taskId?: number | string | null
  fallbackError: string
  retry?: UploadRetryOptions
}

/**
 * Uploads one file for the widget/share chat.
 *
 * Deliberately uses `fetch` rather than `apiRequest`: public visitors carry a
 * guest token, so the shared 401 handling (which redirects to /login) does not
 * apply here.
 *
 * One request carries exactly one file, so a bounded retry of a refused
 * request cannot duplicate a sibling file that already landed.
 */
export async function uploadPublicChatFile({
  url,
  accessToken,
  file,
  taskType,
  taskId,
  fallbackError,
  retry,
}: UploadPublicChatFileOptions): Promise<PublicChatUploadedFile> {
  const sendUpload = async (): Promise<PublicChatUploadedFile> => {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("task_type", taskType)
    if (taskId != null) {
      formData.append("task_id", taskId.toString())
    }

    const response = await fetch(url, {
      method: "POST",
      headers: { "Authorization": `Bearer ${accessToken}` },
      body: formData,
    })
    const parsed = await parseApiResponse(response)
    const data = isJsonRecord(parsed.data) ? parsed.data : null
    const fileId = typeof data?.file_id === "string" ? data.file_id : null

    if (!response.ok || data?.success !== true || !fileId) {
      throw new UploadRequestError(
        getUploadErrorMessage(response, parsed, {
          generic: fallbackError,
          ...UPLOAD_ERROR_MESSAGES,
        }),
        {
          status: response.status,
          retriable: !response.ok && isRetriableUploadStatus(response.status),
          // An unreadable success body is the ambiguous case: the file may be
          // stored under an id this client never learned.
          outcome: uploadOutcomeForStatus(response.status),
        },
      )
    }

    return {
      file_id: fileId,
      name: file.name,
      size: file.size,
      type: file.type,
    }
  }

  const uploaded = await withUploadRetry(sendUpload, retry)
  // Stamp the id onto this file as soon as it lands, not after the caller's
  // `Promise.all` settles: a sibling's failure rejects the aggregate, and a
  // draft resubmitted without this would upload these bytes a second time.
  ;(file as File & { file_id?: string }).file_id = uploaded.file_id
  return uploaded
}
