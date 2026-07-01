# Gmail Auto Trigger Design

## Goal

Implement the product logic behind "Gmail: Run the agent when a new email arrives in Gmail" by turning connected Gmail accounts into real event sources that automatically create trigger runs when matching messages arrive.

## Scope

This spec extends the existing scheduled-events-triggers foundation. It builds real Gmail event ingestion for enabled Gmail triggers.

In scope:

- Register Gmail watches for connected Gmail accounts that have at least one enabled Gmail trigger.
- Receive Google Cloud Pub/Sub push messages for Gmail history notifications.
- Use Gmail `history.list` and `messages.get` to resolve new email messages.
- Filter messages by each trigger's configured label or folder, sender filter, and subject keyword.
- Create non-test trigger runs through the existing trigger execution path.
- Keep Gmail OAuth credentials in the existing `user_oauth` storage.
- Persist watch state separately from OAuth credentials.
- Provide a local development setup that can be tested with ngrok or another HTTPS tunnel.

Out of scope:

- Replacing the existing Gmail MCP connector.
- Sending Gmail replies automatically.
- Implementing App Widget runtime behavior.
- Supporting polling-only Gmail ingestion as the primary product path.
- Supporting multiple Gmail accounts per user in the first implementation if the existing connector UI only exposes one active Gmail account.

## Current State

The existing issue set has built:

- Gmail trigger configuration in the Agent Builder Triggers UI.
- Gmail OAuth connection state using the official Gmail connector.
- Manual `Test trigger` execution that creates hidden trigger tasks.

The missing product logic is automatic ingestion from real Gmail arrival events. The existing issue set explicitly did not include Gmail push delivery, Pub/Sub, Gmail watch renewal, or automatic email-arrival execution.

## Recommended Architecture

Use Gmail push notifications as the source of truth:

1. The user connects Gmail through the existing OAuth connector.
2. An enabled Gmail trigger exists for an agent owned by that user.
3. The backend registers a Gmail watch for the user's Gmail account.
4. Google publishes notifications to a configured Cloud Pub/Sub topic.
5. Pub/Sub pushes the notification to a backend endpoint.
6. The backend decodes the notification and uses the stored Gmail history cursor to fetch newly added messages.
7. Each message is matched against enabled Gmail triggers for that user/account.
8. Matching messages create trigger runs using the existing trigger execution path.

This keeps the architecture close to Google's intended Gmail integration model and avoids polling every inbox.

## Data Model

Add a `gmail_watch_states` table.

Fields:

- `id`
- `user_id`
- `oauth_account_id`
- `email`
- `history_id`
- `watch_expiration`
- `topic_name`
- `last_error`
- `created_at`
- `updated_at`

Constraints:

- `oauth_account_id` references `user_oauth.id` with cascade delete.
- Unique index on `oauth_account_id`.
- Index on `email`.
- Index on `watch_expiration`.

Do not add duplicate OAuth token storage. The watch state records only Gmail cursor and watch metadata.

## Backend Components

### Gmail Credentials Helper

Create a focused helper that builds Google credentials from a `UserOAuth` row and refreshes the token if needed. The helper should update `user_oauth.access_token` and `expires_at` after refresh.

### Gmail Watch Service

Responsibilities:

- Find connected Gmail OAuth accounts with enabled Gmail triggers.
- Call Gmail `users.watch` with the configured Pub/Sub topic.
- Store returned `historyId` and `expiration`.
- Refresh watches before expiration.
- Record errors on the watch state and the related triggers without failing unrelated accounts.

The service should be callable from:

- Trigger create/update when a Gmail trigger becomes enabled.
- OAuth callback after Gmail connection.
- A background renewal loop.

### Pub/Sub Push Endpoint

Add:

`POST /api/triggers/gmail/pubsub`

Expected Pub/Sub envelope:

```json
{
  "message": {
    "data": "base64url-json",
    "messageId": "pubsub-message-id",
    "publishTime": "..."
  },
  "subscription": "..."
}
```

Decoded Gmail payload:

```json
{
  "emailAddress": "user@example.com",
  "historyId": "12345"
}
```

Security:

- Require a shared token for local/dev push testing.
- Accept the token from `X-Xagent-Gmail-Pubsub-Token`.
- Compare it with `XAGENT_GMAIL_PUBSUB_PUSH_TOKEN`.
- If no push token is configured, reject push calls with 503 so local misconfiguration is obvious.

### Gmail History Processor

Responsibilities:

- Load watch state by email.
- Use the stored `history_id` as `startHistoryId`.
- Call Gmail `users.history.list` with `historyTypes=["messageAdded"]`.
- Fetch added messages with `users.messages.get`.
- Normalize payload fields:
  - `message_id`
  - `thread_id`
  - `history_id`
  - `from`
  - `to`
  - `subject`
  - `date`
  - `snippet`
  - `label_ids`
  - optional plain text body preview
- Match enabled Gmail triggers for the watch state's user.
- Create trigger runs with `source_event_id = "gmail:<message_id>"`.
- Update stored `history_id` after successful processing.

Idempotency already exists in trigger runs. Reusing message ID as source event ID prevents duplicate tasks for repeated Pub/Sub notifications.

## Trigger Matching

For each enabled Gmail trigger:

- `watch_label` matches when the Gmail message label IDs contain the configured label. `INBOX` should work directly.
- `sender_filter` matches case-insensitively against the normalized From header.
- `subject_keyword` matches case-insensitively against the Subject header.
- Empty optional filters do not restrict matching.

If multiple Gmail triggers match the same message, create one trigger run per matching trigger.

## Error Handling

- Missing Pub/Sub token: 503.
- Invalid Pub/Sub token: 401.
- Malformed Pub/Sub envelope: 400.
- Unknown email/watch state: 202 accepted, logged warning. Pub/Sub should not retry forever for an account this backend does not know.
- Gmail `startHistoryId` too old: register a new watch, update history cursor, record a warning, and do not create retroactive trigger runs.
- Gmail API failure: 500 only for transient processing failures where Pub/Sub retry is useful.
- Disabled trigger: ignored.
- Duplicate message notification: returns success and creates no duplicate run.

## Configuration

Add config helpers in `src/xagent/config.py`:

- `get_gmail_pubsub_topic_name()`
- `get_gmail_pubsub_push_token()`
- `get_gmail_watch_enabled()`
- `get_gmail_watch_renewal_interval_seconds()`
- `get_gmail_watch_renewal_lead_seconds()`

Environment variables:

- `XAGENT_GMAIL_PUBSUB_TOPIC`
- `XAGENT_GMAIL_PUBSUB_PUSH_TOKEN`
- `XAGENT_GMAIL_WATCH_ENABLED`
- `XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS`
- `XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS`

## Local Test Plan

1. Configure Google Cloud:
   - Enable Gmail API.
   - Enable Pub/Sub API.
   - Create a Pub/Sub topic.
   - Grant `gmail-api-push@system.gserviceaccount.com` permission to publish to the topic.
   - Create a push subscription targeting a public HTTPS tunnel to the local backend.
2. Start backend with:
   - `XAGENT_GMAIL_PUBSUB_TOPIC`
   - `XAGENT_GMAIL_PUBSUB_PUSH_TOKEN`
   - `XAGENT_GMAIL_WATCH_ENABLED=true`
3. Connect Gmail through the existing UI.
4. Enable a Gmail trigger with sender and subject filters.
5. Send a real email matching the filters.
6. Confirm `Recent runs` gets a new non-test completed run.
7. Open the hidden task and confirm the payload contains the real Gmail message fields.

## Test Coverage

Backend tests:

- Gmail watch registration stores `history_id` and expiration.
- Watch renewal skips accounts without enabled Gmail triggers.
- Pub/Sub endpoint rejects missing or invalid push tokens.
- Pub/Sub endpoint decodes valid Gmail notifications.
- History processor matches label, sender, and subject filters.
- Duplicate Gmail message notification creates one trigger run.
- Expired `startHistoryId` path renews watch and does not create retroactive runs.

Frontend tests:

- Existing Gmail trigger UI tests remain valid.
- No new frontend surface is required for the first automatic-ingestion implementation.

## Acceptance Criteria

- A connected Gmail account with an enabled Gmail trigger is registered with Gmail `users.watch`.
- A valid Pub/Sub Gmail notification causes the backend to fetch new messages from Gmail history.
- A real matching Gmail message creates a non-test trigger run.
- The hidden task payload includes Gmail message identity and headers.
- Duplicate Pub/Sub deliveries do not create duplicate tasks.
- Existing manual `Test trigger` behavior still works.
- The UI does not claim automatic Gmail delivery when watch configuration is missing or disabled.
