from sqlalchemy import Column, DateTime, String

from .database import Base


class OidcConsumedToken(Base):  # type: ignore[no-any-unimported]
    """Consumed short-lived OIDC token IDs for replay protection."""

    __tablename__ = "oidc_consumed_tokens"

    token_id = Column(String(96), primary_key=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
