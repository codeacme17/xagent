from __future__ import annotations

import pytest
from fastapi import HTTPException

from xagent.web.api.admin_users import delete_user, get_users
from xagent.web.models.user import User
from xagent.web.services.user_admin_scope import set_hidden_user_filter

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _reset_hidden_filter():
    yield
    set_hidden_user_filter(None)


@pytest.mark.asyncio
async def test_hidden_users_excluded_from_admin_list_and_delete():
    _admin_headers()  # ensures the admin account exists
    _register_second_user("ghost", "ghostpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        ghost_id = int(db.query(User).filter(User.username == "ghost").one().id)

        # Baseline: no filter (standalone default) -> user is listed.
        res = await get_users(1, 100, "", admin, db)
        assert any(u.id == ghost_id for u in res.users)

        set_hidden_user_filter(lambda _db: [ghost_id])

        # Excluded from the list (and total) ...
        res = await get_users(1, 100, "", admin, db)
        assert all(u.id != ghost_id for u in res.users)
        assert res.total == db.query(User).count() - 1
        # ... from search ...
        res = await get_users(1, 100, "ghost", admin, db)
        assert all(u.id != ghost_id for u in res.users)
        # ... and cannot be deleted (would orphan the data it backs).
        with pytest.raises(HTTPException) as exc:
            await delete_user(ghost_id, admin, db)
        assert exc.value.status_code == 404
        assert db.query(User).filter(User.id == ghost_id).count() == 1
    finally:
        db.close()
