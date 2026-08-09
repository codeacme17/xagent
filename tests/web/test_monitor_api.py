"""
测试监控 API 的数据库兼容性
"""

from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import JSON, Column, Integer
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base

from xagent.web.api.monitor import PG_NUL_ESCAPE_PATTERN, get_json_field_expression

Base = declarative_base()


class MockTraceEvent(Base):
    """模拟的 TraceEvent 模型"""

    __tablename__ = "trace_events"

    event_id = Column(Integer, primary_key=True)
    data = Column(JSON)


class TestMonitorDatabaseCompatibility:
    """测试监控 API 的数据库兼容性"""

    def test_postgresql_json_extraction(self):
        """测试 PostgreSQL 的 JSON 字段提取"""
        # 创建模拟的数据库会话
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"
        mock_session.bind = mock_engine

        # 测试字段提取
        column = MockTraceEvent.data
        result = get_json_field_expression(column, "tool_name", mock_session)

        # 验证 PostgreSQL 语法 - 检查是否使用了 ->> 操作符
        result_str = str(result)
        assert "->>" in result_str
        assert "trace_events.data" in result_str

    def test_postgresql_guard_compiles_to_operators_postgresql_has(self):
        """The control-character guard must use operators that exist.

        ``~?`` is not a PostgreSQL operator for ``json``, ``jsonb`` or
        ``text``, so the guard failed every monitoring query on PostgreSQL
        while this SQLite-oriented suite stayed green (#1149). The real
        regex-match operator is ``~`` and it takes a text operand, hence the
        cast. Compiling for the PostgreSQL dialect is what makes this a guard
        rather than a restatement -- ``str(expr)`` renders the same either way.
        """
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data
        compiled = get_json_field_expression(column, "tool_name", mock_session).compile(
            dialect=postgresql.dialect()
        )

        statement = str(compiled)
        assert "~?" not in statement
        assert "CAST(trace_events.data AS TEXT) ~ " in statement
        assert "->>" in statement
        # The pattern travels as a bind parameter, not as inlined SQL.
        assert PG_NUL_ESCAPE_PATTERN in compiled.params.values()

    def test_mysql_json_extraction(self):
        """测试 MySQL 的 JSON 字段提取"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "mysql"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data
        result = get_json_field_expression(column, "tool_name", mock_session)

        # MySQL 应该使用 JSON_EXTRACT 函数
        assert "json_extract" in str(result).lower()

    def test_sqlite_json_extraction(self):
        """测试 SQLite 的 JSON 字段提取"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data
        result = get_json_field_expression(column, "tool_name", mock_session)

        # SQLite 应该使用 json_extract 函数
        assert "json_extract" in str(result).lower()

    def test_field_path_formatting(self):
        """测试字段路径格式化"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_session.bind = mock_engine

        column = MockTraceEvent.data

        # 测试带 '$.' 前缀的路径
        result1 = get_json_field_expression(column, "$.tool_name", mock_session)

        # 测试不带前缀的路径
        result2 = get_json_field_expression(column, "tool_name", mock_session)

        # 两种方式应该产生相同的结果
        assert str(result1) == str(result2)

    def test_none_bind_error(self):
        """测试 bind 为 None 时抛出错误"""
        mock_session = MagicMock()
        mock_session.bind = None

        column = MockTraceEvent.data

        with pytest.raises(ValueError, match="Database session bind is None"):
            get_json_field_expression(column, "tool_name", mock_session)


def test_admin_user_permissions():
    """测试管理员用户权限检查"""
    # 测试管理员用户
    admin_user = Mock()
    admin_user.is_admin = True

    from xagent.web.api.monitor import is_admin_user

    assert is_admin_user(admin_user) is True

    # 测试普通用户
    normal_user = Mock()
    normal_user.is_admin = False

    assert is_admin_user(normal_user) is False


class TestMonitorAPIUserIsolation:
    """测试监控 API 的用户隔离功能"""

    def test_user_query_filtering(self):
        """测试用户查询过滤逻辑"""
        mock_session = MagicMock()
        mock_engine = Mock()
        mock_engine.dialect.name = "sqlite"
        mock_session.bind = mock_engine

        # 模拟普通用户
        normal_user = Mock()
        normal_user.id = 123
        normal_user.is_admin = False

        # 模拟管理员用户
        admin_user = Mock()
        admin_user.id = 1
        admin_user.is_admin = True

        from xagent.web.api.monitor import is_admin_user

        # 测试权限检查
        assert is_admin_user(normal_user) is False
        assert is_admin_user(admin_user) is True


if __name__ == "__main__":
    pytest.main([__file__])
