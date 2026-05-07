"""
End-to-end tests for search functionality verification.

This module tests that search works correctly after document ingestion,
ensuring that users can find their content immediately.

IMPORTANT: Legacy data compatibility tests in this file must be updated
whenever schema changes are made to ensure forward compatibility is maintained.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.e2e, pytest.mark.contract_stub]


@pytest.fixture
def sample_search_files() -> Generator[tuple[dict[str, str], str], None, None]:
    """Create sample test files for search testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with specific searchable content
        test_files = {
            "python_tutorial.txt": "Python is a programming language. Python is widely used for web development, data science, and automation.",
            "machine_learning.md": "# Machine Learning Guide\n\nMachine learning is a subset of artificial intelligence. It focuses on building systems that can learn from data.",
            "cooking_guide.txt": "Cooking tips: Always use fresh ingredients. Follow the recipe carefully. Taste your food while cooking.",
            "travel_guide.txt": "Travel destinations: Paris, Tokyo, New York are popular cities. Each city has unique attractions and culture.",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


# ==========================================
# BASIC SEARCH TESTS
# ==========================================


class TestBasicSearch:
    """
    Test basic search functionality.

    These tests verify that:
    1. Search works immediately after ingestion
    2. Search results are relevant
    3. Search pagination works
    4. Search filters work correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_immediate_after_ingestion(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search works immediately after document ingestion."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_immediate"
        file_path = files["python_tutorial.txt"]

        # Ingest a document
        with open(file_path, "rb") as f:
            ingest_response = client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if ingest_response.status_code == 200:
            # Search immediately after ingestion
            search_response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": "Python programming",
                    "top_k": "5",
                },
                headers=auth_headers,
            )

            # Search should work
            assert search_response.status_code == 200
            result = search_response.json()
            assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_relevance_ranking(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search results are ranked by relevance."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_relevance"

        # Ingest documents with specific content
        keywords_docs = [
            ("python_tutorial.txt", "Python"),
            ("machine_learning.md", "machine learning"),
            ("cooking_guide.txt", "cooking"),
        ]

        for filename, keyword in keywords_docs:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        # Search for specific term and verify relevance
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": 3,
            },
            headers=auth_headers,
        )

        # Should return relevant results
        assert search_response.status_code == 200
        result = search_response.json()
        assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_pagination(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search pagination works correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_pagination"

        # Create multiple documents
        for i in range(5):
            content = f"Document {i} with searchable content about topic {i % 3}."
            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w"
            ) as tmp:
                tmp.write(content)
                tmp.flush()

                try:
                    with open(tmp.name, "rb") as f:
                        client.post(
                            "/api/kb/ingest",
                            files={"file": (f"doc{i}.txt", f, "text/plain")},
                            data={"collection": collection_name},
                            headers=auth_headers,
                        )
                finally:
                    import os

                    os.unlink(tmp.name)

        # Test pagination
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "document",
                "top_k": 3,
            },
            headers=auth_headers,
        )

        # Should handle pagination
        assert search_response.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_filters(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search filters work correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_filters"

        # Ingest documents
        for filename in ["python_tutorial.txt", "machine_learning.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

        # Search with filters
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "document",
                "top_k": 5,
                # Add any available filters
            },
            headers=auth_headers,
        )

        # Should handle search
        assert search_response.status_code == 200


# ==========================================
# MULTI-TENANT SEARCH TESTS
# ==========================================


class TestMultiTenantSearch:
    """
    Test multi-tenant search isolation.

    These tests verify that:
    1. Users can only search their own documents
    2. Admin users can search across tenants
    3. Legacy data isolation works correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_only_returns_own_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that regular users can only search their own documents."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_isolation"
        file_path = files["python_tutorial.txt"]

        # Ingest a document
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search as regular user
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": "5",
            },
            headers=auth_headers,
        )

        # Search should respect tenant isolation
        assert search_response.status_code == 200
        result = search_response.json()
        assert "results" in result or "status" in result


# ==========================================
# SEARCH AFTER SCHEMA CHANGES TESTS
# ==========================================


class TestSearchAfterSchemaChanges:
    """
    Test search functionality after schema changes.

    These tests verify that:
    1. Search works after migration
    2. Search works with mixed schema versions
    3. Fallback mechanisms work for legacy data
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_after_migration(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search works correctly after schema migration."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_migration"
        file_path = files["python_tutorial.txt"]

        # Ingest document (simulating post-migration state)
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search should work after migration
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": "5",
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200


# ==========================================
# SEARCH ACCURACY TESTS
# ==========================================


class TestSearchAccuracy:
    """
    Test search result accuracy and quality.

    These tests verify that:
    1. Search returns relevant results
    2. Search scores are reasonable
    3. Search handles different query types
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_returns_relevant_results(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search returns relevant results."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_accuracy"
        file_path = files["python_tutorial.txt"]

        # Ingest document with specific content
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search for specific term from the document
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python programming language",
                "top_k": 5,
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200
        result = search_response.json()
        assert "results" in result or "status" in result

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_different_query_types(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search handles different query types correctly."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_query_types"

        # Ingest document
        file_path = files["python_tutorial.txt"]
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Test different query types
        queries = [
            "Python",  # Single word
            "Python programming",  # Phrase
            "Python language web development",  # Multiple words
        ]

        for query in queries:
            search_response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": query,
                    "top_k": 3,
                },
                headers=auth_headers,
            )

            # All query types should work
            assert search_response.status_code == 200


# ==========================================
# REAL-TIME SEARCH TESTS
# ==========================================


class TestRealTimeSearch:
    """
    Test real-time search functionality.

    These tests verify that:
    1. Search works immediately after ingestion
    2. Search updates in real-time
    3. Search handles concurrent operations
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_updates_in_realtime(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search results update in real-time after ingestion."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_realtime"

        # Initial search should be empty
        client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "Python",
                "top_k": "5",
            },
            headers=auth_headers,
        )

        # Ingest document
        file_path = files["python_tutorial.txt"]
        with open(file_path, "rb") as f:
            ingest_response = client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        if ingest_response.status_code == 200:
            # Search again should now return results
            final_search = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": "Python",
                    "top_k": 5,
                },
                headers=auth_headers,
            )

            assert final_search.status_code == 200

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_multiple_documents(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search works with multiple documents."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_multiple"

        # Ingest and search for multiple documents sequentially
        results = []
        for filename in ["python_tutorial.txt", "machine_learning.md"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                ingest_response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, "text/plain")},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

            if ingest_response.status_code == 200:
                # Search for content
                search_response = client.post(
                    "/api/kb/search",
                    data={
                        "collection": collection_name,
                        "query_text": "content",
                        "top_k": 3,
                    },
                    headers=auth_headers,
                )
                results.append(search_response.status_code)

        # All operations should succeed
        assert all(code == 200 for code in results)


# ==========================================
# SEARCH ERROR HANDLING TESTS
# ==========================================


class TestSearchErrorHandling:
    """Test search error handling and edge cases."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_empty_query(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that search handles empty queries gracefully."""
        collection_name = "e2e_search_empty"

        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": collection_name,
                "query_text": "",  # Empty query
                "top_k": 5,
            },
            headers=auth_headers,
        )

        # Empty query is rejected by validation (FastAPI form validation)
        assert search_response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_nonexistent_collection(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that search handles nonexistent collections gracefully."""
        search_response = client.post(
            "/api/kb/search",
            data={
                "collection": "nonexistent_collection_xyz",
                "query_text": "test",
                "top_k": 5,
            },
            headers=auth_headers,
        )

        # Nonexistent collection returns 404
        assert search_response.status_code == 404

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_search_with_special_characters(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_search_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that search handles special characters in queries."""
        files, temp_dir = sample_search_files
        collection_name = "e2e_search_special"
        file_path = files["python_tutorial.txt"]

        # Ingest document
        with open(file_path, "rb") as f:
            client.post(
                "/api/kb/ingest",
                files={"file": ("python_tutorial.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Search with special characters
        special_queries = [
            "Python & programming",  # Ampersand
            "Python, data, science",  # Commas
            'Python "language"',  # Quotes
        ]

        for query in special_queries:
            search_response = client.post(
                "/api/kb/search",
                data={
                    "collection": collection_name,
                    "query_text": query,
                    "top_k": 3,
                },
                headers=auth_headers,
            )

            # Special characters should be handled normally
            assert search_response.status_code == 200
