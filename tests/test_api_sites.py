"""
Tests for sites API endpoints.

Verifies that all site-related API endpoints return correct data
structure and content for frontend rendering.
"""

import pytest
import json


class TestSitesAPI:
    """Tests for /api/sites endpoint."""

    def test_get_sites_returns_200(self, client):
        """API returns 200 OK for sites endpoint."""
        response = client.get("/api/sites")
        assert response.status_code == 200

    def test_get_sites_returns_json_array(self, client):
        """API returns a JSON array of sites."""
        response = client.get("/api/sites")
        data = response.get_json()
        assert isinstance(data, list)

    def test_get_sites_returns_many_sites(self, client):
        """API returns 50K+ sites as specified."""
        response = client.get("/api/sites")
        data = response.get_json()
        # Should have at least 50,000 sites per requirements
        assert len(data) >= 50000, f"Expected 50K+ sites, got {len(data)}"

    def test_sites_have_required_fields(self, client, required_site_fields):
        """Each site has all fields required for frontend rendering."""
        response = client.get("/api/sites")
        data = response.get_json()

        # Check first 100 sites for required fields
        for site in data[:100]:
            for field in required_site_fields:
                assert field in site, f"Site missing required field: {field}"

    def test_sites_have_valid_coordinates(self, client):
        """All sites have valid latitude/longitude coordinates."""
        response = client.get("/api/sites")
        data = response.get_json()

        for site in data[:100]:
            lat = site["Latitude"]
            lon = site["Longitude"]

            # Valid US coordinates range
            assert -90 <= lat <= 90, f"Invalid latitude: {lat}"
            assert -180 <= lon <= 180, f"Invalid longitude: {lon}"
            # More specific US bounds
            assert 24 <= lat <= 50, f"Latitude {lat} outside continental US"
            assert -125 <= lon <= -66, f"Longitude {lon} outside continental US"

    def test_sites_have_revenue_scores(self, client):
        """All sites have revenue scores between 0 and 1."""
        response = client.get("/api/sites")
        data = response.get_json()

        for site in data[:100]:
            score = site["revenueScore"]
            assert isinstance(score, (int, float)), f"Score not numeric: {score}"
            assert 0 <= score <= 1, f"Score out of range [0,1]: {score}"


class TestSiteDetailsAPI:
    """Tests for /api/site-details/<site_id> endpoint."""

    def test_get_full_site_details_returns_200(self, client, single_site_id):
        """API returns 200 OK for valid site ID."""
        response = client.get(f"/api/site-details/{single_site_id}")
        assert response.status_code == 200

    def test_site_details_has_categories(self, client, single_site_id):
        """Site details include organized category data."""
        response = client.get(f"/api/site-details/{single_site_id}")
        data = response.get_json()

        assert "site_id" in data
        assert "categories" in data

        expected_categories = [
            "Location",
            "Site Info",
            "Brands",
            "Revenue",
            "Demographics",
            "Performance",
            "Capabilities",
            "Sales",
        ]

        for category in expected_categories:
            assert category in data["categories"], f"Missing category: {category}"

    def test_site_details_revenue_category(self, client, single_site_id):
        """Revenue category has required financial metrics."""
        response = client.get(f"/api/site-details/{single_site_id}")
        data = response.get_json()

        revenue = data["categories"]["Revenue"]
        assert "Avg Monthly Revenue" in revenue
        assert "Total Revenue" in revenue
        assert "Active Months" in revenue
        assert "Revenue Score" in revenue


class TestBulkSiteDetailsAPI:
    """Tests for /api/bulk-site-details endpoint."""

    def test_bulk_details_returns_200(self, client, sample_site_ids):
        """API returns 200 OK for bulk request."""
        response = client.post(
            "/api/bulk-site-details",
            json={"site_ids": sample_site_ids[:3]},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_bulk_details_returns_all_requested(self, client, sample_site_ids):
        """API returns details for all requested sites."""
        requested = sample_site_ids[:5]
        response = client.post(
            "/api/bulk-site-details",
            json={"site_ids": requested},
            content_type="application/json",
        )
        data = response.get_json()

        assert isinstance(data, dict)
        for site_id in requested:
            assert site_id in data, f"Missing site in response: {site_id}"

    def test_bulk_details_empty_request(self, client):
        """API handles empty request gracefully."""
        response = client.post(
            "/api/bulk-site-details",
            json={"site_ids": []},
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.get_json() == {}


class TestFilterOptionsAPI:
    """Tests for /api/filter-options endpoint."""

    def test_filter_options_returns_200(self, client):
        """API returns 200 OK."""
        response = client.get("/api/filter-options")
        assert response.status_code == 200

    def test_filter_options_has_all_categories(self, client, categorical_filter_fields):
        """API returns options for all categorical filter fields."""
        response = client.get("/api/filter-options")
        data = response.get_json()

        for field in categorical_filter_fields:
            assert field in data, f"Missing filter category: {field}"
            assert isinstance(data[field], list), f"Filter options not a list: {field}"

    def test_filter_options_have_values(self, client):
        """Each filter category has at least one option."""
        response = client.get("/api/filter-options")
        data = response.get_json()

        for field, options in data.items():
            assert len(options) > 0, f"No options for filter: {field}"

    def test_state_filter_has_us_states(self, client):
        """State filter includes expected US state codes."""
        response = client.get("/api/filter-options")
        data = response.get_json()

        states = data.get("State", [])
        # Check for some common states
        expected_states = ["TX", "CA", "FL", "NY"]
        for state in expected_states:
            assert state in states, f"Missing expected state: {state}"


class TestFilteredSitesAPI:
    """Tests for /api/filtered-sites endpoint."""

    def test_filtered_sites_returns_200(self, client):
        """API returns 200 OK for filtered request."""
        response = client.post(
            "/api/filtered-sites",
            json={"filters": {"State": "TX"}},
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_filtered_sites_returns_site_ids(self, client):
        """API returns list of matching site IDs."""
        response = client.post(
            "/api/filtered-sites",
            json={"filters": {"State": "TX"}},
            content_type="application/json",
        )
        data = response.get_json()

        assert "site_ids" in data
        assert "count" in data
        assert isinstance(data["site_ids"], list)
        assert data["count"] == len(data["site_ids"])

    def test_filter_reduces_results(self, client):
        """Filtering produces fewer results than total sites."""
        # Get total sites
        all_sites_response = client.get("/api/sites")
        total_sites = len(all_sites_response.get_json())

        # Get filtered sites
        filtered_response = client.post(
            "/api/filtered-sites",
            json={"filters": {"State": "TX"}},
            content_type="application/json",
        )
        filtered_count = filtered_response.get_json()["count"]

        assert filtered_count < total_sites, "Filter should reduce results"
        assert filtered_count > 0, "Filter should return some results"

    def test_empty_filter_returns_empty(self, client):
        """Empty filter dict returns empty results."""
        response = client.post(
            "/api/filtered-sites",
            json={"filters": {}},
            content_type="application/json",
        )
        data = response.get_json()
        assert data["site_ids"] == []
        assert data["count"] == 0
