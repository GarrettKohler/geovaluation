"""
Unit tests for the training details data contract.

Validates that the /api/sites response includes all fields required by the
training details page's siteDataMap, and that the data shapes are correct
for frontend rendering of training/test site results.
"""

import pytest
import json


class TestSitesRevenueDataContract:
    """
    Tests that /api/sites provides all fields needed by the training details
    page's siteDataMap (avgMonthlyRevenue, totalRevenue, activeMonths,
    revenueScore, status).
    """

    def test_sites_include_avg_monthly_revenue(self, client):
        """Sites response includes avgMonthlyRevenue for siteDataMap."""
        response = client.get("/api/sites")
        data = response.get_json()
        # Check a sample of sites
        for site in data[:50]:
            assert "avgMonthlyRevenue" in site, (
                f"Site {site.get('GTVID', '?')} missing avgMonthlyRevenue"
            )
            assert isinstance(site["avgMonthlyRevenue"], (int, float, type(None)))

    def test_sites_include_total_revenue(self, client):
        """Sites response includes totalRevenue for siteDataMap."""
        response = client.get("/api/sites")
        data = response.get_json()
        for site in data[:50]:
            assert "totalRevenue" in site, (
                f"Site {site.get('GTVID', '?')} missing totalRevenue"
            )
            assert isinstance(site["totalRevenue"], (int, float, type(None)))

    def test_sites_include_active_months(self, client):
        """Sites response includes activeMonths for siteDataMap."""
        response = client.get("/api/sites")
        data = response.get_json()
        for site in data[:50]:
            assert "activeMonths" in site, (
                f"Site {site.get('GTVID', '?')} missing activeMonths"
            )

    def test_sites_include_revenue_score(self, client):
        """Sites response includes revenueScore for siteDataMap."""
        response = client.get("/api/sites")
        data = response.get_json()
        for site in data[:50]:
            assert "revenueScore" in site, (
                f"Site {site.get('GTVID', '?')} missing revenueScore"
            )
            score = site["revenueScore"]
            if score is not None:
                assert 0 <= score <= 1, (
                    f"revenueScore {score} out of [0, 1] range"
                )

    def test_sites_include_status(self, client):
        """Sites response includes status field for siteDataMap."""
        response = client.get("/api/sites")
        data = response.get_json()
        for site in data[:50]:
            assert "status" in site, (
                f"Site {site.get('GTVID', '?')} missing status"
            )
            # Status should be a non-empty string
            assert isinstance(site["status"], str) and len(site["status"]) > 0

    def test_sites_include_gtvid(self, client):
        """Sites response includes GTVID for building siteDataMap keys."""
        response = client.get("/api/sites")
        data = response.get_json()
        for site in data[:50]:
            assert "GTVID" in site
            assert isinstance(site["GTVID"], str)
            assert len(site["GTVID"]) > 0

    def test_revenue_values_non_negative(self, client):
        """Revenue values are non-negative (siteDataMap uses || 0 fallback)."""
        response = client.get("/api/sites")
        data = response.get_json()
        for site in data[:100]:
            avg = site.get("avgMonthlyRevenue")
            total = site.get("totalRevenue")
            if avg is not None:
                assert avg >= 0, f"Negative avgMonthlyRevenue: {avg}"
            if total is not None:
                assert total >= 0, f"Negative totalRevenue: {total}"


class TestBulkSiteDetailsDataContract:
    """
    Tests that /api/bulk-site-details still returns correct shape.
    Although the training details page no longer uses this endpoint,
    it may be used by other parts of the app.
    """

    def test_bulk_details_returns_dict_per_site(self, client, sample_site_ids):
        """Each site in bulk response is a dict with string keys."""
        response = client.post(
            "/api/bulk-site-details",
            json={"site_ids": sample_site_ids[:3]},
            content_type="application/json",
        )
        data = response.get_json()
        assert isinstance(data, dict)
        for site_id, details in data.items():
            assert isinstance(details, dict)

    def test_bulk_details_handles_unknown_ids(self, client):
        """Unknown site IDs are simply omitted from the response."""
        response = client.post(
            "/api/bulk-site-details",
            json={"site_ids": ["NONEXISTENT_999", "FAKE_ID_XYZ"]},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        # Unknown IDs should not appear in response
        assert "NONEXISTENT_999" not in data
        assert "FAKE_ID_XYZ" not in data

    def test_bulk_details_no_nan_in_response(self, client, sample_site_ids):
        """Bulk details response has no NaN/Inf values (JSON-safe)."""
        response = client.post(
            "/api/bulk-site-details",
            json={"site_ids": sample_site_ids[:5]},
            content_type="application/json",
        )
        # If response contains NaN, json parsing would fail or return unusual values
        raw_text = response.get_data(as_text=True)
        assert "NaN" not in raw_text, "Response contains NaN"
        assert "Infinity" not in raw_text, "Response contains Infinity"


class TestTrainingDetailsSessionDataShape:
    """
    Tests that validate the expected shape of data the frontend assembles
    for sessionStorage (the siteDataMap contract).

    Since this is frontend logic, these tests verify the API provides
    the correct raw materials for the frontend to build the siteDataMap.
    """

    def test_active_sites_have_revenue_data(self, client):
        """Active sites provide revenue fields needed for siteDataMap."""
        response = client.get("/api/sites")
        data = response.get_json()

        active_sites = [s for s in data if s.get("status") == "Active"]
        assert len(active_sites) > 0, "No active sites found"

        # Active sites should have revenue data for regression training
        sites_with_revenue = [
            s for s in active_sites
            if s.get("avgMonthlyRevenue", 0) > 0
        ]
        # At least some active sites should have revenue
        assert len(sites_with_revenue) > 0, (
            "No active sites with revenue data found"
        )

    def test_site_data_map_fields_complete(self, client):
        """A site can populate all siteDataMap fields from /api/sites response."""
        response = client.get("/api/sites")
        data = response.get_json()

        # Pick a site with revenue data
        site_with_rev = next(
            (s for s in data if s.get("avgMonthlyRevenue", 0) > 0),
            None
        )
        assert site_with_rev is not None, "No site with revenue data found"

        # Simulate siteDataMap construction (mirrors frontend logic)
        site_data_map_entry = {
            "avgMonthlyRevenue": site_with_rev.get("avgMonthlyRevenue", 0),
            "totalRevenue": site_with_rev.get("totalRevenue", 0),
            "activeMonths": site_with_rev.get("activeMonths", 0),
            "revenueScore": site_with_rev.get("revenueScore", 0),
            "status": site_with_rev.get("status", "Unknown"),
            "predictedRevenue": None,  # Would come from model
        }

        # All fields should be populated
        assert site_data_map_entry["avgMonthlyRevenue"] > 0
        assert isinstance(site_data_map_entry["status"], str)
        assert len(site_data_map_entry["status"]) > 0
        assert 0 <= site_data_map_entry["revenueScore"] <= 1

    def test_sufficient_active_sites_for_regression(self, client):
        """Enough active sites with revenue exist for 80/20 train/test split."""
        response = client.get("/api/sites")
        data = response.get_json()

        active_with_revenue = [
            s for s in data
            if s.get("status") == "Active" and s.get("avgMonthlyRevenue", 0) > 0
        ]

        # Need at least 100 sites for meaningful train/test split
        assert len(active_with_revenue) >= 100, (
            f"Only {len(active_with_revenue)} active sites with revenue, "
            "need 100+ for regression split"
        )

    def test_session_storage_serializable(self, client):
        """Site data can be serialized to JSON for sessionStorage."""
        response = client.get("/api/sites")
        data = response.get_json()

        # Build a mock siteDataMap from first 10 sites
        site_data_map = {}
        for site in data[:10]:
            site_data_map[site["GTVID"]] = {
                "avgMonthlyRevenue": site.get("avgMonthlyRevenue") or 0,
                "totalRevenue": site.get("totalRevenue") or 0,
                "activeMonths": site.get("activeMonths") or 0,
                "revenueScore": site.get("revenueScore") or 0,
                "status": site.get("status") or "Unknown",
                "predictedRevenue": None,
            }

        # Should be JSON serializable without errors
        serialized = json.dumps(site_data_map)
        assert len(serialized) > 0

        # Should round-trip cleanly
        deserialized = json.loads(serialized)
        assert len(deserialized) == 10
        for site_id, entry in deserialized.items():
            assert "avgMonthlyRevenue" in entry
            assert "predictedRevenue" in entry
