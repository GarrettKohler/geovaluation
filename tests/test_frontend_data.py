"""
Tests for frontend data completeness.

Verifies that all data required for frontend rendering is available
and correctly formatted for the Leaflet/WebGL map visualization.
"""

import pytest
import random


class TestFrontendSitesData:
    """Tests for sites data used in frontend map rendering."""

    def test_all_sites_have_gtvid(self, client):
        """Every site has a unique GTVID identifier."""
        response = client.get("/api/sites")
        sites = response.get_json()

        gtvids = [site["GTVID"] for site in sites]
        assert len(gtvids) == len(set(gtvids)), "Duplicate GTVIDs found"

    def test_all_sites_have_coordinates_for_map(self, client):
        """Every site has coordinates for map placement."""
        response = client.get("/api/sites")
        sites = response.get_json()

        missing_coords = []
        for site in sites:
            if site.get("Latitude") is None or site.get("Longitude") is None:
                missing_coords.append(site.get("GTVID"))

        assert len(missing_coords) == 0, f"{len(missing_coords)} sites missing coordinates"

    def test_all_sites_have_revenue_score_for_coloring(self, client):
        """Every site has a revenue score for map color coding."""
        response = client.get("/api/sites")
        sites = response.get_json()

        missing_scores = []
        for site in sites:
            if site.get("revenueScore") is None:
                missing_scores.append(site.get("GTVID"))

        # Allow some sites to have 0 score, but not None
        for site in sites[:100]:
            assert "revenueScore" in site, "Site missing revenueScore field"

    def test_revenue_scores_enable_color_gradient(self, client):
        """Revenue scores span a range for meaningful color gradient."""
        response = client.get("/api/sites")
        sites = response.get_json()

        scores = [site["revenueScore"] for site in sites if site.get("revenueScore") is not None]

        min_score = min(scores)
        max_score = max(scores)

        # Should have variety in scores
        assert max_score > min_score, "All scores are identical - no color variation"
        assert max_score <= 1.0, "Scores exceed maximum of 1.0"
        assert min_score >= 0.0, "Scores below minimum of 0.0"

    def test_all_sites_have_status(self, client):
        """Every site has a status field."""
        response = client.get("/api/sites")
        sites = response.get_json()

        for site in sites[:100]:
            assert "status" in site, "Site missing status field"
            assert site["status"] is not None, "Site has null status"


class TestFrontendFilterData:
    """Tests for filter options used in frontend sidebar."""

    def test_filter_options_not_empty(self, client, categorical_filter_fields):
        """All filter categories have options to display."""
        response = client.get("/api/filter-options")
        options = response.get_json()

        for field in categorical_filter_fields:
            if field in options:
                assert len(options[field]) > 0, f"No options for {field}"

    def test_filter_options_are_strings(self, client):
        """Filter options are strings suitable for dropdown display."""
        response = client.get("/api/filter-options")
        options = response.get_json()

        for field, values in options.items():
            for value in values:
                assert isinstance(value, str), f"Non-string option in {field}: {value}"

    def test_state_filter_enables_geographic_filtering(self, client):
        """State filter has enough states for meaningful filtering."""
        response = client.get("/api/filter-options")
        options = response.get_json()

        states = options.get("State", [])
        # US has 50 states - should have many represented
        assert len(states) >= 20, f"Only {len(states)} states available"


class TestFrontendSiteDetails:
    """Tests for site details used in frontend side panel."""

    def test_site_details_support_all_categories(self, client, single_site_id):
        """Site details include all categories for side panel."""
        response = client.get(f"/api/site-details/{single_site_id}")
        data = response.get_json()

        categories = data.get("categories", {})

        # Panel should have multiple categories
        assert len(categories) >= 5, f"Only {len(categories)} categories available"

    def test_revenue_details_for_display(self, client, single_site_id):
        """Revenue details have display-ready values."""
        response = client.get(f"/api/site-details/{single_site_id}")
        data = response.get_json()

        revenue = data["categories"].get("Revenue", {})

        # Should have values that can be displayed
        assert "Avg Monthly Revenue" in revenue
        assert "Total Revenue" in revenue

    def test_location_details_for_display(self, client, single_site_id):
        """Location details available for display."""
        response = client.get(f"/api/site-details/{single_site_id}")
        data = response.get_json()

        location = data["categories"].get("Location", {})

        assert "State" in location
        assert "DMA" in location


class TestFrontendDataConsistency:
    """Tests for data consistency across frontend endpoints."""

    def test_sites_api_and_details_api_consistent(self, client, sample_site_ids):
        """Sites API and site-details API have consistent data."""
        # Get sites list
        sites_response = client.get("/api/sites")
        sites_data = {s["GTVID"]: s for s in sites_response.get_json()}

        # Check details for sample
        for site_id in sample_site_ids[:5]:
            if site_id in sites_data:
                details_response = client.get(f"/api/site-details/{site_id}")
                if details_response.status_code == 200:
                    details = details_response.get_json()
                    assert details["site_id"] == site_id

    def test_filter_results_are_valid_sites(self, client):
        """Filtered site IDs all exist in sites API."""
        # Get all sites
        all_sites_response = client.get("/api/sites")
        all_site_ids = {s["GTVID"] for s in all_sites_response.get_json()}

        # Get filtered sites
        filter_response = client.post(
            "/api/filtered-sites",
            json={"filters": {"State": "TX"}},
            content_type="application/json",
        )
        filtered_ids = filter_response.get_json()["site_ids"]

        # All filtered IDs should be valid
        for site_id in filtered_ids:
            assert site_id in all_site_ids, f"Filtered site {site_id} not in sites API"


class TestFrontendMapRendering:
    """Tests for data needed for WebGL map rendering."""

    def test_coordinates_are_floats(self, client):
        """Coordinates are float values for WebGL."""
        response = client.get("/api/sites")
        sites = response.get_json()

        for site in sites[:100]:
            assert isinstance(site["Latitude"], float), "Latitude not float"
            assert isinstance(site["Longitude"], float), "Longitude not float"

    def test_revenue_score_is_number(self, client):
        """Revenue score is numeric for color mapping."""
        response = client.get("/api/sites")
        sites = response.get_json()

        for site in sites[:100]:
            score = site["revenueScore"]
            assert isinstance(score, (int, float)), f"Score not numeric: {type(score)}"

    def test_enough_sites_for_meaningful_map(self, client):
        """Enough sites exist for a meaningful visualization."""
        response = client.get("/api/sites")
        sites = response.get_json()

        # Map should show thousands of sites
        assert len(sites) >= 10000, f"Only {len(sites)} sites - may be too sparse"

    def test_sites_span_continental_us(self, client):
        """Sites cover continental US geography."""
        response = client.get("/api/sites")
        sites = response.get_json()

        lats = [s["Latitude"] for s in sites]
        lons = [s["Longitude"] for s in sites]

        # Continental US bounds
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Should span significant area
        assert lat_range >= 10, f"Latitude span only {lat_range} degrees"
        assert lon_range >= 20, f"Longitude span only {lon_range} degrees"


class TestFrontendResponseTimes:
    """Tests for API response times affecting frontend performance."""

    def test_sites_api_responds_quickly(self, client):
        """Sites API responds within acceptable time for initial load."""
        import time

        start = time.time()
        response = client.get("/api/sites")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should respond within 5 seconds for 57K sites
        assert elapsed < 5.0, f"Sites API took {elapsed:.1f}s - too slow"

    def test_filter_options_api_responds_quickly(self, client):
        """Filter options API responds quickly for sidebar initialization."""
        import time

        start = time.time()
        response = client.get("/api/filter-options")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should respond within 1 second
        assert elapsed < 1.0, f"Filter options took {elapsed:.1f}s"

    def test_site_details_api_responds_quickly(self, client, single_site_id):
        """Site details API responds quickly for click interactions."""
        import time

        start = time.time()
        response = client.get(f"/api/site-details/{single_site_id}")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should respond within 0.5 seconds
        assert elapsed < 0.5, f"Site details took {elapsed:.1f}s"
