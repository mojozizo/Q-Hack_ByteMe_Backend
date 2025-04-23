import time

import requests

from etl.transform.parsers.abstract_parser import AbstractParser
from etl.util.token_util import get_brightdata_token, get_brightdata_dataset_id


class LinkedInParser(AbstractParser):
    """
    Parser for LinkedIn profiles via Bright Data's dataset API.

    Requires two environment variables:
      - BRIGHTDATA_API_TOKEN: Your Bright Data API token.
      - BRIGHTDATA_DATASET_ID: The unique ID of the Bright Data dataset to use (found in your Bright Data dashboard under Datasets).
    """

    def __init__(self, poll_interval: float = 30.0, timeout: float = 180.0):
        super().__init__()
        self.api_token = get_brightdata_token()
        self.dataset_id = get_brightdata_dataset_id()
        self.base_url = "https://api.brightdata.com/datasets/v3"
        self.snapshot_base = "https://api.brightdata.com/datasets/v3/snapshot"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.headers_download = {
            "Authorization": f"Bearer {self.api_token}",
        }
        self.poll_interval = poll_interval
        self.timeout = timeout

    def parse(self) -> dict:
        """
        This method is not used. Use parse_by_url or parse_by_name instead.
        """
        raise NotImplementedError("Use parse_by_url or parse_by_name instead.")

    def _trigger(self, payload: list, discover_by: str) -> str:
        """
        Sends a trigger request and returns the snapshot_id.
        Uses 'discover_new' type for name-based discovery, 'trigger' for URL-based.
        """
        params = {
            "dataset_id": self.dataset_id,
            "include_errors": "true",
            "discover_by": discover_by
        }
        # set type according to discover_by
        if discover_by == "name":
            params["type"] = "discover_new"

        resp = requests.post(
            f"{self.base_url}/trigger",
            headers=self.headers,
            params=params,
            json=payload
        )
        resp.raise_for_status()
        return resp.json()["snapshot_id"]

    def _fetch_snapshot(self, snapshot_id: str) -> dict:
        """
        Retrieves the completed snapshot via the download endpoint.
        """
        download_url = f"{self.snapshot_base}/{snapshot_id}"
        start = time.time()
        while True:
            dl_resp = requests.get(download_url, headers=self.headers_download)
            dl_resp.raise_for_status()

            if dl_resp.json().get("status") == "complete" or dl_resp.json().get("id") is not None:
                # when searching by name, the status is not in response json, therefore we check for id
                print("completed123123123")
                return dl_resp.json()
            print(dl_resp.json())
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Snapshot {snapshot_id} not ready after {self.timeout}s")
            time.sleep(self.poll_interval)

    def parse_by_url(self, profile_url: str) -> dict:
        """
        Fetch LinkedIn profile data by direct URL.

        Args:
            profile_url: The public LinkedIn profile URL.

        Returns:
            dict: Parsed profile data.
        """
        payload = [{"url": profile_url}]
        snapshot_id = self._trigger(payload, discover_by="url")
        return self._fetch_snapshot(snapshot_id)

    def parse_by_name(self, first_name: str, last_name: str, company_name: str) -> dict:
        """
        Discover LinkedIn profile by person name and company.

        Args:
            first_name: Given name.
            last_name: Family name.
            company_name: Employer to narrow search. - doesnt work right now

        Returns:
            dict: Parsed profile data.
        """
        payload = [{
            "first_name": first_name,
            "last_name": last_name,
            #"company": company_name
        }]
        snapshot_id = self._trigger(payload, discover_by="name")
        return self._fetch_snapshot(snapshot_id)