from sec_edgar_api import EdgarClient
from etl.transform.parsers.abstract_parser import AbstractParser

class SecEdgarParser(AbstractParser):
    """
    Parser for SEC EDGAR using the sec-edgar-api client library.
    """

    def __init__(
        self,
        cik: str,
        user_agent: str = "MyApp/1.0 (contact@example.com)",
        handle_pagination: bool = True
    ):
        super().__init__()
        self.cik = cik.zfill(10)
        self.client = EdgarClient(user_agent=user_agent)
        self.handle_pagination = handle_pagination

    def parse(self) -> dict:
        """
        Fetches and parses the SEC EDGAR submissions via the client.

        Returns:
            dict: Parsed data including company metadata and recent filings.
        """
        subs = self.client.get_submissions(
            cik=self.cik,
            handle_pagination=self.handle_pagination
        )

        company_name = subs.get("name")
        sic = subs.get("sic")
        sic_desc = subs.get("sicDescription")

        recent = subs.get("filings", {}).get("recent", {})
        keys = [
            "accessionNumber", "filingDate", "reportDate",
            "acceptanceDateTime", "act", "form",
            "fileNumber", "filmNumber", "items", "size",
            "isXBRL", "isInlineXBRL", "primaryDocument",
            "primaryDocDescription"
        ]
        arrays = [recent.get(k, []) for k in keys]

        filings = []
        for values in zip(*arrays):
            (acc, fdate, rdate, atime, act, form, fileno, filmno,
             items, size, isx, isix, doc, docdesc) = values
            filings.append({
                "accessionNumber": acc,
                "filingDate": fdate,
                "reportDate": rdate,
                "acceptanceDateTime": atime,
                "act": act,
                "form": form,
                "fileNumber": fileno,
                "filmNumber": filmno,
                "items": items,
                "size": size,
                "isXBRL": isx,
                "isInlineXBRL": isix,
                "primaryDocument": doc,
                "primaryDocDescription": docdesc,
                "documentUrl": (
                    f"https://data.sec.gov/Archives/edgar/data/"
                    f"{int(self.cik)}/{acc.replace('-', '')}/{doc}"
                )
            })

        return {
            "cik": self.cik,
            "companyName": company_name,
            "sic": sic,
            "sicDescription": sic_desc,
            "filings": filings
        }

    def fetch_all_facts(self) -> dict:
        """
        Fetches all XBRL facts for the company via the companyfacts endpoint.

        Returns:
            dict: JSON from /api/xbrl/companyfacts/CIK##########.json
        """
        return self.client.get_company_facts(cik=self.cik)

    def fetch_concept(self, taxonomy: str, tag: str) -> dict:
        """
        Fetches XBRL facts for a specific taxonomy and tag.

        Args:
            taxonomy: e.g. "us-gaap" or "ifrs-full"
            tag:       e.g. "Revenues", "NetIncomeLoss"

        Returns:
            dict: JSON from /api/xbrl/companyconcept/CIK##########/{taxonomy}/{tag}.json
        """
        return self.client.get_company_concept(
            cik=self.cik,
            taxonomy=taxonomy,
            tag=tag
        )
