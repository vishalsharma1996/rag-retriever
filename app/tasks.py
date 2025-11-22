from app.celery_worker import celery_app
import re
@celery_app.task(name="preprocess_query_batch", queue="cpu")
def preprocess_query_batch(queries, company_map, reverse_company_map):
    """
    Preprocess a batch of textual queries and detect associated tickers.

    Args:
        queries (list[str]): List of raw user queries.
        company_map (dict): Maps company names -> tickers. Example: {"apple": "AAPL"}
        reverse_company_map (dict): Maps tickers -> company names. Example: {"AAPL": "apple"}

    Returns:
        processed_queries (list[str]): Cleaned and possibly modified queries.
        tickers (list[str]): Detected tickers or "No Matches Found".
    """
    processed_queries = []
    tickers = []

    for query in queries:
        query = query.lower().replace("`s",'').replace("'s",'')
        query = query.replace('%',' percent ').replace('#',' number ').replace('$',' dollar ')
        query = re.sub('[^0-9a-z ]','',query)
        query = query.strip()
        words = query.split()

        # Case 1
        first = [company_map[w] for w in words if w in company_map]
        if first:
            processed_queries.append(query)
            tickers.append(first[0])
            continue

        # Case 2
        first = [w for w in words if w in reverse_company_map]
        if first:
            ticker = first[0]
            modified = query.replace(ticker, reverse_company_map[ticker])
            processed_queries.append(modified)
            tickers.append(ticker)
            continue

        # Case 3
        processed_queries.append(query)
        tickers.append("No Matches Found")

    return processed_queries, tickers
