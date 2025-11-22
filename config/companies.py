def add_company_map():
  ''' Add company mapping for metadata filtering'''
  company_map = {
          'microsoft':'msft', 'adobe':'adbe', 'coupang':'cpng', 'linde':'lin',
          'oracle':'orcl', 'nvidia':'nvda', 'delta':'dal', 'tesla':'tsla',
          'netflix':'nflx', 'home':'hd', 'amazon':'amzn', 'apple':'aapl',
          'appl':'aapl', 'johnson':'jnj', 'jp':'jpm', 'visa':'v', 'unitedhealth':'unh',
          'google':'googl', 'alphabet':'googl', 'berkshire':'brka',
          'meta':'meta', 'pg':'pg'
      }
  reverse_company_map = {v: k for k, v in company_map.items()}
  return company_map, reverse_company_map
