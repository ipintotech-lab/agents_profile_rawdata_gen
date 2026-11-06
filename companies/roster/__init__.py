"""The five companies, ten agents each."""
from . import fintech, food_factory, law_office, logistics, supermarket

COMPANIES = {c.key: c for c in (fintech.COMPANY, food_factory.COMPANY, law_office.COMPANY, supermarket.COMPANY,
                                logistics.COMPANY)}
