from sqlalchemy import create_engine, inspect
engine = create_engine('duckdb:///ecommerce.duckdb')
insp = inspect(engine)
print('orders FKs:', insp.get_foreign_keys('orders'))
print('order_items FKs:', insp.get_foreign_keys('order_items'))
print('support_tickets FKs:', insp.get_foreign_keys('support_tickets'))
