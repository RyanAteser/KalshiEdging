import zipfile, io, pandas as pd, sys

zip_path = sys.argv[1] if len(sys.argv) > 1 else r"E:\april_lastweek_15m (1).zip"

with zipfile.ZipFile(zip_path) as zf:
    all_names = zf.namelist()
    prices = [n for n in all_names if "prices_eth" in n and n.endswith(".parquet")]
    book   = [n for n in all_names if "book_eth"   in n and n.endswith(".parquet")]
    print(f"prices_eth files: {len(prices)}")
    print(f"book_eth   files: {len(book)}")

    if prices:
        print(f"\n-- prices sample: {prices[0]}")
        df = pd.read_parquet(io.BytesIO(zf.open(prices[0]).read()))
        print("columns:", df.columns.tolist())
        print(df.head(2).to_string())

    if book:
        print(f"\n-- book sample: {book[0]}")
        df2 = pd.read_parquet(io.BytesIO(zf.open(book[0]).read()))
        print("columns:", df2.columns.tolist())
        print(df2.head(2).to_string())
