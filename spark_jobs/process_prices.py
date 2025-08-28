import yfinance as yf
import pandas as pd
import os

# 環境変数設定（重要: WindowsのSparkでParquet書き込み対策）
try:
    import env_setup
except ImportError:
    pass  # なければ無視

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, round as spark_round
from pyspark.sql.window import Window

# Sparkセッション開始
spark = SparkSession.builder \
    .appName("StockPriceETL") \
    .master("local[*]") \
    .getOrCreate()

# 銘柄設定
TICKERS = ["VOD.L", "BP.L", "HSBA.L"]
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# 株価取得
def fetch_stock_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["Ticker"] = ticker
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# 全銘柄結合
all_data = pd.concat([fetch_stock_data(t) for t in TICKERS], ignore_index=True)

# pandas → Spark
spark_df = spark.createDataFrame(all_data).dropna()

# 日次リターン計算
window_spec = Window.partitionBy("Ticker").orderBy("Date")
spark_df = spark_df.withColumn("Prev_Close", lag("Close").over(window_spec))
spark_df = spark_df.withColumn(
    "Return(%)",
    spark_round((col("Close") - col("Prev_Close")) / col("Prev_Close") * 100, 2)
)

# 保存先ディレクトリの存在確認
output_path = "../data/processed/stock_prices.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Parquet保存
spark_df.write.mode("overwrite").parquet(output_path)

print(f"✅ データ保存完了: {output_path}")

spark.stop()
