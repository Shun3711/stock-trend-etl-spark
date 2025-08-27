import yfinance as yf
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, round as spark_round
from pyspark.sql.window import Window
import os

os.environ["JAVA_HOME"] = r"C:\\Users\\Pupi\\AppData\\Local\\Programs\\Eclipse Adoptium\\jdk-8.0.452.9-hotspot"
os.environ["HADOOP_HOME"] = r"C:\\hadoop"
os.environ["SPARK_HOME"] = r"C:\\spark\\spark-3.3.2-bin-hadoop3"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Test") \
    .master("local[*]") \
    .getOrCreate()

# 対象銘柄（イギリス株）
TICKERS = ["VOD.L", "BP.L", "HSBA.L"]
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Sparkセッションの開始
spark = SparkSession.builder \
    .appName("StockPriceETL") \
    .getOrCreate()

def fetch_stock_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)
    df = df.reset_index()
    df["Ticker"] = ticker
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]


# 全銘柄のデータを結合
all_data = pd.concat([fetch_stock_data(ticker) for ticker in TICKERS])

# pandas → Sparkへ変換
spark_df = spark.createDataFrame(all_data)

# 欠損除去
spark_df = spark_df.dropna()

# 日次リターン（%）を計算
window_spec = Window.partitionBy("Ticker").orderBy("Date")
spark_df = spark_df.withColumn("Prev_Close", lag("Close").over(window_spec))
spark_df = spark_df.withColumn(
    "Return(%)",
    spark_round((col("Close") - col("Prev_Close")) / col("Prev_Close") * 100, 2)
)

# 出力パス
output_path = "../data/processed/stock_prices.parquet"

# 保存（Parquet形式）
spark_df.write.mode("overwrite").parquet(output_path)

print(f"✅ データ保存完了: {output_path}")

spark.stop()
