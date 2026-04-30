import duckdb
import argparse
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def analyze_dataset_priors(parquet_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyzes the oracle frequencies, mean WERs, and error correlations of base models 
    in a given parquet dataset using DuckDB for high-speed, low-memory execution.
    """
    if verbose:
        logger.info(f"Analyzing priors and error correlations for {parquet_path}...")
    else:
        # If not verbose, we still want to log the final tables but without the intro text
        pass

    # 1. Oracle Frequencies and Mean WER
    # We resolve ties using the order: Hubert -> Whisper -> W2V2
    oracle_query = f"""
    SELECT 
        oracle_model, 
        COUNT(*) as count, 
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM (
        SELECT 
            CASE 
                WHEN hubert_wer <= whisper_wer AND hubert_wer <= w2v2_wer THEN 'hubert'
                WHEN whisper_wer <= hubert_wer AND whisper_wer <= w2v2_wer THEN 'whisper'
                ELSE 'w2v2' 
            END AS oracle_model
        FROM '{parquet_path}'
    ) 
    GROUP BY oracle_model
    ORDER BY percentage DESC
    """

    # 2. Mean WER per Model
    mean_wer_query = f"""
    SELECT 
        ROUND(AVG(hubert_wer), 4) as hubert_mean,
        ROUND(AVG(whisper_wer), 4) as whisper_mean,
        ROUND(AVG(w2v2_wer), 4) as w2v2_mean
    FROM '{parquet_path}'
    """

    # 3. Error Correlation (Pearson)
    # Shows which models fail on the same acoustic difficulties
    corr_query = f"""
    SELECT 
        ROUND(corr(hubert_wer, w2v2_wer), 4) as hubert_w2v2_corr,
        ROUND(corr(hubert_wer, whisper_wer), 4) as hubert_whisper_corr,
        ROUND(corr(w2v2_wer, whisper_wer), 4) as w2v2_whisper_corr
    FROM '{parquet_path}'
    """
    
    try:
        oracle_df = duckdb.query(oracle_query).df()
        mean_wer_df = duckdb.query(mean_wer_query).df()
        corr_df = duckdb.query(corr_query).df()
    except Exception as e:
        logger.error(f"Failed to analyze dataset {parquet_path} using DuckDB: {e}")
        return {}

    # Print tables nicely
    if verbose:
        print("\n=== ORACLE FREQUENCY (ENTIRE DATASET) ===")
        print(oracle_df.to_string(index=False))
        
        print("\n=== MEAN WER (ENTIRE DATASET) ===")
        print(mean_wer_df.to_string(index=False))

        print("\n=== ERROR CORRELATION (MODEL COMPLEMENTARITY) ===")
        print(corr_df.to_string(index=False))
        print("\n")

    # Format the results to return as a dictionary
    results = {
        "oracle_frequencies": oracle_df.set_index("oracle_model").to_dict("index"),
        "mean_wer": mean_wer_df.iloc[0].to_dict(),
        "error_correlation": corr_df.iloc[0].to_dict()
    }
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyze oracle priors and correlations for a parquet dataset using DuckDB.")
    parser.add_argument("--parquet", type=str, required=True, help="Path to the processed parquet file")
    args = parser.parse_args()
    
    analyze_dataset_priors(args.parquet, verbose=True)
