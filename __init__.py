# /home/guyh/hys/__init__.py

import os
import sys
import torch

# Add the current directory to sys.path.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from quantdata_get import QuantConfig, get_quant_windows_for_symbol
from data_process import prepare_symbol_daily_data

try:
    import model
except ImportError:
    model = None

try:
    import result as result_mod
except ImportError:
    result_mod = None

# Global configuration Users may modify stock symbols, time range, and weighting parameters.
SYMBOLS = [stock code]  # e.g., SYMBOLS = ["A601088", "A600938", "A601857"]

START_DATE = "2023-01-01"
END_DATE = "2024-03-01"

# Historical window length in trading days.
T = 20

# Dimension of text embedding vectors.
EMBED_DIM = 768

# Weights assigned to news and QA embeddings.
W_NEWS = 0.2
W_QA = 0.8

# Whether to regenerate FinBERT-based text embeddings. This process is computationally expensive and is disabled by default.
RUN_TEXT_EMB = False

# Whether to execute model training/prediction and evaluation after data preparation.
RUN_MODEL = True
RUN_EVAL = True

def main():
    print(">>> HYS system started.")

    # ============================================================
    # STEP 1 (Optional):
    # Generate FinBERT-based text embeddings.
    # This step is computationally intensive and is disabled by default.
    # ============================================================
    # if RUN_TEXT_EMB:
    #     print(">>> [STEP 1] Building text embeddings with FinBERT ...")
    #     from text_embedding import build_all_embeddings
    #
    #     build_all_embeddings(
    #         model_name_or_path=os.path.join(CURRENT_DIR, "finbert-tone"),
    #         data_dir=CURRENT_DIR,
    #         output_dir=os.path.join(CURRENT_DIR, "daily_vecs"),
    #         num_gpus=1,
    #         batch_size=128,
    #         max_length=128,
    #         chunk_size=20000,
    #         progress=True,
    #     )
    #     print(">>> [STEP 1] Text embeddings generated and saved to daily_vecs/")
    # else:
    #     print(">>> [STEP 1] Skipped. Existing embeddings in daily_vecs/ will be used.")

    # ============================================================
    # STEP 2:
    # Retrieve quantitative market data and construct T-day windows.
    # Failed symbols are skipped without interrupting the pipeline.
    # ============================================================
    print("\n>>> [STEP 2] Building quantitative windows ...\n")

    quant_cfg = QuantConfig()

    all_quant_results = {}
    failed_syms = []

    for sym in SYMBOLS:
        print(f">>> Processing quantitative data for {sym} ...")
        try:
            res = get_quant_windows_for_symbol(
                symbol=sym,
                start_date=START_DATE,
                end_date=END_DATE,
                T=T,
                config=quant_cfg,
            )
            all_quant_results[sym] = res
        except Exception as e:
            print(f"[STEP 2] Failed to retrieve data for {sym}. Symbol skipped. Reason: {e}")
            failed_syms.append(sym)
            continue

    print(
        f"\n>>> [STEP 2] Quantitative windows generated. "
        f"Successful: {len(all_quant_results)}, Failed: {len(failed_syms)}.\n"
    )

    if failed_syms:
        fail_path = os.path.join(CURRENT_DIR, "failed_symbols_step2.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_syms))
        print(f">>> [STEP 2] Failure log saved to: {fail_path}\n")

    if len(all_quant_results) == 0:
        print(
            ">>> [STEP 2] No symbols were processed successfully. "
            "Subsequent steps cannot be executed."
        )
        return

    # ============================================================
    # STEP 3:
    # Align textual and quantitative information to generate final
    # daily input representations for the prediction model.
    # Results are saved to disk for subsequent use.
    # ============================================================
    print(">>> [STEP 3] Building final per-day model input vectors ...\n")

    chinadaily_dir = os.path.join(CURRENT_DIR, "daily_vecs", "chinadaily")
    qa_dir = os.path.join(CURRENT_DIR, "daily_vecs", "qa")
    save_dir = os.path.join(CURRENT_DIR, "daily_final")
    os.makedirs(save_dir, exist_ok=True)

    failed_step3 = []

    for sym, quant_res in all_quant_results.items():
        print(f">>> Aligning textual and quantitative data for {sym} ...")
        try:
            final_data = prepare_symbol_daily_data(
                symbol=sym,
                quant_res=quant_res,
                w_news=W_NEWS,
                w_qa=W_QA,
                embed_dim=EMBED_DIM,
                chinadaily_dir=chinadaily_dir,
                qa_dir=qa_dir,
            )

            text_vecs = final_data["text_vecs"]
            quant_vecs = final_data["quant_vecs"]

            print(
                f">>> {sym} Final daily data shape: "
                f"{text_vecs.shape} {quant_vecs.shape}"
            )

            save_path = os.path.join(save_dir, f"{sym}_daily.pt")
            torch.save(final_data, save_path)
            print(f">>> {sym} Daily final vectors saved to: {save_path}\n")

        except Exception as e:
            print(f"[STEP 3] Failed during alignment or saving for {sym}. Symbol skipped. Reason: {e}")
            failed_step3.append(sym)
            continue

    print(">>> [STEP 3] Data preparation completed.")
    print(">>> Daily representations are available in daily_final/<symbol>_daily.pt.\n")

    if failed_step3:
        fail3_path = os.path.join(CURRENT_DIR, "failed_symbols_step3.txt")
        with open(fail3_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_step3))
        print(f">>> [STEP 3] Failure log saved to: {fail3_path}\n")

    # ============================================================
    # STEP 4:
    # Execute model training and prediction.
    # ============================================================
    if RUN_MODEL:
        if model is None:
            print(">>> [STEP 4] model.py not found. Model execution skipped.")
        else:
            print(">>> [STEP 4] Executing model training and prediction ...")
            if hasattr(model, "main"):
                model.main()
            else:
                print(">>> [STEP 4] main() function not found in model.py.")
            print(">>> [STEP 4] Model execution completed.\n")
    else:
        print(">>> [STEP 4] Model training and prediction skipped because RUN_MODEL=False.\n")

    # ============================================================
    # STEP 5:
    # Execute performance evaluation.
    # ============================================================
    if RUN_EVAL:
        if result_mod is None:
            print(">>> [STEP 5] result.py not found. Evaluation skipped.")
        else:
            print(">>> [STEP 5] Running evaluation ...")
            if hasattr(result_mod, "main"):
                result_mod.main()
            else:
                print(">>> [STEP 5] main() function not found in result.py.")
            print(">>> [STEP 5] Evaluation completed.\n")
    else:
        print(">>> [STEP 5] Evaluation skipped because RUN_EVAL=False.\n")

    print(">>> Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
