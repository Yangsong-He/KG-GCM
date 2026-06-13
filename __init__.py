# /home/guyh/hys/__init__.py

import os
import sys
import torch

# ---------------------------------------------------------------------
# 把当前目录加入 sys.path，方便直接 import 本目录下的模块
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ---------------------------------------------------------------------
# 导入自己写的模块
# ---------------------------------------------------------------------
from quantdata_get import QuantConfig, get_quant_windows_for_symbol
from data_process import prepare_symbol_daily_data

# 这里假设你已经有 model.py 和 result.py，并且里面都有 main()
# - model.main()：读取 daily_final/<symbol>_daily.pt，训练/预测，输出到 results/
# - result.main()：读取 results/<symbol>_results.pt，计算指标，输出到 result/
try:
    import model
except ImportError:
    model = None

try:
    import result as result_mod
except ImportError:
    result_mod = None

# ---------------------------------------------------------------------
# 全局配置（你后面可以在这里改股票代码 / 时间区间 / 权重）
# ---------------------------------------------------------------------
SYMBOLS = ["A601088", "A600938", "A601857", "A600028", "A600019", "A600941", "A601985", "A600900", "A003816"]

START_DATE = "2023-01-01"
END_DATE = "2024-03-01"

# 过去 T 天作为一个窗口
T = 20

# 文本词向量维度（FinBERT base 是 768）
EMBED_DIM = 768

# 文本加权：新闻 / QA 的权重
W_NEWS = 0.2
W_QA = 0.8
# 是否重新跑 FinBERT 文本向量（非常耗时，一般设 False）
RUN_TEXT_EMB = False

# 是否在数据准备完之后立刻跑模型和评估
RUN_MODEL = True
RUN_EVAL = True


def main():
    print(">>> HYS system started.")

    # # ============================================================
    # # STEP 1（可选）：重新构建 FinBERT 文本向量（很慢，默认关闭）
    # # ============================================================
    # if RUN_TEXT_EMB:
    #     print(">>> [STEP 1] Building text embeddings with FinBERT ...")
    #     from text_embedding import build_all_embeddings  # 惰性导入
    #
    #     build_all_embeddings(
    #         model_name_or_path=os.path.join(CURRENT_DIR, "finbert-tone"),
    #         data_dir=CURRENT_DIR,
    #         output_dir=os.path.join(CURRENT_DIR, "daily_vecs"),
    #         num_gpus=1,          # 多卡之前报 NCCL 错误，这里默认单卡稳定
    #         batch_size=128,
    #         max_length=128,
    #         chunk_size=20000,
    #         progress=True,
    #     )
    #     print(">>> [STEP 1] ✅ Text embeddings built and saved into daily_vecs/")
    # else:
    #     print(">>> [STEP 1] Skipped building text embeddings (use existing daily_vecs/*)")

    # ============================================================
    # STEP 2：获取量化数据 + 构造 T 日窗口（失败跳过，不崩）
    # ============================================================
    print("\n>>> [STEP 2] Building quantitative windows ...\n")

    quant_cfg = QuantConfig()

    all_quant_results = {}
    failed_syms = []

    for sym in SYMBOLS:
        print(f">>> Processing Quant Data for {sym} ...")
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
            print(f"[STEP 2] ❌ {sym} 获取失败，跳过。原因：{e}")
            failed_syms.append(sym)
            continue

    print(f"\n>>> [STEP 2] ✅ Quant windows built. 成功 {len(all_quant_results)} 只，失败 {len(failed_syms)} 只。\n")

    if failed_syms:
        fail_path = os.path.join(CURRENT_DIR, "failed_symbols_step2.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_syms))
        print(f">>> [STEP 2] 失败列表已保存: {fail_path}\n")

    if len(all_quant_results) == 0:
        print(">>> [STEP 2] ❌ 没有任何股票成功获取量化数据，后续步骤无法继续。")
        return

    # ============================================================
    # STEP 3：对齐文本 + 量化，得到每天最终的输入向量，并保存到文件
    #         只处理 STEP2 成功的股票
    # ============================================================
    print(">>> [STEP 3] Building final per-day model input vectors ...\n")

    chinadaily_dir = os.path.join(CURRENT_DIR, "daily_vecs", "chinadaily")
    qa_dir = os.path.join(CURRENT_DIR, "daily_vecs", "qa")
    save_dir = os.path.join(CURRENT_DIR, "daily_final")
    os.makedirs(save_dir, exist_ok=True)

    failed_step3 = []

    for sym, quant_res in all_quant_results.items():
        print(f">>> Aligning text + quant for {sym} ...")
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

            text_vecs = final_data["text_vecs"]   # (N, 768)
            quant_vecs = final_data["quant_vecs"] # (N, 768)

            print(
                f">>> {sym} ✅ Final daily data shape: "
                f"{text_vecs.shape} {quant_vecs.shape}"
            )

            save_path = os.path.join(save_dir, f"{sym}_daily.pt")
            torch.save(final_data, save_path)
            print(f">>> {sym} ✅ Saved daily final vectors to: {save_path}\n")

        except Exception as e:
            print(f"[STEP 3] ❌ {sym} 对齐/保存失败，跳过。原因：{e}")
            failed_step3.append(sym)
            continue

    print(">>> [STEP 3] ✅ ALL DATA PREPARED AND SAVED (for successful symbols).")
    print(">>> You can now feed daily_final/<symbol>_daily.pt into the prediction model.\n")

    if failed_step3:
        fail3_path = os.path.join(CURRENT_DIR, "failed_symbols_step3.txt")
        with open(fail3_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_step3))
        print(f">>> [STEP 3] 失败列表已保存: {fail3_path}\n")

    # ============================================================
    # STEP 4：调用 model.py 进行训练 / 预测
    # ============================================================
    if RUN_MODEL:
        if model is None:
            print(">>> [STEP 4] ❌ model.py 未找到，跳过模型训练与预测。")
        else:
            print(">>> [STEP 4] Running model.py (train / predict) ...")
            if hasattr(model, "main"):
                model.main()
            else:
                print(">>> [STEP 4] ⚠️ model.py 中未找到 main()，请确认。")
            print(">>> [STEP 4] ✅ model.py finished.\n")
    else:
        print(">>> [STEP 4] Skipped model training/prediction (RUN_MODEL=False)\n")

    # ============================================================
    # STEP 5：调用 result.py 进行评估
    # ============================================================
    if RUN_EVAL:
        if result_mod is None:
            print(">>> [STEP 5] ❌ result.py 未找到，跳过评估。")
        else:
            print(">>> [STEP 5] Running result.py (evaluation) ...")
            if hasattr(result_mod, "main"):
                result_mod.main()
            else:
                print(">>> [STEP 5] ⚠️ result.py 中未找到 main()，请确认。")
            print(">>> [STEP 5] ✅ result.py finished.\n")
    else:
        print(">>> [STEP 5] Skipped evaluation (RUN_EVAL=False)\n")

    print(">>> 🎯 PIPELINE DONE. All steps finished.")



if __name__ == "__main__":
    main()
