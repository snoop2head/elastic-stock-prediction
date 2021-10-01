import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Korean font path designation
import matplotlib.font_manager as fm

fontpath = "./font/NanumBarunGothic.ttf"
font = fm.FontProperties(fname=fontpath, size=14)

# apply retina display for clearer visualization
# %config InlineBackend.figure_format = 'retina'


def visualize_4x4_grid(
    root_path="./open",
    stock_list_name="Stock_list.csv",
    prediction_file_name="ElasticNetCV.csv",
):
    stock_list = pd.read_csv(os.path.join(root_path, stock_list_name))
    stock_list["종목코드"] = stock_list["종목코드"].apply(lambda x: str(x).zfill(6))

    df_result_all = pd.read_csv(os.path.join(root_path, prediction_file_name))

    # sample 9 companies from stock_list
    SAMPLE_NUM = 16
    sampled_data = stock_list.sample(SAMPLE_NUM)

    # visualize 16 companies subplots from df_result_all
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    # make margin between figures
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # visualize df_result_all
    for i, code in enumerate(sampled_data["종목코드"]):
        ax[i // 4, i % 4].plot(df_result_all.loc[:, code])

        # find the matching row of sampled_data from code
        company_name = stock_list[stock_list["종목코드"] == code]["종목명"].values[0]
        ax[i // 4, i % 4].set_title(company_name, fontproperties=font)
        ax[i // 4, i % 4].set_xlabel("Date")
        ax[i // 4, i % 4].set_ylabel("Close Price")
        ax[i // 4, i % 4].grid()
