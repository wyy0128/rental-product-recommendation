import pandas as pd

TEST_VISITS = "C:/pycharm/DL_Project_1/rental-product-recommendation-system/metrika_visits_test.csv"
OUT_SUB = "C:/pycharm/DL_Project_1/rental-product-recommendation-system/submission_itemcf.csv"

test_df = pd.read_csv(TEST_VISITS, sep=None, engine="python", dtype=str)
sub_df  = pd.read_csv(OUT_SUB, dtype=str)

# 清理空格
test_ids = set(test_df["visit_id"].str.strip())
sub_ids  = set(sub_df["visit_id"].str.strip())

print("test rows:", len(test_df), "unique:", len(test_ids))
print("sub  rows:", len(sub_df),  "unique:", len(sub_ids))

print("missing in submission:", len(test_ids - sub_ids))
print("extra in submission:", len(sub_ids - test_ids))

# 打印几个 Kaggle 提示缺失的 id 是否在 test 里
need = ["5932712846819852575","4635160539250819387","5901764679479329100"]
for x in need:
    print(x, "in test?", x in test_ids, "| in sub?", x in sub_ids)
