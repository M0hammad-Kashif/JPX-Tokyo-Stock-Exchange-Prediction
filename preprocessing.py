import tensorflow as tf
from tqdm import tqdm

from config import *
from utils import *

df_price = pd.read_csv(f"{train_files_dir}/stock_prices.csv")
df_price_supplemental = pd.read_csv(f"{supplemental_files_dir}/stock_prices.csv")

df_price = pd.concat([df_price, df_price_supplemental])

# generate AdjustedClose
df_price = adjust_price(df_price)

lista = pd.read_csv(f"{base_dir}/stock_list.csv")
sectores = lista[["SecuritiesCode"]]
df = sectores.loc[sectores.set_index('SecuritiesCode').index.isin(df_price.set_index('SecuritiesCode').index)]

df_price = df_price.reset_index().merge(df, how="left").set_index('Date')

codes = sorted(df_price["SecuritiesCode"].unique())
buff = []
print("Creating Features")
for code in tqdm(codes):
    feat = get_features_for_predict(df_price, code)
    buff.append(feat)
feature = pd.concat(buff)

feature.drop(["High", "Low"], axis=1, inplace=True)
feature['month'] = pd.Categorical(feature['month'])
feature['day'] = pd.Categorical(feature['day'])
feature['dow'] = pd.Categorical(feature['dow'])

X_train, y_train, X_test, y_test = get_features_and_label(
    df_price, codes, feature
)
X_train = X_train.reset_index(drop=True)
X_test = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_train.reset_index(drop=True)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

BATCH_SIZE = 256
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
