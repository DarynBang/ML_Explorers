import glob

path_3ds = glob("MNIST/*3d*")

train_data    = []
val_data      = []
test_data     = []
train_labels  = []
val_labels    = []
test_labels   = []
labels_map    = {}
for idx, path in enumerate(path_3ds):
  data = np.load(path)
  data_train_images = data["train_images"]
  data_val_images   = data['val_images']
  data_test_images = data["test_images"]
  train_data.append(data_train_images)
  train_labels.append(np.ones(len(data_train_images))*idx)
  val_data.append(data_val_images)
  val_labels.append(np.ones(len(data_val_images))*idx)
  test_data.append(data_test_images)
  test_labels.append(np.ones(len(data_test_images))*idx)
  labels_map[path.split("/")[-1].split(".")[0]] = idx



train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)
val_data = np.concatenate(val_data)
val_labels = np.concatenate(val_labels)
test_data = np.concatenate(test_data)
test_labels = np.concatenate(test_labels)
# Shuffle the data and labels
train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
val_data, val_labels = shuffle(val_data, val_labels, random_state=42)
test_data, test_labels = shuffle(test_data, test_labels, random_state=42)
