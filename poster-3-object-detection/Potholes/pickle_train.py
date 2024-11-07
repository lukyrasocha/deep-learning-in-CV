import pickle 


with open('train_proposals/train_image.pkl', 'rb') as f:
    train_image = pickle.load(f)
with open('train_proposals/train_target.pkl', 'rb') as f:
    train_target = pickle.load(f)


print(len(train_image))
print(len(train_target))