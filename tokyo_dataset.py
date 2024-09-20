import pandas as pd
import numpy as np

def classification_to_bandit_problem(contexts, labels, num_actions=None):
  """Normalize contexts and encode deterministic rewards."""

  if num_actions is None:
    num_actions = np.max(labels) + 1
  num_contexts = contexts.shape[0]

  # Due to random subsampling in small problems, some features may be constant
  sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

  # Normalize features
  contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

  # One hot encode labels as rewards
  rewards = np.zeros((num_contexts, num_actions))
  rewards[np.arange(num_contexts), labels] = 1.0

  return contexts, rewards, (np.ones(num_contexts), labels)


def safe_std(values):
  """Remove zero std values for ones."""
  return np.array([val if val != 0.0 else 1.0 for val in values])

def get_tokyo_dataset():
    df = pd.read_csv('datasets/Tokyo_dataset.csv')
    num_actions = len(df['venueCategoryId'].unique())
    users = [0, 1]
    labels = df[df.columns[-1]].astype('category').cat.codes.as_matrix()
    df = df.drop([df.columns[-1]], axis=1)
    contexts = df.as_matrix()

    sampled_vals = classification_to_bandit_problem(contexts, labels, num_actions)

    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]

    return dataset, opt_rewards, opt_actions, num_actions, context_dim
