import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'    

def get_base_class_statistics(file):
    base_means = []
    base_cov = []
    data = pickle.load(file)
    
    # get mean and covariances of base classes
    for key in data.keys():
        feature = torch.tensor(data[key])
        base_means.append(torch.mean(feature, axis = 0))
        np_feature = np.array(data[key])
        np_cov = np.cov(np_feature.T)
        cov = torch.from_numpy(np_cov)
        base_cov.append(cov)
    
    # convert from numpy arrays to torch tensors
    torch_means = torch.zeros(len(base_means), len(base_means[0])).to(device)
    torch_cov = torch.zeros(len(base_cov), len(base_cov[0]), len(base_cov[0][0])).to(device)
    for i in range(len(base_means)):
        torch_means[i] = base_means[i].clone().detach()
        for j in range(len(base_cov[0])):
            torch_cov[i][j] = base_cov[i][j].clone().detach()
    
    return torch_means, torch_cov

def transform_tukey(data, lam, tukey):
    if tukey == False:
        return data.clone().detach()
    transformed_data = torch.zeros(data.shape).to(device)
    for i in range(len(data)):
        feature = data[i]
        if lam != 0:
            transformed_data[i] = torch.pow(feature, lam)
        else:
            transformed_data[i] = torch.log(feature)
    return transformed_data

def calibrate(base_means, base_cov, feature, k, alpha):
    distances = torch.zeros(len(base_means)).to(device)
    for i in range(len(base_means)):
        distances[i] = torch.cdist(feature.unsqueeze(0), base_means[i].unsqueeze(0))
    _, indices = torch.topk(distances, k)
    selected_means = torch.index_select(base_means, 0, indices)
    selected_cov = torch.index_select(base_cov, 0, indices)
    calibrated_mean = (torch.sum(selected_means, axis=0) + feature) / (k + 1)
    calibrated_cov = torch.sum(selected_cov, axis=0) / k + alpha
    return calibrated_mean, calibrated_cov
    
def get_accuracy(ndatas, labels, lam, n_generation, tukey):  
    acc = []
    # run algorithm on each task 
    for task, labels in tqdm(zip(ndatas, labels)):
        tukey_data = transform_tukey(task, lam, tukey)
        # separate train data and labels from tests
        support_data = tukey_data[:n_train]
        support_labels = labels[:n_train]
        test_data = tukey_data[n_train:]
        test_labels = labels[n_train:]
        
        train_data = support_data.clone()
        train_labels = support_labels.clone()

        # sample from calibrated distribution
        for feature, label in zip(support_data, support_labels):
            calibrated_mean, calibrated_cov = calibrate(base_means, base_cov, feature, 2, alpha)
            sampled_data_numpy = np.random.multivariate_normal(mean=calibrated_mean, cov=calibrated_cov, size=n_generation)
            sampled_data = torch.tensor(sampled_data_numpy)
            sampled_labels = torch.ones(n_generation).to(device)
            sampled_labels = sampled_labels * label
            # bring all the data required for training together
            train_data = torch.cat((train_data, sampled_data), dim=0)
            train_labels = torch.cat((train_labels, sampled_labels), dim=0)
        
        classifier = LogisticRegression(max_iter=1000).fit(X=train_data.cpu(), y=train_labels.cpu())
        predicts = classifier.predict(test_data.cpu())
        count = 0
        for predict, true in zip(predicts, test_labels):
            if predict == true:
                count += 1
        acc.append(count/len(predicts))
    
    return np.mean(acc)

import matplotlib.pyplot as plt
#def vary_lambda(ndatas, labels):
  #lambdas = [-2, -1, -0.5, 0, 0.5, 1, 2]

  # accs_no_gen = []
  # n_generation = 0
  # for lam in lambdas:
  #   accs_no_gen.append(get_accuracy(ndatas, labels, lam, n_generation))

  #accs_with_gen = []
  #n_generation = int(750 / n_shot)
  # for lam in lambdas:
  #accs_with_gen.append(get_accuracy(ndatas, labels, -2, n_generation))

  #plt.figure(figsize=(10, 10))
  #plt.plot(accs, label=' training w/ generated features')
  # plt.plot([3, 4, 5, 6, 7, 8, 7, 6, 5], label='training w/o generated features')
  #plt.xlabel('Lambda parameter in Tukey transformation', fontsize=13)
  #plt.ylabel('Test accuracy (5way-1shot)', fontsize=13)
  #plt.legend(prop={'size': 12})

  #plt.savefig('lambda variation.png')
  
def vary_n_generation(ndatas, labels): 
  n_generations = [0, 10, 50, 100, 150, 300, 500, 650, 750]

  tukey = False
  accs_no_tukey = []
  for n_generation in n_generations:
    acc = get_accuracy(ndatas, labels, lam, n_generation, tukey)
    accs_no_tukey.append(acc)
    print(acc)
    
  tukey = True
  accs_with_tukey = []
  for n_generation in n_generations:
    acc = get_accuracy(ndatas, labels, lam, n_generation, tukey)
    accs_with_tukey.append(acc)
    print(acc)
  plt.figure(figsize=(10, 10))
  plt.plot(n_generations, accs_no_tukey, label='training w/o Tukey transformation')
  plt.plot(n_generations, accs_with_tukey, label='training w Tukey transformation')

  plt.xlabel('Number of generated features per class', fontsize=13)
  plt.ylabel('Test accuracy (5way-1shot)', fontsize=13)
  plt.legend(prop={'size': 12})

  plt.savefig('n generation variation.png')
  
from sklearn.manifold import TSNE
def generate_clusters(ndatas, labels):
    counter = 0
    for task, labels in tqdm(zip(ndatas, labels)):
        tukey_data = transform_tukey(task, lam, tukey)
        # separate train data and labels from tests
        support_data = tukey_data[:n_train]
        support_labels = labels[:n_train]
        test_data = tukey_data[n_train:]
        test_labels = labels[n_train:]
        
        train_data = support_data.clone()
        train_labels = support_labels.clone()
        
        colours = ['b', 'g', 'r', 'c', 'k']
        # sample from calibrated distribution
        for feature, label in zip(support_data, support_labels):
            calibrated_mean, calibrated_cov = calibrate(base_means, base_cov, feature, 2, alpha)
            sampled_data_numpy = np.random.multivariate_normal(mean=calibrated_mean, cov=calibrated_cov, size=n_generation)
            sampled_data = torch.tensor(sampled_data_numpy)
            sampled_labels = torch.ones(n_generation).to(device)
            sampled_labels = sampled_labels * label
            # bring all the data required for training together
            train_data = torch.cat((train_data, sampled_data), dim=0)
            train_labels = torch.cat((train_labels, sampled_labels), dim=0)
            
            tsne = TSNE(n_components=2, perplexity=3, early_exaggeration=30, n_iter=5000).fit_transform(sampled_data)
            plt.scatter(tsne[:,0], tsne[:,1], c=colours[label], s=0.5)
            
        tsne = TSNE(n_components=2).fit_transform(support_data)
        for i in range(len(support_labels)):
            plt.scatter(tsne[i,0], tsne[i,1], c=colours[support_labels[i]])
        plt.savefig('cluster' + str(counter) + '.png')
        plt.close()
        counter += 1

# parameter initialization
dataset = 'miniImagenet'
n_shot = 1
n_ways = 5
n_queries = 15
n_runs = 5
n_train = n_ways * n_shot
n_test = n_ways * n_queries
n_total = n_train + n_test
lam = 0.5
alpha = 0.21
n_generation = int(750 / n_shot)

if __name__ == '__main__':
    
    # code from authors' github, gets data split as tasks
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_total, -1).to(device)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs,
                n_shot + n_queries, 5).clone().view(n_runs, n_total).to(device)
    base_means = None
    base_cov = None
    base_features_path = "checkpoints/%s/base_features.plk"%dataset
    with open(base_features_path, 'rb') as input_file:
        base_means, base_cov = get_base_class_statistics(input_file)
        
    from torch.distributions.multivariate_normal import MultivariateNormal
    
    vary_n_generation(ndatas, labels)
    #generate_clusters(ndatas, labels)
    #print(get_accuracy(ndatas, labels, lam, n_generation, True))
