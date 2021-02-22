from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv

def iris_data():
    # # iris
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x = MinMaxScaler().fit_transform(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9,random_state=0)
    headers = [f'param-{i}' for i in range(len(x_train[0]))]
    return x_train,x_test,y_train,y_test,headers

def digits_data():
    mnist = datasets.load_digits()
    x = mnist.data
    x = MinMaxScaler().fit_transform(x)
    y = mnist.target
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9,random_state=0)
    headers = [f'param-{i}' for i in range(len(x_train[0]))]
    return x_train,x_test,y_train,y_test,headers

def random_data(k_means):
    # custom
    x = []
    vec_size = 5
    n_points = 1000
    for i in range(n_points):
        vec = []
        # vec.append(i)
        for j in range(vec_size):
            vec.append((random.random() * 0.1) + math.sin((i * 0.01) + (j * (i%11) * 0.1)))
        vec.append(random.random())
        x.append(vec)
    y = KMeans(k_means).fit_predict(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9,random_state=0)
    headers = [f'param-{i}' for i in range(len(x_train[0]))]
    return x_train,x_test,y_train,y_test,headers

def audio_analysis(k_means):
    # audio analysis
    x = []
    with open('10_210106_001652_nrt_analysis.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x.append([float(val) for val in row[1:]])

    headers = []
    with open('04_full_dataset_column_headers.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            headers.append(row[1])

    x = MinMaxScaler().fit_transform(x)
    y = KMeans(k_means).fit_predict(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9,random_state=0)
    return x_train,x_test,y_train,y_test,headers

# ===============================================================
#                     actual script
# ================================================================
#                        params
# ================================================
# maybe these should be rougly equal... in terms of representation...
# so if we want about 1/10 of the original features, maybe pca param should be 0.1 ?
n_pca_to_use = 10 # how many PCAs components to sum the coefficients of
final_n_features_target = 10 # how many top features to pick out (e.g., how many features do you want to have?)
# ================================================

x_train, x_test, y_train, y_test, headers = audio_analysis(30)
# x_train, x_test, y_train, y_test, headers = digits_data() # k means = 10
# x_train, x_test, y_train, y_test, headers = random_data(25)

# svm
linearsvm = svm.SVC(kernel='linear',C=1,decision_function_shape='ovo').fit(x_train,y_train)
orig_svm_score = linearsvm.score(x_test,y_test)
print('svm score:',orig_svm_score)

# pca
pca = PCA(n_pca_to_use).fit(x_train)

def plot_abs_sum(data,n_components,title,print_top_n = None):
    if n_components == None:
        n_components = len(data)

    sums = np.zeros(len(data[0]))
    for i in range(n_components):
        sums += np.abs(data[i])

    plt.plot(sums)
    plt.title(title)

    sorted_list = []

    if print_top_n != None:
        sums_min = np.min(sums)
        sums_max = np.max(sums)
        sorted_idx = np.argsort(sums)
        sorted_idx = np.flip(sorted_idx)
        sorted_list = sorted_idx
        print(f'Top {print_top_n} features for {title}')
        for i, idx in enumerate(sorted_idx):
            if i < print_top_n:
                print(i,headers[idx])
                plt.plot([idx,idx],[sums_min,sums_max],'k--')
        print('')
    
    plt.show()

    return sorted_list

def plot_each_abs(data,each_label,title):
    for i, row in enumerate(data):
        plt.plot(np.abs(row),label=f'{each_label}{i}')

    plt.title(title)
    # plt.legend()
    plt.show()

# TSNE reduced just for fun
tsne_x = TSNE(2).fit_transform(x_train)
tsne_x = MinMaxScaler().fit_transform(tsne_x)
for i, pt in enumerate(tsne_x):
    col = f'C{y_train[i]}'
    # print(pt,col)
    plt.plot(pt[0],pt[1],c=col,marker='.')
plt.show()

# PCA 
print('')
print('pca components:')
print(pca.components_)
print(pca.components_.shape)

print('PCA Explained Variance Ratios')
rolling_sum = 0
for i, ratio in enumerate(pca.explained_variance_ratio_):
    rolling_sum += ratio
    print(i,ratio,rolling_sum)
print('')

# plot each might still be interesting to see
plot_each_abs(pca.components_,'PC','All PCs, abs')

# not using this...
plot_abs_sum(pca.components_,None,f'First {pca.n_components_} PCs, abs, summed',final_n_features_target)

pca_sorted_order = plot_abs_sum(pca.components_,None,f'All PCs ({pca.n_components_}), abs, summed',final_n_features_target)

# SVM
print('svm coef:')
print(linearsvm.coef_)
print(linearsvm.coef_.shape)

# plot each might be interesting to see
plot_each_abs(linearsvm.coef_,'SVM','All SVMs, abs')

# not using this
plot_abs_sum(linearsvm.coef_,pca.n_components_,f'First {pca.n_components_} SVMs, abs, summed',final_n_features_target)

svm_sorted_order = plot_abs_sum(linearsvm.coef_,None,'All SVMs, abs, summed',final_n_features_target)

# ===== COMPARE ? =========
# plt.plot(svm_sorted_order,label='SVM')
# plt.plot(pca_sorted_order,label='PCA')
# plt.legend()
# plt.title('Comparing the sorted ording')
# plt.show()

for i in range(len(svm_sorted_order)):
    print(i,headers[svm_sorted_order[i]],headers[pca_sorted_order[i]])

# print('svm sorted order',svm_sorted_order,'svm sorted order')
# print('pca sorted order',pca_sorted_order,'pca sorted order')
print('SVM ordering compared to PCA')
svm_headers_order = []
svm_indices_order = []
for i in range(final_n_features_target):
    svm_id = svm_sorted_order[i]
    pca_id = np.where(pca_sorted_order == svm_id)[0]
    plt.plot([i,pca_id],[0,1],'b--')
    print(i,headers[svm_id],pca_id)
    svm_indices_order.append(svm_id)
    svm_headers_order.append(headers[svm_id])

print('PCA ordering compared to SVM')
pca_headers_order = []
pca_indices_order = []
for i in range(final_n_features_target):
    pca_id = pca_sorted_order[i]
    svm_id = np.where(svm_sorted_order == pca_id)[0]
    plt.plot([i,svm_id],[1,0],'r--')
    print(i,headers[pca_id],svm_id)
    pca_indices_order.append(pca_id)
    pca_headers_order.append(headers[pca_id])

org = {'both':[],'pca_only':[],'svm_only':[]}
for header in svm_headers_order:
    if header in pca_headers_order:
        org['both'].append(header)
    else:
        org['svm_only'].append(header)
for header in pca_headers_order:
    if not header in svm_headers_order:
        org['pca_only'].append(header)

print('both',org['both'])
both_len = len(org['both'])
print(f'{both_len} / {final_n_features_target}\n')
print('svm_only',org['svm_only'])
print('pca_only',org['pca_only'])
print(f'{final_n_features_target - both_len} / {final_n_features_target}\n')

plt.show()

def test_with_subvector(subvector):
    new_x_train_svm = x_train[:,subvector]
    new_x_test_svm = x_test[:,subvector]
    new_svm_svm = svm.SVC(kernel='linear',C=1,decision_function_shape='ovo').fit(new_x_train_svm,y_train)
    return new_svm_svm.score(new_x_test_svm,y_test)

print('original svm score:',orig_svm_score)
print('svm selected subvector score:',test_with_subvector(svm_indices_order))
print('pca selected subvector score:',test_with_subvector(pca_indices_order))

rand_indices = list(range(len(x_train[0])))
np.random.shuffle(rand_indices)
length = len(svm_indices_order)
for i in range(5):
    start = i * length
    print('random subvector score:',test_with_subvector(rand_indices[start:start + length]))
# SOME OTHER SVM DATA THAT MIGHT BE INTERESTING
# print('w = ',linearsvm.coef_)
# print('b = ',linearsvm.intercept_)
# print('Indices of support vectors = ', linearsvm.support_)
# print('Support vectors = ', linearsvm.support_vectors_)
# print('Number of support vectors for each class = ', linearsvm.n_support_)
# print('Coefficients of the support vector in the decision function = ', np.abs(linearsvm.dual_coef_),linearsvm.dual_coef_.shape)