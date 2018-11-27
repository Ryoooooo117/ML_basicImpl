import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')

        string = ''
        for idx_cls in range(node.num_cls):
            string += str(node.labels.count(idx_cls)) + ' '
        print(indent + ' num of sample / cls: ' + string)

        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the index of the feature to be split

        self.feature_uniq_split = None # the possible unique values of the feature to be split


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array,
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of
                      corresponding training samples
                      e.g.
                                  ○ ○ ○ ○
                                  ● ● ● ●
                                ┏━━━━┻━━━━┓
                               ○ ○       ○ ○
                               ● ● ● ●

                      branches = [[2,2], [4,0]]
            '''
            ########################################################
            # TODO: compute the conditional entropy
            ########################################################

            branches = np.asarray(branches)
            branchTotalNum = np.sum(branches, axis=0)
            totalNum = np.sum(branches)

            entropies = []
            for i in range(len(branches[0])):
                branchCol = branches[:, i]
                entropy = []
                for j in range(len(branchCol)):
                    if branchCol[j]/branchTotalNum[i] == 0:
                        entropy.append(0.0)
                    else:
                        entropy.append(-branchCol[j] / branchTotalNum[i] * np.log(branchCol[j] / branchTotalNum[i]))
                entropies.append(np.sum(entropy))
            conditionalEntropy = entropies * (branchTotalNum / totalNum)
            return np.sum(conditionalEntropy)

        best_dim = None
        minEntropy = 9223372036854775807
        print('self feature', self.features, 'self label', self.labels)
        for idx_dim in range(len(self.features[0])):
        ############################################################
        # TODO: compare each split using conditional entropy
        #       find the best split
        ############################################################

            fts = np.asarray(self.features)
            C = len(np.unique(fts[:, idx_dim]))
            B = len(np.unique(self.labels))
            branches = np.zeros([C,B])
            print(' idx_dim', idx_dim, ' np.unique(fts[:, idx_dim])', np.unique(fts[:, idx_dim]))
            if C == 1: continue
            for i in range(len(np.unique(fts[:, idx_dim]))):
                branch = np.unique(fts[:, idx_dim])[i]
                for j in range(len(np.unique(self.labels))):
                    label = np.unique(self.labels)[j]
                    indices = [i for i, x in enumerate(fts[:, idx_dim].tolist()) if x == branch]
                    for idx in indices:
                        if self.labels[idx] == label:
                            branches[i][j] += 1

            ce  = conditional_entropy(branches)

            branches = np.array(branches).astype(int).tolist()
            if ce < minEntropy:
                minEntropy = ce
                best_dim = idx_dim
                self.dim_split = best_dim
                self.feature_uniq_split = np.unique(fts[:, idx_dim]).tolist()
            print('        ce: ', ce, ' idx_dim', idx_dim, ' feat_unique_values', self.feature_uniq_split, ' branches', branches)

        # print('minEntropy ', minEntropy, ' best_dim ',best_dim)
        ############################################################
        # TODO: split the node, add child nodes
        ############################################################

        # print('self.num_cls',self.num_cls)
        # print('self.cls_max', self.cls_max)
        # print('len(self.features[0])', len(self.features[0]))

        if self.feature_uniq_split is None:
            self.splittable = False
            return

        fts = np.asarray(self.features)
        classes = fts[:, self.dim_split]
        print('classes ',classes)
        for split in self.feature_uniq_split:
            childFeature = []
            childLabel = []
            for i in range(len(classes)):                   # 4 * 1
                c = classes[i]
                if c == split:
                    childFeature.append(self.features[i])   # 2*1
                    childLabel.append(self.labels[i])
            print('    self.feature_uniq_split ',self.feature_uniq_split,'  childFeature ',childFeature)
            node = TreeNode(childFeature,childLabel,np.max(childLabel)+1)
            self.children.append(node)

        # split the child nodes
        if len(self.children) == 0:
            return

        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            # feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



