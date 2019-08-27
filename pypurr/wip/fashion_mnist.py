import time

import numpy as np
from skimage.feature import local_binary_pattern, hog, daisy, ORB, CENSURE, BRIEF, corner_fast, corner_peaks
from skimage.measure import compare_ssim
from skimage.transform import integral_image
from skimage.util import view_as_windows
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import skimage.util as util


from pypurr.train.helpers.model import LambdaRow, flatten

class PatchPCA(TransformerMixin, BaseEstimator):
    """"""

    def __init__(self, pca: BaseEstimator):
        """Constructor for ProbFromClf"""
        self.pca = pca

    def fit(self, X, y=None):
        extractor = PatchExtractor(patch_size=(14, 14), max_patches=4)

        patch_X = extractor.transform(X)

        n_patches, _, _, = patch_X.shape

        self.pca.fit(patch_X.reshape(n_patches, -1))

        return self

    def _extract_patch(self, arr):
        win_arr = view_as_windows(arr, window_shape=14, step=14)

        n_patches_i, n_patches_j, _, _ = win_arr.shape

        return self.pca.transform(win_arr.reshape(n_patches_i*n_patches_j, -1)).flatten()

    def transform(self, X):
        return np.array([self._extract_patch(x) for x in X])

    def get_params(self, deep=True):
        return dict(pca=self.pca)

    def set_params(self, **params):
        return self.pca.set_params(**params)


class FilterLabel():
    """"""

    def __init__(self, transformer, selected_labels=None,  max_labels=None):
        """Constructor for FilterLabel"""
        self.labels = selected_labels
        self.transformer = transformer
        self.max_labels = max_labels

    def fit(self, X, y=None):

        labels = np.unique(y)

        selected_labels = self.labels if self.labels else np.random.choice(labels, size=self.max_labels)

        self.transformer.fit(X[np.isin(y,selected_labels)])

        return self

    def transform(self, X):
        return self.transformer.transform(X)


class ExemplarTransform():
    """"""

    def __init__(self):
        """Constructor for FilterLabel"""
        self.exemplars = None

    def fit(self, X, y=None):
        idx = np.random.choice(np.arange(X.shape[0]), size=256)

        self.examplars = X[idx, ::]

        return self

    def transform(self, X):
        return np.array([ [ compare_ssim(x, ex) for ex in self.examplars] for x in X])

class BagOfWords():
    """"""
    
    def __init__(self, word_extractor):
        """Constructor for BagOfWords"""
        self.word_extractor = word_extractor
        self.km = None

    def fit(self, X, y=None):

        idx = np.random.choice(np.arange(X.shape[0]), size=1000)

        X = X[idx,::]

        X_ = np.vstack([ self.word_extractor(x) for x in X])

        self.km = KMeans(n_clusters=256, n_init=1, max_iter=128)

        self.km.fit(X_)

        return self

    def transform(self, X):
        return np.vstack([  np.bincount(self.km.predict(self.word_extractor(x)), minlength=257) for x in X] )
        

def train(pipe: Pipeline):
    print("Loading data")
    data = np.load("/home/matthieu/Workspace/data/fashion_mnist.npz")
    xtrain, xtest, ytrain, ytest = data["xtrain"], data["xtest"], data["ytrain"], data["ytest"]

    for fold, (x, y) in [("train", (xtrain, ytrain)),("test", (xtest, ytest))]:
        print("{0} : X : {1} Y : {2}".format(fold, x.shape, y.shape))


    t0 = time.time()
    pipe.fit(xtrain, ytrain.ravel())
    print("Training time : {:05f} seconds".format(time.time()-t0))


    for tag, x, y in [
        ("train", xtrain, ytrain), ("test", xtest, ytest)]:

        print("--- Fold {} ----".format(tag))

        for val_metric in [confusion_matrix, accuracy_score]:

            print("{0}  --> \n {1}".format(val_metric.__name__, val_metric(y, pipe.predict(x))))

if __name__ == '__main__':

    print("-------- Integral features ------------")

    integral_pipeline = Pipeline([
        ("integral_image", LambdaRow(integral_image)),
        ("flattener", LambdaRow(flatten)),
        ("scaler", StandardScaler()),
        ("extra_trees",
         ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])
    # train(integral_pipeline)

    print("-------- Raw Pixel features ------------")

    pix_pipeline = Pipeline([
        ("flattener", LambdaRow(flatten)),
        ("scaler", StandardScaler()),
        ("extra_trees",
         ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])
    # train(pix_pipeline)

    print("-------- Random Proj features ------------")

    rnd_pipeline = Pipeline([
        ("flattener", LambdaRow(flatten)),
        ("random_proj", RBFSampler(n_components=250)),
        ("scaler", StandardScaler()),
        ("extra_trees",
         ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])
    # train(rnd_pipeline)

    print("-------- LBP features ------------")

    radius = 3

    lbp_pipeline = Pipeline([
        ("integral_image", LambdaRow(lambda x: local_binary_pattern(x, 8*radius,radius))),
        ("flattener", LambdaRow(flatten)),
        ("extra_trees",
         ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])
    # train(lbp_pipeline)

    print("-------- Patch PCA features ------------")

    patch_pca_pipeline = Pipeline([
        ("patch_pca", PatchPCA(pca=PCA(n_components=25, whiten=True))),
        ("extra_trees",
         ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])
    # train(patch_pca_pipeline)

    # print("-------- DAISY ------------")
    #
    # daisy_pipeline = Pipeline([
    #     ("daisy", LambdaRow(lambda x: daisy(x, radius=7, histograms=4, orientations=4, step=7).flatten())),
    #     ("extra_trees",
    #      ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    # ])
    # train(daisy_pipeline)

    print("-------- HOG ------------")

    hog_pipeline = Pipeline([
        ("hog", LambdaRow(lambda x: hog(x))),
        ("extra_trees",
         ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])
    # train(hog_pipeline)

    print("-------- Voting Classifier --------------")

    pipeline = VotingClassifier(
        [
            ("pixels", pix_pipeline),
            ("integral", integral_pipeline),
            ("hog", hog_pipeline),
            ("lbp", lbp_pipeline),
            ("patch_pca", patch_pca_pipeline)

        ],
        voting="soft"

    )

    # train(pipeline)

    # data = np.load("/home/matthieu/Workspace/data/fashion_mnist.npz")
    #
    # xtrain, xtest, ytrain, ytest = data["xtrain"], data["xtest"], data["ytrain"], data["ytest"]

    # orb = BRIEF(patch_size=7)
    #
    # regular_keypoints = np.array([ (i,j) for i in range(0,28,7) for j in range(0,28,7)])
    #
    # orb.extract(xtrain[0, ::], keypoints=regular_keypoints)
    #
    # print(orb.descriptors)
    #
    # print(orb.descriptors.shape)

    # padded_im = util.pad(xtrain[0, ::], 16, mode="constant", constant_values=0)
    #
    # print(padded_im.shape)
    #
    # orb = ORB(n_keypoints=16, n_scales=4, fast_n=8, fast_threshold=0.1)
    #
    # orb.detect_and_extract(padded_im)
    #
    # print(orb.descriptors)
    #
    # print(orb.descriptors.shape)
    #
    # print(corner_peaks(corner_fast(xtrain[0,::], n=8, threshold=0.1), min_distance=1).shape)

    print("-------- Rotation forest proxy --------------")

    filter_pipeline = Pipeline([
        ("flattener", LambdaRow(flatten)),
        ("rotation_forest", BaggingClassifier(
            base_estimator=Pipeline([
                ("pca_label", FilterLabel(max_labels=3, transformer=PCA(n_components=25, whiten=True))),
                ("tree",
                 ExtraTreeClassifier(max_depth=None, criterion="entropy", max_features=16, splitter="random"))

            ]),
            n_estimators=50
        ))
    ])

    # train(filter_pipeline)

    # def describe(x):
    #     padded_im = util.pad(x, 16, mode="constant", constant_values=0)
    #
    #     orb = ORB(n_keypoints=16, downscale=1.2, n_scales=4, fast_n=8, fast_threshold=0.1)
    #
    #     orb.detect_and_extract(padded_im)
    #
    #     return orb.descriptors.astype(float)
    #
    # print("Loading data")
    # data = np.load("/home/matthieu/Workspace/data/fashion_mnist.npz")
    # xtrain, xtest, ytrain, ytest = data["xtrain"], data["xtest"], data["ytrain"], data["ytest"]
    #
    # bow = BagOfWords(word_extractor=describe)
    #
    # bow.fit(xtrain)
    #
    # print(bow.transform(xtrain).shape)

    print("-------- Exemplar forest --------------")

    exemplar_pipeline = Pipeline([
        ("flattener", LambdaRow(flatten)),
        ("exemplar_forest", BaggingClassifier(
            base_estimator=Pipeline([
                ("exemplar", ExemplarTransform()),
                ("tree",
                 ExtraTreeClassifier(max_depth=None, criterion="entropy", max_features=1, splitter="random"))

            ]),
            n_estimators=50
        ))
    ])

    train(exemplar_pipeline)

