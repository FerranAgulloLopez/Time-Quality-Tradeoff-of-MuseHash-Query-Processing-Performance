import os
import random
from urllib.request import urlopen, urlretrieve

import h5py
import numpy
import numpy as np


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print("downloading %s -> %s..." % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", "%s.hdf5" % dataset)


def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = "http://ann-benchmarks.com/%s.hdf5" % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](hdf5_fn)
        else:
            if os.path.isfile(which):
                hdf5_fn = which
    hdf5_f = h5py.File(hdf5_fn, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_f.attrs["dimension"]) if "dimension" in hdf5_f.attrs else len(hdf5_f["train"][0])

    return hdf5_f, dimension


# Everything below this line is related to creating datasets
# You probably never need to do this at home,
# just rely on the prepared datasets at http://ann-benchmarks.com


def write_output(train, test, fn, distance, point_type="float", count=100):
    from ann_benchmarks.algorithms.bruteforce import bruteforce

    f = h5py.File(fn, "w")
    f.attrs["type"] = "dense"
    f.attrs["distance"] = distance
    f.attrs["dimension"] = len(train[0])
    f.attrs["point_type"] = point_type
    print("train size: %9d * %4d" % train.shape)
    print("test size:  %9d * %4d" % test.shape)
    f.create_dataset("train", (len(train), len(train[0])), dtype=train.dtype)[:] = train
    f.create_dataset("test", (len(test), len(test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset("neighbors", (len(test), count), dtype="i")
    distances = f.create_dataset("distances", (len(test), count), dtype="f")
    bf = bruteforce(distance)

    bf.fit(0, train)
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print("%d/%d..." % (i, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()


"""
param: train and test are arrays of arrays of indices.
"""


def write_sparse_output(train, test, fn, distance, dimension, count=100):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS

    f = h5py.File(fn, "w")
    f.attrs["type"] = "sparse"
    f.attrs["distance"] = distance
    f.attrs["dimension"] = dimension
    f.attrs["point_type"] = "bit"
    print("train size: %9d * %4d" % (train.shape[0], dimension))
    print("test size:  %9d * %4d" % (test.shape[0], dimension))

    # We ensure the sets are sorted
    train = numpy.array(list(map(sorted, train)))
    test = numpy.array(list(map(sorted, test)))

    flat_train = numpy.hstack(train.flatten())
    flat_test = numpy.hstack(test.flatten())

    f.create_dataset("train", (len(flat_train),), dtype=flat_train.dtype)[:] = flat_train
    f.create_dataset("test", (len(flat_test),), dtype=flat_test.dtype)[:] = flat_test
    neighbors = f.create_dataset("neighbors", (len(test), count), dtype="i")
    distances = f.create_dataset("distances", (len(test), count), dtype="f")

    f.create_dataset("size_test", (len(test),), dtype="i")[:] = list(map(len, test))
    f.create_dataset("size_train", (len(train),), dtype="i")[:] = list(map(len, train))

    bf = BruteForceBLAS(distance, precision=train.dtype)
    bf.fit(train)
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print("%d/%d..." % (i, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()


def train_test_split(X, test_size=10000, dimension=None):
    import sklearn.model_selection

    if dimension is None:
        dimension = X.shape[1]
    print("Splitting %d*%d into train/test" % (X.shape[0], dimension))
    return sklearn.model_selection.train_test_split(X, test_size=test_size, random_state=1)


def glove(out_fn, d):
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X_train, X_test = train_test_split(X)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


def _load_texmex_vectors(f, n, k):
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn):
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", "sift.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def gist(out_fn):
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def _load_mnist_vectors(fn):
    import gzip
    import struct

    print("parsing vectors in %s..." % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn):
    download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "mnist-train.gz")  # noqa
    download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "mnist-test.gz")  # noqa
    train = _load_mnist_vectors("mnist-train.gz")
    test = _load_mnist_vectors("mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn):
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-train.gz",
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-test.gz",
    )
    train = _load_mnist_vectors("fashion-mnist-train.gz")
    test = _load_mnist_vectors("fashion-mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn):
    yadisk_key = "https://yadi.sk/d/11eDCm7Dsn9GA"
    response = urlopen(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        + yadisk_key
        + "&path=/deep10M.fvecs"
    )
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(",")[0][9:-1]
    filename = os.path.join("data", "deep-image.fvecs")
    download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = numpy.fromfile(filename, dtype=numpy.float32)
    dim = fv.view(numpy.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv)
    write_output(X_train, X_test, out_fn, "angular")


def transform_bag_of_words(filename, n_dimensions, out_fn):
    import gzip

    from scipy.sparse import lil_matrix
    from sklearn import random_projection
    from sklearn.feature_extraction.text import TfidfTransformer

    with gzip.open(filename, "rb") as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(n_components=n_dimensions).fit_transform(B)
        X_train, X_test = train_test_split(C)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


def nytimes(out_fn, n_dimensions):
    fn = "nytimes_%s.txt.gz" % n_dimensions
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz", fn
    )  # noqa
    transform_bag_of_words(fn, n_dimensions, out_fn)


def random_float(out_fn, n_dims, n_samples, centers, distance):
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1)
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def random_bitstring(out_fn, n_dims, n_samples, n_queries):
    import sklearn.datasets

    Y, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=n_queries, random_state=1)
    X = numpy.zeros((n_samples, n_dims), dtype=numpy.bool_)
    for i, vec in enumerate(Y):
        X[i] = numpy.array([v > 0 for v in vec], dtype=numpy.bool_)

    X_train, X_test = train_test_split(X, test_size=n_queries)
    write_output(X_train, X_test, out_fn, "hamming", "bit")


def word2bits(out_fn, path, fn):
    import tarfile

    local_fn = fn + ".tar.gz"
    url = "http://web.stanford.edu/~maxlam/word_vectors/compressed/%s/%s.tar.gz" % (path, fn)  # noqa
    download(url, local_fn)
    print("parsing vectors in %s..." % local_fn)
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        n_words, k = [int(z) for z in next(f).strip().split()]
        X = numpy.zeros((n_words, k), dtype=numpy.bool_)
        for i in range(n_words):
            X[i] = numpy.array([float(z) > 0 for z in next(f).strip().split()[1:]], dtype=numpy.bool_)

        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def sift_hamming(out_fn, fn):
    import tarfile

    local_fn = fn + ".tar.gz"
    url = "http://sss.projects.itu.dk/ann-benchmarks/datasets/%s.tar.gz" % fn
    download(url, local_fn)
    print("parsing vectors in %s..." % local_fn)
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        lines = f.readlines()
        X = numpy.zeros((len(lines), 256), dtype=numpy.bool_)
        for i, line in enumerate(lines):
            X[i] = numpy.array([int(x) > 0 for x in line.decode().strip()], dtype=numpy.bool_)
        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def kosarak(out_fn):
    import gzip

    local_fn = "kosarak.dat.gz"
    # only consider sets with at least min_elements many elements
    min_elements = 20
    url = "http://fimi.uantwerpen.be/data/%s" % local_fn
    download(url, local_fn)

    X = []
    dimension = 0
    with gzip.open("kosarak.dat.gz", "r") as f:
        content = f.readlines()
        # preprocess data to find sets with more than 20 elements
        # keep track of used ids for reenumeration
        for line in content:
            if len(line.split()) >= min_elements:
                X.append(list(map(int, line.split())))
                dimension = max(dimension, max(X[-1]) + 1)

    X_train, X_test = train_test_split(numpy.array(X), test_size=500, dimension=dimension)
    write_sparse_output(X_train, X_test, out_fn, "jaccard", dimension)


def random_jaccard(out_fn, n=10000, size=50, universe=80):
    random.seed(1)
    l = list(range(universe))
    X = []
    for i in range(n):
        X.append(random.sample(l, size))

    X_train, X_test = train_test_split(numpy.array(X), test_size=100, dimension=universe)
    write_sparse_output(X_train, X_test, out_fn, "jaccard", universe)


def lastfm(out_fn, n_dimensions, test_size=50000):
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)
    import implicit
    from implicit.approximate_als import augment_inner_product_matrix
    from implicit.datasets.lastfm import get_lastfm

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = numpy.append(model.user_factors, numpy.zeros((model.user_factors.shape[0], 1)), axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, user_factors, out_fn, "angular")


def movielens(fn, ratings_file, out_fn, separator="::", ignore_header=False):
    import zipfile

    url = "http://files.grouplens.org/datasets/movielens/%s" % fn

    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        file = z.open(ratings_file)
        if ignore_header:
            file.readline()

        print("preparing %s" % out_fn)

        users = {}
        X = []
        dimension = 0
        for line in file:
            el = line.decode("UTF-8").split(separator)

            userId = el[0]
            itemId = int(el[1])
            rating = float(el[2])

            if rating < 3:  # We only keep ratings >= 3
                continue

            if userId not in users:
                users[userId] = len(users)
                X.append([])

            X[users[userId]].append(itemId)
            dimension = max(dimension, itemId + 1)

        X_train, X_test = train_test_split(numpy.array(X), test_size=500, dimension=dimension)
        write_sparse_output(X_train, X_test, out_fn, "jaccard", dimension)


def movielens1m(out_fn):
    movielens("ml-1m.zip", "ml-1m/ratings.dat", out_fn)


def movielens10m(out_fn):
    movielens("ml-10m.zip", "ml-10M100K/ratings.dat", out_fn)


def movielens20m(out_fn):
    movielens("ml-20m.zip", "ml-20m/ratings.csv", out_fn, ",", True)


def vgg16_features(out_fn, dataset_path):
    vgg16_features_path = f'{dataset_path}/vgg16_features/'

    # load split info
    train_split = np.genfromtxt(os.path.join(dataset_path, 'train.txt'), dtype=int)
    test_split = np.genfromtxt(os.path.join(dataset_path, 'test.txt'), dtype=int)

    # load vgg16 data
    vgg16_train_data = np.zeros((train_split.shape[0], 4096))
    for index in range(train_split.shape[0]):
        print(f'Pending files to load: {train_split.shape[0] - index}')
        vgg16_train_data[index] = np.genfromtxt(os.path.join(vgg16_features_path, f'feature_{train_split[index]}.txt'))
    vgg16_test_data = np.zeros((test_split.shape[0], 4096))
    for index in range(test_split.shape[0]):
        print(f'Pending files to load: {test_split.shape[0] - index}')
        vgg16_test_data[index] = np.genfromtxt(os.path.join(vgg16_features_path, f'feature_{test_split[index]}.txt'))

    # create final dataset
    write_output(vgg16_train_data, vgg16_test_data, out_fn, 'euclidean')


def muse_hash(out_fn, dataset_path, modalities, bits, metric):

    def populate_hash_codes(path, split, data, modalities):
        for index in range(split.shape[0]):
            print(f'Pending files to load: {split.shape[0] - index}')
            if len(modalities) == 1:
                data[index] = np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
            elif len(modalities) == 2:
                data[index] = np.bitwise_xor(
                    np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                    np.genfromtxt(os.path.join(path, modalities[1], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                )
            elif len(modalities) == 3:
                data[index] = np.bitwise_and(
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[1], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    ),
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[2], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    ),
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[1], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[2], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    )
                )
            elif len(modalities) == 4:
                aux1 = np.bitwise_and(
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[1], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    ),
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[2], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    )
                )
                aux2 = np.bitwise_and(
                    aux1,
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[0], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[3], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    ),
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[1], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[2], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    )
                )
                data[index] = np.bitwise_and(
                    aux2,
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[1], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[3], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    ),
                    np.bitwise_xor(
                        np.genfromtxt(os.path.join(path, modalities[2], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int),
                        np.genfromtxt(os.path.join(path, modalities[3], f'{bits}bit', f'bin_feature_{split[index]}.txt'), dtype=int)
                    )
                )
            else:
                raise ValueError('Unsupported number of modality')
        return data

    # load split info
    train_split = np.genfromtxt(os.path.join(dataset_path, 'train.txt'), dtype=int)
    test_split = np.genfromtxt(os.path.join(dataset_path, 'test.txt'), dtype=int)

    # load train hash codes data
    hash_codes_train_data = np.zeros((train_split.shape[0], bits))
    hash_codes_train_data = populate_hash_codes(f'{dataset_path}/hash_codes/', train_split, hash_codes_train_data, modalities)

    # load test hash codes data
    hash_codes_test_data = np.zeros((test_split.shape[0], bits))
    hash_codes_test_data = populate_hash_codes(f'{dataset_path}/hash_codes/', test_split, hash_codes_test_data, modalities)

    # create final dataset
    if metric == 'hamming':
        write_output(hash_codes_train_data, hash_codes_test_data, out_fn, metric, point_type='bit')
    else:
        write_output(hash_codes_train_data, hash_codes_test_data, out_fn, metric)


def fake_dataset(out_fn, train_size, bits):
    hash_codes_train_data = np.random.rand(train_size, bits)
    hash_codes_test_data = np.random.rand(450, bits)
    write_output(hash_codes_train_data, hash_codes_test_data, out_fn, 'euclidean')


DATASETS = {
    "deep-image-96-angular": deep_image,
    "fashion-mnist-784-euclidean": fashion_mnist,
    "gist-960-euclidean": gist,
    "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "mnist-784-euclidean": mnist,
    "random-xs-20-euclidean": lambda out_fn: random_float(out_fn, 20, 10000, 100, "euclidean"),
    "random-s-100-euclidean": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "euclidean"),
    "random-xs-20-angular": lambda out_fn: random_float(out_fn, 20, 10000, 100, "angular"),
    "random-s-100-angular": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "angular"),
    "random-xs-16-hamming": lambda out_fn: random_bitstring(out_fn, 16, 10000, 100),
    "random-s-128-hamming": lambda out_fn: random_bitstring(out_fn, 128, 50000, 1000),
    "random-l-256-hamming": lambda out_fn: random_bitstring(out_fn, 256, 100000, 1000),
    "random-s-jaccard": lambda out_fn: random_jaccard(out_fn, n=10000, size=20, universe=40),
    "random-l-jaccard": lambda out_fn: random_jaccard(out_fn, n=100000, size=70, universe=100),
    "sift-128-euclidean": sift,
    "nytimes-256-angular": lambda out_fn: nytimes(out_fn, 256),
    "nytimes-16-angular": lambda out_fn: nytimes(out_fn, 16),
    "word2bits-800-hamming": lambda out_fn: word2bits(out_fn, "400K", "w2b_bitlevel1_size800_vocab400K"),
    "lastfm-64-dot": lambda out_fn: lastfm(out_fn, 64),
    "sift-256-hamming": lambda out_fn: sift_hamming(out_fn, "sift.hamming.256"),
    "kosarak-jaccard": lambda out_fn: kosarak(out_fn),
    "movielens1m-jaccard": movielens1m,
    "movielens10m-jaccard": movielens10m,
    "movielens20m-jaccard": movielens20m,
    'vgg16-features-au_air': lambda out_fn: vgg16_features(out_fn, './data/au_air_dataset'),
    'vgg16-features-lcs': lambda out_fn: vgg16_features(out_fn, './data/lcs_dataset'),
    'muse-hash-visual-temporal-spatial-16-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 16, 'euclidean'),
    'muse-hash-visual-temporal-spatial-32-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 32, 'euclidean'),
    'muse-hash-visual-temporal-spatial-64-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 64, 'euclidean'),
    'muse-hash-visual-temporal-spatial-128-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 128, 'euclidean'),
    'muse-hash-visual-temporal-spatial-256-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 256, 'euclidean'),
    'muse-hash-visual-temporal-spatial-512-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 512, 'euclidean'),
    'muse-hash-visual-temporal-spatial-1024-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 1024, 'euclidean'),
    'muse-hash-visual-temporal-spatial-2048-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal', 'spatial'], 2048, 'euclidean'),
    'muse-hash-visual-temporal-16-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 16, 'euclidean'),
    'muse-hash-visual-temporal-32-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 32, 'euclidean'),
    'muse-hash-visual-temporal-64-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 64, 'euclidean'),
    'muse-hash-visual-temporal-128-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 128, 'euclidean'),
    'muse-hash-visual-temporal-256-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 256, 'euclidean'),
    'muse-hash-visual-temporal-512-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 512, 'euclidean'),
    'muse-hash-visual-temporal-1024-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 1024, 'euclidean'),
    'muse-hash-visual-temporal-2048-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual', 'temporal'], 2048, 'euclidean'),
    'muse-hash-visual-16-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 16, 'euclidean'),
    'muse-hash-visual-32-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 32, 'euclidean'),
    'muse-hash-visual-64-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 64, 'euclidean'),
    'muse-hash-visual-128-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 128, 'euclidean'),
    'muse-hash-visual-256-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 256, 'euclidean'),
    'muse-hash-visual-512-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 512, 'euclidean'),
    'muse-hash-visual-1024-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 1024, 'euclidean'),
    'muse-hash-visual-2048-euclidean-au_air': lambda out_fn: muse_hash(out_fn, './data/au_air_dataset', ['visual'], 2048, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-16-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 16, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-32-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 32, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-64-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 64, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-128-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 128, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-256-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 256, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-512-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 512, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-1024-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 1024, 'euclidean'),
    'muse-hash-visual-temporal-spatial-textual-2048-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial', 'textual'], 2048, 'euclidean'),
    'muse-hash-visual-temporal-spatial-16-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 16, 'euclidean'),
    'muse-hash-visual-temporal-spatial-32-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 32, 'euclidean'),
    'muse-hash-visual-temporal-spatial-64-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 64, 'euclidean'),
    'muse-hash-visual-temporal-spatial-128-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 128, 'euclidean'),
    'muse-hash-visual-temporal-spatial-256-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 256, 'euclidean'),
    'muse-hash-visual-temporal-spatial-512-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 512, 'euclidean'),
    'muse-hash-visual-temporal-spatial-1024-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 1024, 'euclidean'),
    'muse-hash-visual-temporal-spatial-2048-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal', 'spatial'], 2048, 'euclidean'),
    'muse-hash-visual-temporal-16-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 16, 'euclidean'),
    'muse-hash-visual-temporal-32-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 32, 'euclidean'),
    'muse-hash-visual-temporal-64-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 64, 'euclidean'),
    'muse-hash-visual-temporal-128-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 128, 'euclidean'),
    'muse-hash-visual-temporal-256-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 256, 'euclidean'),
    'muse-hash-visual-temporal-512-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 512, 'euclidean'),
    'muse-hash-visual-temporal-1024-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 1024, 'euclidean'),
    'muse-hash-visual-temporal-2048-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual', 'temporal'], 2048, 'euclidean'),
    'muse-hash-visual-16-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 16, 'euclidean'),
    'muse-hash-visual-32-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 32, 'euclidean'),
    'muse-hash-visual-64-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 64, 'euclidean'),
    'muse-hash-visual-128-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 128, 'euclidean'),
    'muse-hash-visual-256-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 256, 'euclidean'),
    'muse-hash-visual-512-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 512, 'euclidean'),
    'muse-hash-visual-1024-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 1024, 'euclidean'),
    'muse-hash-visual-2048-euclidean-lcs': lambda out_fn: muse_hash(out_fn, './data/lcs_dataset', ['visual'], 2048, 'euclidean'),
    'fake-small-32': lambda out_fn: fake_dataset(out_fn, 28000, 32),
    'fake-small-128': lambda out_fn: fake_dataset(out_fn, 28000, 128),
    'fake-small-512': lambda out_fn: fake_dataset(out_fn, 28000, 512),
    'fake-small-2048': lambda out_fn: fake_dataset(out_fn, 28000, 2048),
    'fake-medium-32': lambda out_fn: fake_dataset(out_fn, 112000, 32),
    'fake-medium-128': lambda out_fn: fake_dataset(out_fn, 112000, 128),
    'fake-medium-512': lambda out_fn: fake_dataset(out_fn, 112000, 512),
    'fake-medium-2048': lambda out_fn: fake_dataset(out_fn, 112000, 2048),
    'fake-large-1': lambda out_fn: fake_dataset(out_fn, 448000, 1),
    'fake-large-2': lambda out_fn: fake_dataset(out_fn, 448000, 2),
    'fake-large-4': lambda out_fn: fake_dataset(out_fn, 448000, 4),
    'fake-large-8': lambda out_fn: fake_dataset(out_fn, 448000, 8),
    'fake-large-16': lambda out_fn: fake_dataset(out_fn, 448000, 16),
    'fake-large-32': lambda out_fn: fake_dataset(out_fn, 448000, 32),
    'fake-large-128': lambda out_fn: fake_dataset(out_fn, 448000, 128),
    'fake-large-512': lambda out_fn: fake_dataset(out_fn, 448000, 512),
    'fake-large-2048': lambda out_fn: fake_dataset(out_fn, 448000, 2048)
}
