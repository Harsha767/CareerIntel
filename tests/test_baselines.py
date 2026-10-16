from careerintel.models.baselines import TfidfRanker


def test_tfidf_ranks_relevant_first():
    ids = [1, 2, 3]
    texts = [
        "python pandas machine learning data scientist",
        "java spring boot backend developer",
        "graphic design illustrator photoshop",
    ]
    r = TfidfRanker().fit(ids, texts)
    assert r.rank("python machine learning", top_k=1)[0][0] == 1
