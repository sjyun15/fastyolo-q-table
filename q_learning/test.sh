rm -rf annotations_cache
rm output/*
python main.py test
python etc/make_result.py test
python voc_eval.py /SSD/scam/q_learning/output/comp4_det_test_
