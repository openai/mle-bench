import numpy as np

fast_competitions = [
    "leaf-classification",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "playground-series-s3e18",
    "spaceship-titanic",
    "us-patent-phrase-to-phrase-matching",
]

sample_submission_scores = {
    "3d-object-detection-for-autonomous-vehicles": 0.0,
    "aerial-cactus-identification": 0.5,
    "AI4Code": 0.39187,
    "alaska2-image-steganalysis": 0.58571,
    "aptos2019-blindness-detection": 0.0,
    "billion-word-imputation": 5.56118,
    "bms-molecular-translation": 109.59778,
    "cassava-leaf-disease-classification": 0.11584,
    "cdiscount-image-classification-challenge": 0.00885,
    "chaii-hindi-and-tamil-question-answering": 0.0,
    "champs-scalar-coupling": 1.99777,
    "denoising-dirty-documents": 0.28616,
    "detecting-insults-in-social-commentary": 0.5,
    "dog-breed-identification": 4.78749,
    "dogs-vs-cats-redux-kernels-edition": 0.69315,
    "facebook-recruiting-iii-keyword-extraction": 0.07645,
    "freesound-audio-tagging-2019": 0.02197,
    "google-quest-challenge": -0.01016,
    "google-research-identify-contrails-reduce-global-warming": 6e-05,
    "h-and-m-personalized-fashion-recommendations": 0.0,
    "herbarium-2020-fgvc7": 0.0,
    "herbarium-2021-fgvc8": 0.0,
    "herbarium-2022-fgvc9": 0.0,
    "histopathologic-cancer-detection": 0.5,
    "hms-harmful-brain-activity-classification": 1.40995,
    "hotel-id-2021-fgvc8": 0.00232,
    "hubmap-kidney-segmentation": 0.0,
    "icecube-neutrinos-in-deep-ice": 1.55139,
    "imet-2020-fgvc7": 0.00006,
    "inaturalist-2019-fgvc6": 0.9996,
    "invasive-species-monitoring": 0.5,
    "iwildcam-2020-fgvc7": 0.00117,
    "iwildcam-2019-fgvc6": 0.02315,
    "jigsaw-toxic-comment-classification-challenge": 0.5,
    "jigsaw-unintended-bias-in-toxicity-classification": 0.5,
    "kuzushiji-recognition": 0.0,
    "leaf-classification": 4.59512,
    "learning-agency-lab-automated-essay-scoring-2": 0.01323,
    "lmsys-chatbot-arena": 1.09861,
    "ml2021spring-hw2": 0.05167,
    "mlsp-2013-birds": 0.5,
    "movie-review-sentiment-analysis-kernels-only": 0.51314,
    "multi-modal-gesture-recognition": 0.91966,
    "new-york-city-taxi-fare-prediction": 10.02927,
    "nfl-player-contact-detection": 0.0,
    "nomad2018-predict-transparent-conductors": 0.21174,
    "osic-pulmonary-fibrosis-progression": -14.9683,
    "paddy-disease-classification": 0.0,
    "petfinder-pawpularity-score": 37.37587,
    "plant-pathology-2020-fgvc7": 0.5,
    "plant-pathology-2021-fgvc8": 0.24507,
    "plant-seedlings-classification": 0.07057,
    "playground-series-s3e18": 0.5,
    "predict-volcanic-eruptions-ingv-oe": 24402200,
    "random-acts-of-pizza": 0.5,
    "ranzcr-clip-catheter-line-classification": 0.5,
    "rsna-2022-cervical-spine-fracture-detection": 0.69315,
    "rsna-breast-cancer-detection": 0.02212,
    "rsna-miccai-brain-tumor-radiogenomic-classification": 0.5,
    "seti-breakthrough-listen": 0.5,
    "siim-covid19-detection": 0.24492,
    "siim-isic-melanoma-classification": 0.5,
    "smartphone-decimeter-2022": 3122773.6564,
    "spaceship-titanic": 0.50345,
    "spooky-author-identification": 1.08468,
    "stanford-covid-vaccine": 0.63824,
    "statoil-iceberg-classifier-challenge": 0.69315,
    "tabular-playground-series-may-2022": 0.5,
    "tabular-playground-series-dec-2021": 0.56458,
    "tensorflow-speech-recognition-challenge": 0.0,
    "tensorflow2-question-answering": 0.57117,
    "text-normalization-challenge-english-language": 0.93326,
    "text-normalization-challenge-russian-language": 0.87501,
    "tgs-salt-identification-challenge": 0.0,
    "the-icml-2013-whale-challenge-right-whale-redux": 0.5,
    "tweet-sentiment-extraction": 0.0,
    "us-patent-phrase-to-phrase-matching": np.nan,
    "uw-madison-gi-tract-image-segmentation": 0.24608,
    "ventilator-pressure-prediction": 17.65486,
    "vesuvius-challenge-ink-detection": 0.0,
    "vinbigdata-chest-xray-abnormalities-detection": 0.0475,
    "whale-categorization-playground": 0.11549,
}