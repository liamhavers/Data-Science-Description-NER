"""Hand-curated data science skills/tools taxonomy used to seed the spaCy EntityRuler.

Each entry is (canonical_name, category, [surface forms to match, lowercased]).
`category` becomes the spaCy entity label; `canonical_name` is looked up via
`ent.ent_id_` so variant spellings (e.g. "scikit-learn" / "sklearn") collapse
to one skill during aggregation.
"""

TAXONOMY = [
    # LANGUAGE
    ("Python", "LANGUAGE", ["python"]),
    ("SQL", "LANGUAGE", ["sql"]),
    ("Java", "LANGUAGE", ["java"]),
    ("Scala", "LANGUAGE", ["scala"]),
    ("C++", "LANGUAGE", ["c++"]),
    ("Julia", "LANGUAGE", ["julia"]),

    # CLOUD
    ("AWS", "CLOUD", ["aws", "amazon web services"]),
    ("Azure", "CLOUD", ["azure", "microsoft azure"]),
    ("GCP", "CLOUD", ["gcp", "google cloud platform", "google cloud"]),

    # DATABASE
    ("PostgreSQL", "DATABASE", ["postgresql", "postgres"]),
    ("MySQL", "DATABASE", ["mysql"]),
    ("MongoDB", "DATABASE", ["mongodb", "mongo"]),
    ("Snowflake", "DATABASE", ["snowflake"]),
    ("Redshift", "DATABASE", ["redshift", "amazon redshift"]),
    ("BigQuery", "DATABASE", ["bigquery", "google bigquery"]),
    ("NoSQL", "DATABASE", ["nosql"]),

    # LIBRARY
    ("TensorFlow", "LIBRARY", ["tensorflow"]),
    ("PyTorch", "LIBRARY", ["pytorch"]),
    ("scikit-learn", "LIBRARY", ["scikit-learn", "scikit learn", "sklearn"]),
    ("Pandas", "LIBRARY", ["pandas"]),
    ("NumPy", "LIBRARY", ["numpy"]),
    ("Keras", "LIBRARY", ["keras"]),
    ("XGBoost", "LIBRARY", ["xgboost"]),

    # TOOL
    ("Docker", "TOOL", ["docker"]),
    ("Kubernetes", "TOOL", ["kubernetes", "k8s"]),
    ("Airflow", "TOOL", ["airflow", "apache airflow"]),
    ("Spark", "TOOL", ["spark", "apache spark", "pyspark"]),
    ("Tableau", "TOOL", ["tableau"]),
    ("Power BI", "TOOL", ["power bi", "powerbi"]),
    ("Git", "TOOL", ["git"]),
    ("Excel", "TOOL", ["excel"]),
    ("Hadoop", "TOOL", ["hadoop"]),
    ("Jupyter", "TOOL", ["jupyter"]),

    # METHOD
    ("A/B Testing", "METHOD", ["a/b testing", "ab testing", "a/b test", "ab test"]),
    ("Machine Learning", "METHOD", ["machine learning"]),
    ("Deep Learning", "METHOD", ["deep learning"]),
    ("NLP", "METHOD", ["nlp", "natural language processing"]),
    ("Computer Vision", "METHOD", ["computer vision"]),
    ("Statistics", "METHOD", ["statistics", "statistical analysis"]),
    ("Data Visualization", "METHOD", ["data visualization", "data viz"]),
    ("ETL", "METHOD", ["etl"]),
    ("Time Series", "METHOD", ["time series"]),
]
