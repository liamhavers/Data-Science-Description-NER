"""Hand-curated data science skills/tools taxonomy used to seed the spaCy EntityRuler.

Each entry is (canonical_name, category, [surface forms to match, lowercased]).
`category` becomes the spaCy entity label; `canonical_name` is looked up via
`ent.ent_id_` so variant spellings (e.g. "scikit-learn" / "sklearn") collapse
to one skill during aggregation.

Expanded from the v1 hand-picked list using frequency counts of the dataset's
own `job_skills.csv` reference column (see CLAUDE.md) -- generic/non-actionable
terms from that column (e.g. "Communication", "Problem Solving", "Data
Analysis") were deliberately excluded; only specific tools/languages/methods
were added.
"""

TAXONOMY = [
    # LANGUAGE
    ("Python", "LANGUAGE", ["python"]),
    ("SQL", "LANGUAGE", ["sql"]),
    ("Java", "LANGUAGE", ["java"]),
    ("Scala", "LANGUAGE", ["scala"]),
    ("C++", "LANGUAGE", ["c++"]),
    ("Julia", "LANGUAGE", ["julia"]),
    ("R", "LANGUAGE", ["r programming", "r language"]),  # bare "r" excluded: too ambiguous a token
    ("MATLAB", "LANGUAGE", ["matlab"]),

    # CLOUD
    ("AWS", "CLOUD", ["aws", "amazon web services"]),
    ("Azure", "CLOUD", ["azure", "microsoft azure"]),
    ("GCP", "CLOUD", ["gcp", "google cloud platform", "google cloud"]),
    ("Amazon S3", "CLOUD", ["s3", "amazon s3"]),
    ("Amazon EMR", "CLOUD", ["emr", "amazon emr", "aws emr"]),
    ("SageMaker", "CLOUD", ["sagemaker", "amazon sagemaker"]),
    ("Vertex AI", "CLOUD", ["vertex ai"]),

    # DATABASE
    ("PostgreSQL", "DATABASE", ["postgresql", "postgres"]),
    ("MySQL", "DATABASE", ["mysql"]),
    ("MongoDB", "DATABASE", ["mongodb", "mongo"]),
    ("Snowflake", "DATABASE", ["snowflake"]),
    ("Redshift", "DATABASE", ["redshift", "amazon redshift"]),
    ("BigQuery", "DATABASE", ["bigquery", "google bigquery"]),
    ("NoSQL", "DATABASE", ["nosql"]),
    ("Oracle", "DATABASE", ["oracle"]),
    ("SQL Server", "DATABASE", ["sql server", "microsoft sql server", "mssql"]),
    ("Cassandra", "DATABASE", ["cassandra"]),
    ("Elasticsearch", "DATABASE", ["elasticsearch"]),
    ("Redis", "DATABASE", ["redis"]),

    # LIBRARY
    ("TensorFlow", "LIBRARY", ["tensorflow"]),
    ("PyTorch", "LIBRARY", ["pytorch"]),
    ("scikit-learn", "LIBRARY", ["scikit-learn", "scikit learn", "sklearn"]),
    ("Pandas", "LIBRARY", ["pandas"]),
    ("NumPy", "LIBRARY", ["numpy"]),
    ("Keras", "LIBRARY", ["keras"]),
    ("XGBoost", "LIBRARY", ["xgboost"]),
    ("Matplotlib", "LIBRARY", ["matplotlib"]),
    ("Seaborn", "LIBRARY", ["seaborn"]),
    ("Plotly", "LIBRARY", ["plotly"]),
    ("NLTK", "LIBRARY", ["nltk"]),
    ("Hugging Face / Transformers", "LIBRARY", ["huggingface", "hugging face", "transformers"]),
    ("LightGBM", "LIBRARY", ["lightgbm"]),
    ("CatBoost", "LIBRARY", ["catboost"]),
    ("statsmodels", "LIBRARY", ["statsmodels"]),
    ("OpenCV", "LIBRARY", ["opencv"]),
    ("Gurobi", "LIBRARY", ["gurobi"]),

    # TOOL
    ("Docker", "TOOL", ["docker"]),
    ("Kubernetes", "TOOL", ["kubernetes", "k8s"]),
    ("Airflow", "TOOL", ["airflow", "apache airflow"]),
    ("Spark", "TOOL", ["spark", "apache spark", "pyspark"]),
    ("Tableau", "TOOL", ["tableau"]),
    ("Power BI", "TOOL", ["power bi", "powerbi"]),
    ("Git", "TOOL", ["git"]),
    ("Excel", "TOOL", ["excel", "microsoft excel"]),
    ("Hadoop", "TOOL", ["hadoop"]),
    ("Jupyter", "TOOL", ["jupyter"]),
    ("Kafka", "TOOL", ["kafka", "apache kafka"]),
    ("Hive", "TOOL", ["hive", "apache hive"]),
    ("Databricks", "TOOL", ["databricks"]),
    ("Jira", "TOOL", ["jira"]),
    ("GitHub", "TOOL", ["github"]),
    ("GitLab", "TOOL", ["gitlab"]),
    ("Bitbucket", "TOOL", ["bitbucket"]),
    ("Terraform", "TOOL", ["terraform"]),
    ("Ansible", "TOOL", ["ansible"]),
    ("Linux", "TOOL", ["linux", "unix/linux", "unix"]),
    ("SAS", "TOOL", ["sas"]),
    ("Looker", "TOOL", ["looker", "looker studio"]),
    ("dbt", "TOOL", ["dbt"]),
    ("Alteryx", "TOOL", ["alteryx"]),
    ("Qlik", "TOOL", ["qlik"]),
    ("MLflow", "TOOL", ["mlflow"]),
    ("SAP", "TOOL", ["sap"]),

    # METHOD
    ("A/B Testing", "METHOD", ["a/b testing", "ab testing", "a/b test", "ab tests", "a/b experiments"]),
    ("Machine Learning", "METHOD", ["machine learning"]),
    ("Deep Learning", "METHOD", ["deep learning"]),
    ("NLP", "METHOD", ["nlp", "natural language processing"]),
    ("Computer Vision", "METHOD", ["computer vision", "machine vision"]),
    ("Statistics", "METHOD", ["statistics", "statistical analysis"]),
    ("Data Visualization", "METHOD", ["data visualization", "data viz"]),
    ("ETL", "METHOD", ["etl"]),
    ("Time Series", "METHOD", ["time series"]),
    ("Agile", "METHOD", ["agile", "agile development", "agile methodology"]),
    ("Scrum", "METHOD", ["scrum"]),
    ("CI/CD", "METHOD", ["ci/cd", "cicd", "continuous integration", "ci / cd"]),
    ("Data Warehousing", "METHOD", ["data warehousing", "data warehouse"]),
    ("Data Governance", "METHOD", ["data governance"]),
    ("Data Mining", "METHOD", ["data mining"]),
    ("Data Modeling", "METHOD", ["data modeling", "data modelling"]),
    ("Big Data", "METHOD", ["big data"]),
    ("Data Pipelines", "METHOD", ["data pipelines", "data pipeline"]),
    ("Business Intelligence", "METHOD", ["business intelligence", "microsoft bi"]),
    ("DevOps", "METHOD", ["devops"]),
    ("Feature Engineering", "METHOD", ["feature engineering"]),
    ("Hypothesis Testing", "METHOD", ["hypothesis testing"]),
    ("Regression", "METHOD", ["regression"]),
    ("Classification", "METHOD", ["classification"]),
    ("Clustering", "METHOD", ["clustering"]),
    ("MapReduce", "METHOD", ["mapreduce", "map reduce"]),
]
