"""
This script is designed to be run on AWS EMR (Elastic MapReduce), leveraging the power of Apache Spark and Hadoop.
It processes a large dataset of one milion movie ratings to find similar movies based on user ratings using cosine similarity.

AWS EMR Configuration:
- AWS EMR is configured to handle Apache Spark applications, allowing for scalable and efficient processing of big data tasks.
- The default Spark configuration provided by EMR is utilized, which can be overridden by command-line options during the spark-submit process.

Memory Configuration:
- The default executor memory in EMR may be set to 512MB, which is insufficient for processing datasets with millions of ratings.
- It is recommended to increase the executor memory to at least 1GB for this task by using the '--executor-memory 1g' option in the spark-submit command.
- Example command to run this script on EMR's master node: spark-submit --executor-memory 1g MovieSimilarities1M.py <movie_id>
- This adjustment is crucial as executors may fail if running with default memory settings, especially when the dataset is large.
"""

"""
AWS EMR and Amazon S3 Integration for Spark Applications:
---------------------------------------------------------
To run this script we need the configuration and execution of a Spark application on AWS EMR (Elastic MapReduce),
leveraging Amazon S3 for efficient data storage and access. AWS EMR provides a managed Hadoop framework using Apache Spark
that simplifies running big data frameworks.

Setup and Configuration Process:
1. **AWS S3 Integration**:
   - Store your data on Amazon S3 to ensure it is accessible by EMR nodes.
   - Use the format 's3://bucket-name/path/to/file' to specify file paths in your Spark application.
   - Ensure proper IAM roles and permissions are set to allow EMR instances to access S3 resources.

2. **Spin up an EMR Cluster**:
   - Use the AWS Management Console to configure and launch an EMR cluster.
   - Select appropriate instance types and sizes, depending on your computational needs (e.g., m5.xlarge for balanced compute, memory, and networking).
   - Opt for On-Demand or Spot instances based on cost efficiency and availability requirements.

3. **Cluster Configuration**:
   - Configure the cluster with high availability options to ensure robustness and fault tolerance.
   - Select a mix of core and task nodes to optimize both computing power and cost.
   - Choose additional security and networking configurations to comply with organizational policies.

4. **Security and Access**:
   - Set up EC2 security groups and key pairs to secure access to the EMR cluster.
   - Define roles with necessary permissions for accessing S3 buckets and managing EMR resources.

5. **Running the Spark Application**:
   - Deploy your Spark application using the 'spark-submit' command from the master node of your EMR cluster.
   - Monitor execution through the EMR console and log outputs for debugging and verification purposes.

6. **Resource and Cost Management**:
   - Consider setting auto-scaling policies to adjust the cluster size according to the workload.
   - Enable logging to S3 to keep a record of the job execution and for auditing purposes.
   - Ensure to terminate the cluster or set auto-termination policies to minimize costs.
"""

"""
Connecting to Amazon EMR Primary Node via SSH:
----------------------------------------------
This script can be managed and interacted with directly via SSH if running on an Amazon EMR cluster.
To establish a secure SSH connection, use PuTTY on Windows by configuring the session with the cluster's public DNS
and appropriate private key file. This connection allows for running interactive queries, examining log files, and managing the cluster.
"""

import logging
import sys
from typing import Dict, List, Tuple, Iterable
from pyspark import SparkConf, SparkContext
from math import sqrt


def configure_logging() -> None:
    """Configures the logging settings for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_movie_names() -> Dict[int, str]:
    """Loads movie names from a file.

    Reads movie data from 'movies.dat' and maps movie IDs to movie titles.

    Returns:
        Dict[int, str]: A dictionary mapping movie IDs to movie titles.
    """
    movie_names = {}
    try:
        with open("movies.dat", encoding="ascii", errors="ignore") as f:
            for line in f:
                fields = line.split("::")
                movie_names[int(fields[0])] = fields[1]
    except FileNotFoundError:
        logging.error("movies.dat file not found.")
    return movie_names


def make_pairs(
    user_ratings: Tuple[int, Tuple[Tuple[int, float], Tuple[int, float]]]
) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """Creates a pair of movies and their ratings from user ratings.

    Args:
        user_ratings (Tuple[int, Tuple[Tuple[int, float], Tuple[int, float]]]): User ratings containing two movie-rating pairs.

    Returns:
        Tuple[Tuple[int, int], Tuple[float, float]]: A tuple containing a pair of movies and their corresponding ratings.
    """
    ratings = user_ratings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return (movie1, movie2), (rating1, rating2)


def filter_duplicates(
    user_ratings: Tuple[int, Tuple[Tuple[int, float], Tuple[int, float]]]
) -> bool:
    """Filters out duplicate movie pairs from user ratings.

    Args:
        user_ratings (Tuple[int, Tuple[Tuple[int, float], Tuple[int, float]]]): User ratings containing two movie-rating pairs.

    Returns:
        bool: True if the first movie ID is less than the second, otherwise False.
    """
    ratings = user_ratings[1]
    (movie1, _), (movie2, _) = ratings
    return movie1 < movie2


def compute_cosine_similarity(
    rating_pairs: Iterable[Tuple[float, float]]
) -> Tuple[float, int]:
    """Computes the cosine similarity between pairs of movie ratings.

    Args:
        rating_pairs (Iterable[Tuple[float, float]]): An iterable of tuples containing ratings for two movies.

    Returns:
        Tuple[float, int]: A tuple containing the similarity score and the number of pairs.
    """
    num_pairs = 0
    sum_xx = sum_yy = sum_xy = 0.0
    for rating_x, rating_y in rating_pairs:
        sum_xx += rating_x * rating_x
        sum_yy += rating_y * rating_y
        sum_xy += rating_x * rating_y
        num_pairs += 1

    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = (sum_xy / denominator) if denominator else 0.0

    return score, num_pairs


def main(movie_id: int) -> None:
    """Main function to compute movie similarities and print top similar movies.

    This function performs the following steps:
    1. Configures logging for the script.
    2. Initializes a Spark context for distributed computing.
    3. Loads movie names from a local file.
    4. Reads movie ratings from an AWS S3 bucket using Spark's S3 support.
    5. Computes movie similarities using cosine similarity.
    6. Filters and sorts similar movies based on quality thresholds.
    7. Outputs the top similar movies.

    Args:
        movie_id (int): The movie ID for which to find similar movies.
    """
    configure_logging()

    # Step 2: Initialize Spark configuration and context
    conf = SparkConf().setAppName("MovieSimilarities")
    sc = SparkContext(conf=conf)
    logging.info("Spark context initialized.")

    # Step 3: Load movie names from a local file
    logging.info("Loading movie names...")
    name_dict = load_movie_names()

    # Step 4: Read movie ratings from AWS S3
    # Highlights the use of AWS S3 for distributed data processing
    data = sc.textFile("s3n://sundog-spark/ml-1m/ratings.dat")
    logging.info("Loaded movie ratings from AWS S3.")

    # Map ratings to key/value pairs: user ID => movie ID, rating
    ratings = data.map(lambda l: l.split("::")).map(
        lambda l: (int(l[0]), (int(l[1]), float(l[2])))
    )

    # Partition data and join ratings
    ratings_partitioned = ratings.partitionBy(100)
    joined_ratings = ratings_partitioned.join(ratings_partitioned)

    # Filter duplicates and map to movie pairs
    unique_joined_ratings = joined_ratings.filter(filter_duplicates)
    movie_pairs = unique_joined_ratings.map(make_pairs).partitionBy(100)

    # Compute similarities for movie pairs
    movie_pair_ratings = movie_pairs.groupByKey()
    movie_pair_similarities = movie_pair_ratings.mapValues(
        compute_cosine_similarity
    ).persist()

    # Save results
    movie_pair_similarities.sortByKey()
    movie_pair_similarities.saveAsTextFile("movie-sims")

    # Extract and display top 10 similar movies
    score_threshold = 0.97
    co_occurrence_threshold = 50

    filtered_results = movie_pair_similarities.filter(
        lambda pair_sim: (
            pair_sim[0][0] == movie_id or pair_sim[0][1] == movie_id
        )
        and pair_sim[1][0] > score_threshold
        and pair_sim[1][1] > co_occurrence_threshold
    )

    results = (
        filtered_results.map(lambda pair_sim: (pair_sim[1], pair_sim[0]))
        .sortByKey(ascending=False)
        .take(10)
    )

    logging.info(f"Top 10 similar movies for {name_dict[movie_id]}:")
    for sim, pair in results:
        similar_movie_id = pair[0] if pair[0] != movie_id else pair[1]
        logging.info(
            f"{name_dict[similar_movie_id]}\tscore: {sim[0]:.2f}\tstrength: {sim[1]}"
        )

    sc.stop()
    logging.info("Spark context stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            movieID = int(sys.argv[1])
            main(movieID)
        except ValueError:
            logging.error("Please provide a valid movie ID as an integer.")
    else:
        logging.error("Movie ID argument missing. Please provide a movie ID.")
