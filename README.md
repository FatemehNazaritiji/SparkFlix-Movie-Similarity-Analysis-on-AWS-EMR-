# SparkFlix: Movie Similarity Analysis on AWS EMR

## Project Overview
SparkFlix is a big data application that processes a large dataset of one milion movie ratings to identify similar movies based on user ratings using cosine similarity. This project is designed to run on AWS EMR, leveraging the computational power of Apache Spark and the distributed storage capabilities of Amazon S3.

This project is part of my learning from the Udemy course ["Taming Big Data with Apache Spark - Hands On!"](https://www.udemy.com/course/taming-big-data-with-apache-spark-hands-on/?couponCode=ST10MT8624).

## Features
- **Cosine Similarity Calculation**: Computes similarity scores between movies based on user ratings.
- **Scalable Architecture**: Utilizes AWS EMR to manage and scale Apache Spark jobs efficiently.
- **Data Integration**: Uses Amazon S3 for data storage, ensuring high availability and scalability.

## System Requirements
- AWS Account
- Access to AWS EMR and S3 services
- Apache Spark on AWS EMR
- Suitable EC2 instance types (e.g., m5.xlarge) for the EMR cluster

## Configuration and Setup
1. **AWS S3 Setup**:
   - Upload the dataset and the Spark script to an S3 bucket.
   - Ensure the EMR role has the necessary permissions to access S3 resources.

2. **EMR Cluster Configuration**:
   - Launch an EMR cluster from the AWS Management Console with Apache Spark installed.
   - Configure instance types and numbers based on computational needs.

3. **Security Configuration**:
   - Set up EC2 key pairs for SSH access to the EMR master node.
   - Configure security groups to ensure secure access to the cluster.

## Running the Application
1. **Connect to the EMR Master Node via SSH**:
   - Use an SSH client like PuTTY or terminal to connect using your EC2 key pair.
2. **Submit the Spark Job**:
   - Run the command below from the master node's terminal:
     ```bash
     spark-submit --executor-memory 1g s3://path-to-your-script/MovieSimilarities.py <movie_id>
     ```
   - Replace `<movie_id>` with the ID of the movie for which you want to find similarities.

## Monitoring and Logs
- Monitor the job execution through the EMR console.
- Access detailed logs stored in S3 for auditing and debugging purposes.

## Resource Management
- Consider enabling auto-scaling to manage computation resources based on workload.
- Set auto-termination policies to minimize costs after job completion.

## Termination
- Manually terminate the EMR cluster from the AWS Console or set up auto-termination policies to avoid unnecessary charges.

## Documentation and Help
- For more details on AWS EMR configuration and commands, visit the AWS official documentation.
- Check Spark documentation for details on Spark configurations and optimizations.

