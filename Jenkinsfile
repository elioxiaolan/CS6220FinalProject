pipeline {
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                // Cloning the repository to our workspace
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                // Build the Docker image using the Dockerfile in the repository
                sh 'docker build -t bank_marketing_app .'
            }
        }

        stage('Run Docker Container') {
            steps {
                // Run the Docker container
                sh 'docker run -p 8501:8501 bank_marketing_app'
            }
        }

        stage('Test Application') {
            steps {
                // (Optional) Add steps to verify the application is running correctly
                sh 'curl -I http://localhost:8501'
            }
        }
    }
}

