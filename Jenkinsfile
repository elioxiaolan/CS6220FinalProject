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
                sh 'docker run -d --name bank_marketing_app_container -p 8501:8501 bank_marketing_app'
            }
        }

        stage('Test Application') {
            steps {
                // (Optional) Add steps to verify the application is running correctly
                sh 'curl -I http://localhost:8501'
            }
        }
    }

    post {
        always {
            // Clean up: Stop and remove the Docker container
            sh 'docker stop bank_marketing_app_container || true'
            sh 'docker rm bank_marketing_app_container || true'
            // Optionally clean up the Docker image
            sh 'docker rmi bank_marketing_app || true'
            cleanWs() // Clean workspace after build
        }

        success {
            echo 'Docker image built, container started, and app is accessible!'
        }

        failure {
            echo 'Something went wrong with the build, container, or app access.'
        }
    }
}

