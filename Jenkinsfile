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
                sh 'docker run --name bank_marketing_app_container -p 8501:8501 bank_marketing_app'
            }
        }

        stage('Test Application') {
            steps {
                // Get the container IP address and store it in an environment variable
                sh "CONTAINER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' bank_marketing_app_container) && echo \$CONTAINER_IP"
        
                // Use the container IP address to check if the app is running
                sh 'curl -I http://$CONTAINER_IP:8501'
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

